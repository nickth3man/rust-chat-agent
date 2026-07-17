//! Transport client: builds requests, streams SSE responses, validates HTTP
//! status, and retries retryable failures with exponential backoff + jitter.
//!
//! Design notes:
//! - Streaming uses `"stream": true` and parses Server-Sent Events from the
//!   response body via `reqwest::Response::chunk()`. No extra SSE crate needed.
//! - Deltas are delivered through a caller-supplied callback (`FnMut(&str)`)
//!   so transport stays decoupled from rendering (testable, swappable).
//! - Retry only happens *before* any output is shown. Once a delta has been
//!   emitted, further errors become non-retryable to avoid duplicate output.

use crate::error::ChatError;
use crate::Message;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Per-attempt backoff base; doubled each attempt.
const BACKOFF_BASE: Duration = Duration::from_millis(500);
/// Upper bound on a single backoff delay.
const BACKOFF_MAX: Duration = Duration::from_secs(10);

// ── Wire types ──────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Default, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
}

// ── SSE parser ──────────────────────────────────────────────────────────

/// Buffers raw bytes and yields complete SSE events (separated by blank lines).
/// Uses a byte buffer so multi-byte UTF-8 sequences split across chunks are
/// not corrupted.
struct SseParser {
    buffer: Vec<u8>,
}

impl SseParser {
    fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Drain all complete events from the buffer. Each event is the raw text
    /// between blank-line separators.
    fn drain_events(&mut self) -> Vec<String> {
        let mut events = Vec::new();
        while let Some(pos) = find_subsequence(&self.buffer, b"\n\n") {
            let event_bytes: Vec<u8> = self.buffer.drain(..pos + 2).collect();
            // Drop the trailing separator.
            let body = &event_bytes[..event_bytes.len().saturating_sub(2)];
            if let Ok(s) = String::from_utf8(body.to_vec()) {
                events.push(s);
            }
        }
        events
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

/// Extract the content delta (if any) from a single SSE event.
/// Returns `Some(None)` to signal `[DONE]`.
fn parse_event(event: &str) -> Option<Option<String>> {
    for line in event.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                return Some(None);
            }
            if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                if let Some(choice) = chunk.choices.into_iter().next() {
                    if let Some(content) = choice.delta.content {
                        if !content.is_empty() {
                            return Some(Some(content));
                        }
                    }
                }
            }
        }
    }
    None
}

// ── Chat client ─────────────────────────────────────────────────────────

/// Decoupled transport + retry. Owns the HTTP client, credentials, and model.
/// Cloneable because `reqwest::Client` is cheaply cloneable (Arc internals).
#[derive(Clone)]
pub struct ChatClient {
    http: Client,
    api_key: String,
    model: String,
    max_retries: u32,
    api_url: String,
}

impl ChatClient {
    pub fn new(http: Client, api_key: String, model: String, max_retries: u32) -> Self {
        Self {
            http,
            api_key,
            model,
            max_retries,
            api_url: API_URL.to_string(),
        }
    }

    /// Override the API endpoint. Test-only; production uses the real OpenRouter URL.
    #[cfg(test)]
    fn with_api_url(mut self, api_url: &str) -> Self {
        self.api_url = api_url.to_string();
        self
    }

    /// Stream a completion, invoking `on_delta` for each text chunk as it
    /// arrives. Returns the full accumulated assistant message on success.
    ///
    /// Retries retryable failures (timeouts, 429, 5xx, network) with
    /// exponential backoff + jitter, but only *before* any output is shown.
    pub async fn stream_complete<F>(&self, messages: &[Message], mut on_delta: F) -> Result<String, ChatError>
    where
        F: FnMut(&str),
    {
        let max_attempts = self.max_retries + 1;
        for attempt in 1..=max_attempts {
            let mut emitted = false;
            // Wrap the caller's callback so we can observe whether any output
            // was produced; if so, suppress retry on later failure.
            let result = self.try_once(messages, |delta| {
                emitted = true;
                on_delta(delta);
            })
            .await;

            match result {
                Ok(content) => return Ok(content),
                Err(err) => {
                    if !should_retry(emitted, &err, attempt, max_attempts) {
                        return Err(err);
                    }
                    // Backoff before the next attempt.
                    let delay = backoff_delay(attempt, &err);
                    tokio::time::sleep(delay).await;
                }
            }
        }
        // Loop exits only via return; this is unreachable but keeps types honest.
        Err(ChatError::Network("exhausted retries".into()))
    }

    /// A single attempt: send the request, validate status, parse the stream.
    async fn try_once<F>(&self, messages: &[Message], mut on_delta: F) -> Result<String, ChatError>
    where
        F: FnMut(&str),
    {
        let response = self
            .http
            .post(&self.api_url)
            .bearer_auth(&self.api_key)
            .json(&ChatRequest {
                model: &self.model,
                messages,
                stream: true,
            })
            .send()
            .await
            .map_err(|e| classify_reqwest_error(&e))?;

        // Recommendation A: validate HTTP status *before* parsing the body.
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(classify_status(status.as_u16(), body));
        }

        // Stream the body and parse SSE events.
        let mut parser = SseParser::new();
        let mut content = String::new();
        let mut stream_started = false;

        let mut response = response;

        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| {
                if stream_started {
                    ChatError::StreamInterrupted(e.to_string())
                } else {
                    classify_reqwest_error(&e)
                }
            })?
        {
            parser.push(&chunk);
            for event in parser.drain_events() {
                match parse_event(&event) {
                    Some(None) => {
                        // [DONE] — stream complete.
                        if content.trim().is_empty() {
                            return Err(ChatError::EmptyResponse);
                        }
                        return Ok(content);
                    }
                    Some(Some(delta)) => {
                        stream_started = true;
                        on_delta(&delta);
                        content.push_str(&delta);
                    }
                    None => { /* keepalive or non-content event */ }
                }
            }
        }

        // Stream ended without an explicit [DONE]; accept what we have.
        if content.trim().is_empty() {
            Err(ChatError::EmptyResponse)
        } else {
            Ok(content)
        }
    }
}

// ── Error classification ───────────────────────────────────────────────

/// Pure retry decision: retry only when no output has been emitted yet,
/// the error is retryable, and attempts remain. Extracted for testability.
fn should_retry(emitted: bool, err: &ChatError, attempt: u32, max_attempts: u32) -> bool {
    !emitted && err.is_retryable() && attempt < max_attempts
}

fn classify_reqwest_error(e: &reqwest::Error) -> ChatError {
    if e.is_timeout() {
        ChatError::Timeout
    } else {
        ChatError::Network(e.to_string())
    }
}

fn classify_status(status: u16, body: String) -> ChatError {
    match status {
        401 | 403 => ChatError::Auth(body),
        400 => ChatError::InvalidRequest(body),
        408 => ChatError::Timeout,
        429 => ChatError::RateLimit { retry_after: None },
        s if (500..600).contains(&s) => ChatError::Server { status: s, body },
        _ => ChatError::Server { status, body },
    }
}

/// Exponential backoff with jitter: `min(base * 2^(attempt-1), max) + jitter`.
/// Jitter is derived from the system clock to avoid a `rand` dependency.
fn backoff_delay(attempt: u32, err: &ChatError) -> Duration {
    // Honor a server-provided Retry-After for 429s when present.
    if let ChatError::RateLimit {
        retry_after: Some(d),
    } = err
    {
        return *d;
    }
    let exp = attempt.saturating_sub(1);
    let base_ms = BACKOFF_BASE.as_millis() as u64;
    let exp_factor = 1u64.checked_shl(exp).unwrap_or(u64::MAX);
    let raw = base_ms.saturating_mul(exp_factor);
    let capped = raw.min(BACKOFF_MAX.as_millis() as u64);
    // Pseudo-jitter from wall-clock nanos (0..=250ms).
    let jitter = jitter_millis();
    Duration::from_millis(capped + jitter)
}

fn jitter_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    nanos % 250
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_parser_yields_complete_events() {
        let mut p = SseParser::new();
        // First push: an incomplete event (no trailing blank line yet).
        p.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"He\"}}]}\n");
        assert!(p.drain_events().is_empty()); // incomplete: needs blank line
        // Second push completes the event separator.
        p.push(b"\n");
        let evs = p.drain_events();
        assert_eq!(evs.len(), 1);
    }

    #[test]
    fn sse_parser_handles_split_at_arbitrary_boundary() {
        // A complete event split at an arbitrary byte offset still yields
        // exactly one event once the trailing separator arrives.
        let mut p = SseParser::new();
        let full = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        p.push(&full[..full.len() - 3]);
        p.push(&full[full.len() - 3..]);
        let evs = p.drain_events();
        assert_eq!(evs.len(), 1);
    }

    #[test]
    fn parse_event_extracts_delta() {
        let ev = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}";
        assert_eq!(parse_event(ev), Some(Some("Hello".to_string())));
    }

    #[test]
    fn parse_event_done_signal() {
        let ev = "data: [DONE]";
        assert_eq!(parse_event(ev), Some(None));
    }

    #[test]
    fn parse_event_ignores_non_content_lines() {
        let ev = ": ping\ndata: {\"choices\":[]}";
        assert_eq!(parse_event(ev), None);
    }

    #[test]
    fn classify_status_auth() {
        assert!(matches!(
            classify_status(401, "no".into()),
            ChatError::Auth(_)
        ));
        assert!(matches!(
            classify_status(403, "no".into()),
            ChatError::Auth(_)
        ));
    }

    #[test]
    fn classify_status_rate_limit() {
        assert!(matches!(
            classify_status(429, "slow".into()),
            ChatError::RateLimit { .. }
        ));
    }

    #[test]
    fn classify_status_server() {
        assert!(matches!(
            classify_status(503, "down".into()),
            ChatError::Server { status: 503, .. }
        ));
    }

    #[test]
    fn classify_status_invalid_request() {
        assert!(matches!(
            classify_status(400, "bad".into()),
            ChatError::InvalidRequest(_)
        ));
        assert!(!classify_status(400, "bad".into()).is_retryable());
    }

    #[test]
    fn backoff_grows_then_caps() {
        let e = ChatError::Network("x".into());
        let d1 = backoff_delay(1, &e);
        let d5 = backoff_delay(5, &e);
        let d10 = backoff_delay(10, &e);
        assert!(d5 > d1, "backoff should grow: {:?} <= {:?}", d5, d1);
        // Capped around BACKOFF_MAX + jitter.
        assert!(d10 <= BACKOFF_MAX + Duration::from_millis(250));
    }

    #[test]
    fn backoff_honors_retry_after() {
        let e = ChatError::RateLimit {
            retry_after: Some(Duration::from_secs(7)),
        };
        assert_eq!(backoff_delay(1, &e), Duration::from_secs(7));
    }

    // ── Retry decision logic (recommendation C, exhaustive) ──

    #[test]
    fn should_retry_when_retryable_unemitted_attempts_remain() {
        let e = ChatError::Network("conn refused".into());
        assert!(should_retry(false, &e, 1, 3));
        assert!(should_retry(false, &e, 2, 3));
    }

    #[test]
    fn should_not_retry_after_output_emitted() {
        // Even a retryable error must not retry once deltas were shown.
        let e = ChatError::Network("conn refused".into());
        assert!(!should_retry(true, &e, 1, 3));
    }

    #[test]
    fn should_not_retry_non_retryable_errors() {
        assert!(!should_retry(false, &ChatError::Auth("k".into()), 1, 3));
        assert!(!should_retry(false, &ChatError::InvalidRequest("b".into()), 1, 3));
        assert!(!should_retry(false, &ChatError::EmptyResponse, 1, 3));
    }

    #[test]
    fn should_not_retry_on_final_attempt() {
        let e = ChatError::Network("conn refused".into());
        assert!(should_retry(false, &e, 2, 3)); // one left
        assert!(!should_retry(false, &e, 3, 3)); // last attempt
        assert!(!should_retry(false, &e, 4, 3)); // over budget
    }

    // ── End-to-end retry loop against an unreachable host ──
    // These prove the retry *wiring* (not just the math): the loop actually
    // re-invokes the transport, accumulates backoff, and stops cleanly.

    fn unreachable_client(max_retries: u32) -> ChatClient {
        let http = Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap();
        // Port 9 (discard) / high closed port: connection refused → Network error.
        ChatClient::new(http, "key".into(), "model".into(), max_retries)
            .with_api_url("http://127.0.0.1:9/chat")
    }

    #[tokio::test]
    async fn no_retry_when_disabled_fails_fast() {
        let client = unreachable_client(0);
        let start = std::time::Instant::now();
        let result = client.stream_complete(&[], |_| {}).await;
        let elapsed = start.elapsed();
        assert!(result.is_err(), "expected failure");
        // No retries ⇒ no backoff sleep; only connection-refused time.
        assert!(
            elapsed < Duration::from_secs(2),
            "max_retries=0 should fail fast, took {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn retries_accumulate_backoff_then_fail() {
        let client = unreachable_client(2); // 3 attempts, 2 backoffs
        let start = std::time::Instant::now();
        let result = client.stream_complete(&[], |_| {}).await;
        let elapsed = start.elapsed();
        assert!(result.is_err(), "expected failure after retries");
        // Backoffs for attempts 1 and 2: ~500ms + ~1000ms base (+ jitter each).
        // Minimum accumulated sleep is well over a single attempt's time.
        assert!(
            elapsed >= Duration::from_millis(1400),
            "expected backoff to accumulate across retries, took {:?}",
            elapsed
        );
        assert!(
            elapsed <= Duration::from_secs(8),
            "unexpectedly slow, took {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn retry_exhaustion_returns_retryable_error_kind() {
        // The final error surfaced should be the retryable transport error
        // (Network or Timeout), confirming we didn't mask it on the last attempt.
        let client = unreachable_client(1);
        let err = client.stream_complete(&[], |_| {}).await.unwrap_err();
        assert!(
            matches!(err, ChatError::Network(_) | ChatError::Timeout),
            "expected Network/Timeout, got {:?}",
            err
        );
    }
}
