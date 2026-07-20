// Shared HTTP POST + bounded-retry plumbing, used by both the OpenRouter
// chat-completions call (`llm.rs`) and the Firecrawl search call
// (`search.rs`). Retry classification, backoff, and the journal path are all
// parameters here rather than globals, so the retry loop can be driven from
// tests (wiremock servers, `skip_sleep`, injected `max_attempts`) without any
// billed network calls.

use crate::journal::journal;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

/// One POST attempt: send -> check status -> parse JSON. Wrapped by
/// `post_json` in a bounded retry loop. Kept separate so the retry
/// classification can inspect a single error in isolation.
///
/// Note: HTTP `Retry-After` on 429 responses is NOT honored — by the time
/// `error_for_status()` converts the response to an error, the headers are
/// consumed and unavailable. We retry with fixed exponential backoff instead.
pub async fn try_post_json(
    client: &reqwest::Client,
    url: &str,
    bearer_key: &str,
    body: &Value,
) -> Result<Value> {
    Ok(client
        .post(url)
        .bearer_auth(bearer_key)
        .json(body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

/// Walk the anyhow cause chain for the first underlying `reqwest::Error`.
pub fn find_reqwest_error(err: &anyhow::Error) -> Option<&reqwest::Error> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<reqwest::Error>())
}

/// Whether an error returned by `try_post_json` should be retried. Retries on:
///
///   - timeouts and connection failures (`is_timeout`, `is_connect`)
///   - HTTP 429 (Too Many Requests) and any 5xx server error
///
/// Does NOT retry on:
///
///   - other 4xx responses (400/401/403/404/...): deterministic, money-wasting
///   - body-decode failures (`is_decode`): response shape changed, deterministic
pub fn is_retryable_reqwest_error(rerr: &reqwest::Error) -> bool {
    if rerr.is_timeout() || rerr.is_connect() {
        return true;
    }
    if rerr.is_decode() {
        return false;
    }
    if let Some(status) = rerr.status() {
        return crate::is_retryable_status(status.as_u16());
    }
    false
}

pub fn is_retryable_post_json_error(err: &anyhow::Error) -> bool {
    find_reqwest_error(err).is_some_and(is_retryable_reqwest_error)
}

/// Short label for the journal entry describing why a `post_json` attempt
/// is being retried. Prefers timeout/connect/status labels when present.
pub fn post_json_retry_reason(err: &anyhow::Error) -> String {
    // Walk the full chain (not just the first reqwest error) so a decode
    // failure wrapping a later status still surfaces the status label.
    for cause in err.chain() {
        if let Some(rerr) = cause.downcast_ref::<reqwest::Error>() {
            if rerr.is_timeout() {
                return "timeout".to_string();
            }
            if rerr.is_connect() {
                return "connect".to_string();
            }
            if let Some(status) = rerr.status() {
                return format!("status {}", status.as_u16());
            }
        }
    }
    "unknown".to_string()
}

/// Backoff duration for a 0-indexed attempt number (250, 500, 1000, 2000,
/// 4000 ms, capped). Delegates to `answerbot::backoff_ms` for the schedule.
pub fn backoff_duration(attempt: u32) -> Duration {
    Duration::from_millis(crate::backoff_ms(attempt))
}

/// Sleep for the backoff of `attempt_0based`, or not at all when
/// `skip_sleep` is set (tests).
pub async fn sleep_backoff(attempt_0based: u32, skip_sleep: bool) {
    let d = if skip_sleep {
        Duration::ZERO
    } else {
        backoff_duration(attempt_0based)
    };
    tokio::time::sleep(d).await;
}

/// Journal a retry event and sleep for the backoff of the prior attempt.
/// `attempt` is 1-based (matches journal fields). Shared by the LLM/rewrite
/// retry loops in `llm.rs`; `post_json` journals an extra `url` field itself
/// and calls `sleep_backoff` directly.
pub async fn journal_retry_and_sleep(
    journal_path: &str,
    event: &str,
    attempt: u32,
    reason: &str,
    skip_sleep: bool,
) {
    journal(
        journal_path,
        json!({
            "event": event,
            "attempt": attempt,
            "reason": reason,
        }),
    );
    sleep_backoff(attempt.saturating_sub(1), skip_sleep).await;
}

/// POST a JSON body with bearer auth, return the parsed JSON response.
/// Shared by the `OpenRouter` chat-completions call and the Firecrawl search
/// call so HTTP-status and timeout handling lives in one place. Retries
/// transient failures (timeout, connect, 429, 5xx) with exponential backoff
/// up to `max_attempts`. Non-retryable errors (other 4xx, body-decode
/// failures) propagate immediately. Every retry is journaled to
/// `journal_path`. `skip_sleep` swaps the exponential backoff for a
/// zero-duration sleep — used by tests so the retry loop stays fast.
///
/// `max_attempts == 0` performs no attempts and falls through to the final
/// `bail!` below; this makes that branch testable without needing to induce
/// exactly-`u32::MAX`-many failures.
#[allow(clippy::too_many_arguments)]
pub async fn post_json(
    client: &reqwest::Client,
    url: &str,
    bearer_key: &str,
    body: &Value,
    journal_path: &str,
    max_attempts: u32,
    skip_sleep: bool,
) -> Result<Value> {
    for attempt in 0..max_attempts {
        match try_post_json(client, url, bearer_key, body).await {
            Ok(v) => return Ok(v),
            // Non-retryable failures (deterministic 4xx, body-decode) go
            // through unchanged — we never wasted a billed retry on them.
            Err(e) if !is_retryable_post_json_error(&e) => return Err(e),
            Err(e) if attempt + 1 >= max_attempts => {
                return Err(e).context(format!("after {} attempts", attempt + 1));
            }
            Err(e) => {
                journal(
                    journal_path,
                    json!({
                        "event": "network_retry",
                        "attempt": attempt + 1,
                        "url": url,
                        "reason": post_json_retry_reason(&e),
                    }),
                );
                sleep_backoff(attempt, skip_sleep).await;
            }
        }
    }
    bail!("internal: network retry loop exited without returning")
}
