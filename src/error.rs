//! Typed errors for the chat agent, with retryability classification.
//!
//! Retryable: timeouts, rate limits, server errors, network failures
//! that occur *before* any streamed output has been shown.
//! Not retryable: auth failures, invalid requests, parse errors,
//! and any failure that happens after streaming has started (the user
//! would see duplicated output on retry).

use std::time::Duration;

/// Per-request error kind. `is_retryable()` drives the retry policy.
#[derive(Debug)]
pub enum ChatError {
    /// 401 / 403 — credentials or permissions. Never succeeds on retry.
    Auth(String),
    /// 400 — malformed request, unsupported model/parameter. Never succeeds on retry.
    InvalidRequest(String),
    /// 429 — rate limited. `retry_after` honors a `Retry-After` header when present.
    RateLimit { retry_after: Option<Duration> },
    /// Request exceeded the configured timeout before any response arrived.
    Timeout,
    /// 5xx — upstream provider/server fault.
    Server { status: u16, body: String },
    /// Transport-level failure (connection reset, DNS, etc.) before streaming began.
    Network(String),
    /// A failure that occurred *after* streamed output was already displayed.
    /// Never retried, because retrying would duplicate visible output.
    StreamInterrupted(String),
    /// Response body could not be parsed as the expected shape.
    #[allow(dead_code)]
    Parse(String),
    /// API returned a valid response with no usable choice.
    EmptyResponse,
}

impl ChatError {
    /// Whether the retry loop should attempt this request again.
    pub fn is_retryable(&self) -> bool {
        match self {
            ChatError::Auth(_) => false,
            ChatError::InvalidRequest(_) => false,
            ChatError::EmptyResponse => false,
            ChatError::StreamInterrupted(_) => false,
            ChatError::Parse(_) => false,
            ChatError::RateLimit { .. } => true,
            ChatError::Timeout => true,
            ChatError::Server { .. } => true,
            ChatError::Network(_) => true,
        }
    }
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Auth(msg) => write!(f, "authentication failed (401/403): {msg}"),
            ChatError::InvalidRequest(msg) => write!(f, "invalid request (400): {msg}"),
            ChatError::RateLimit { retry_after } => match retry_after {
                Some(d) => write!(f, "rate limited (429); retry after {:?}", d),
                None => write!(f, "rate limited (429)"),
            },
            ChatError::Timeout => write!(f, "request timed out"),
            ChatError::Server { status, body } => {
                write!(f, "OpenRouter returned {status}: {body}")
            }
            ChatError::Network(msg) => write!(f, "network error: {msg}"),
            ChatError::StreamInterrupted(msg) => {
                write!(f, "stream interrupted after partial output: {msg}")
            }
            ChatError::Parse(msg) => write!(f, "failed to parse response: {msg}"),
            ChatError::EmptyResponse => write!(f, "OpenRouter returned no choices"),
        }
    }
}

impl std::error::Error for ChatError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_is_not_retryable() {
        assert!(!ChatError::Auth("bad key".into()).is_retryable());
    }

    #[test]
    fn invalid_request_is_not_retryable() {
        assert!(!ChatError::InvalidRequest("no model".into()).is_retryable());
    }

    #[test]
    fn rate_limit_is_retryable() {
        assert!(ChatError::RateLimit { retry_after: None }.is_retryable());
        assert!(ChatError::RateLimit {
            retry_after: Some(Duration::from_secs(5))
        }
        .is_retryable());
    }

    #[test]
    fn server_errors_are_retryable() {
        assert!(ChatError::Server {
            status: 503,
            body: "unavailable".into()
        }
        .is_retryable());
    }

    #[test]
    fn network_is_retryable() {
        assert!(ChatError::Network("connection reset".into()).is_retryable());
    }

    #[test]
    fn stream_interrupted_is_not_retryable() {
        // Partial output already shown; retrying would duplicate it.
        assert!(!ChatError::StreamInterrupted("mid-stream".into()).is_retryable());
    }

    #[test]
    fn timeout_is_retryable() {
        assert!(ChatError::Timeout.is_retryable());
    }

    #[test]
    fn parse_and_empty_are_not_retryable() {
        assert!(!ChatError::Parse("bad json".into()).is_retryable());
        assert!(!ChatError::EmptyResponse.is_retryable());
    }
}
