use std::error::Error;
use std::fmt;
use std::time::Duration;

/// Errors produced by network-backed tools and providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolNetError {
    Network(String),
    Timeout,
    HttpStatus {
        status: u16,
        body: String,
        retry_after: Option<Duration>,
    },
    Parse(String),
    BodyTooLarge {
        limit: usize,
    },
    Content(String),
}

impl ToolNetError {
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Network(_) | Self::Timeout => true,
            Self::HttpStatus { status, .. } => *status == 429 || (500..600).contains(status),
            Self::Parse(_) | Self::BodyTooLarge { .. } | Self::Content(_) => false,
        }
    }

    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::HttpStatus { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Parse the delta-seconds form of Retry-After. HTTP-date values are left
    /// for the HTTP layer because they require a clock and date parser.
    pub fn parse_retry_after(value: Option<&str>) -> Option<Duration> {
        let seconds = value?.trim().parse::<u64>().ok()?;
        Some(Duration::from_secs(seconds))
    }
}

impl fmt::Display for ToolNetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Network(message) => write!(f, "network error: {message}"),
            Self::Timeout => write!(f, "network request timed out"),
            Self::HttpStatus { status, body, .. } => {
                write!(f, "provider returned HTTP {status}: {body}")
            }
            Self::Parse(message) => write!(f, "provider response parse failed: {message}"),
            Self::BodyTooLarge { limit } => write!(f, "response exceeded the {limit}-byte limit"),
            Self::Content(message) => write!(f, "unsupported provider content: {message}"),
        }
    }
}
impl Error for ToolNetError {}

#[derive(Debug)]
pub enum AppError {
    Config(String),
    MissingCredential { provider: String, env_var: String },
    RankFailed(String),
    SessionLog(String),
    Compact(String),
    Tool(ToolNetError),
    Internal(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "configuration error: {message}"),
            Self::MissingCredential { provider, env_var } => {
                write!(
                    f,
                    "provider {provider} requires non-empty environment variable {env_var}"
                )
            }
            Self::RankFailed(message) => write!(f, "ranking failed: {message}"),
            Self::SessionLog(message) => write!(f, "session log error: {message}"),
            Self::Compact(message) => write!(f, "compaction failed: {message}"),
            Self::Tool(error) => error.fmt(f),
            Self::Internal(message) => write!(f, "internal error: {message}"),
        }
    }
}
impl Error for AppError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tool(error) => Some(error),
            _ => None,
        }
    }
}
impl From<ToolNetError> for AppError {
    fn from(error: ToolNetError) -> Self {
        Self::Tool(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retry_after_delta_is_safe_and_precise() {
        assert_eq!(
            ToolNetError::parse_retry_after(Some(" 12 ")),
            Some(Duration::from_secs(12))
        );
        assert_eq!(ToolNetError::parse_retry_after(Some("tomorrow")), None);
        assert_eq!(
            ToolNetError::parse_retry_after(Some("999999999999999999999")),
            None
        );
    }
    #[test]
    fn retry_classification_matches_statuses() {
        assert!(
            ToolNetError::HttpStatus {
                status: 429,
                body: String::new(),
                retry_after: None
            }
            .is_retryable()
        );
        assert!(
            ToolNetError::HttpStatus {
                status: 503,
                body: String::new(),
                retry_after: None
            }
            .is_retryable()
        );
        assert!(
            !ToolNetError::HttpStatus {
                status: 404,
                body: String::new(),
                retry_after: None
            }
            .is_retryable()
        );
    }
}
