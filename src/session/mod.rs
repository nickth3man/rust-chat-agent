//! The process-local session journal.  A journal is deliberately append-only in
//! memory and is replaced on disk after each logical mutation.

use crate::config::SessionConfig;
use crate::contracts::error::AppError;
use crate::contracts::provenance::TurnProvenance;
use crate::contracts::session::{
    LogicalEvent, ResultSummary, SessionDocument, SessionMetadata, ToolResultSummary,
    TranscriptEntry, TranscriptRole,
};
use crate::contracts::types::SearchHit;
use crate::tools::meta_search::SearchActivity;
use serde_json::{Map, Value};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::NamedTempFile;
use tokio::sync::Mutex;

const FORMAT: &str = "openrouter-chat-session";
const VERSION: u32 = 1;
// LogicalEvent predates optional retry counts and cannot represent None. This
// value preserves unknown rather than presenting it as a real retry count.
const UNKNOWN_RETRY_COUNT: u32 = u32::MAX;

#[derive(Clone)]
pub struct SessionLogger {
    path: Arc<PathBuf>,
    secrets: Arc<Vec<String>>,
    document: Arc<Mutex<SessionDocument>>,
}

impl SessionLogger {
    pub fn new<I, S>(config: SessionConfig, secrets: I) -> Result<Self, AppError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        std::fs::create_dir_all(&config.log_directory)
            .map_err(|e| session_error("create log directory", e))?;
        let now = timestamp();
        let id = format!("{}-{}", std::process::id(), unique_id());
        let path = config.log_directory.join(format!("session-{id}.json"));
        let document = SessionDocument {
            metadata: SessionMetadata {
                format: FORMAT.into(),
                version: VERSION,
                session_id: id,
                created_at: now.clone(),
                updated_at: now,
            },
            transcript: Vec::new(),
            events: Vec::new(),
            provenance: Vec::new(),
        };
        let secrets = secrets
            .into_iter()
            .map(Into::into)
            .filter(|s| !s.is_empty())
            .collect();
        let logger = Self {
            path: Arc::new(path),
            secrets: Arc::new(secrets),
            document: Arc::new(Mutex::new(document)),
        };
        {
            let guard = logger
                .document
                .try_lock()
                .map_err(|_| AppError::SessionLog("initialization lock unavailable".into()))?;
            logger.write_document(&guard)?;
        }
        Ok(logger)
    }

    pub fn path(&self) -> &Path {
        self.path.as_path()
    }

    pub async fn snapshot(&self) -> SessionDocument {
        self.document.lock().await.clone()
    }

    pub async fn record_transcript(&self, mut entry: TranscriptEntry) -> Result<(), AppError> {
        entry.content = self.scrub(&entry.content);
        self.mutate(|doc| doc.transcript.push(entry)).await
    }
    pub async fn record_user(&self, timestamp: String, content: String) -> Result<(), AppError> {
        self.record_transcript(TranscriptEntry {
            role: TranscriptRole::User,
            content,
            created_at: timestamp,
        })
        .await
    }
    pub async fn record_assistant(
        &self,
        timestamp: String,
        content: String,
    ) -> Result<(), AppError> {
        self.record_transcript(TranscriptEntry {
            role: TranscriptRole::Assistant,
            content,
            created_at: timestamp,
        })
        .await
    }
    pub async fn record_tool(&self, timestamp: String, content: String) -> Result<(), AppError> {
        self.record_transcript(TranscriptEntry {
            role: TranscriptRole::Tool,
            content,
            created_at: timestamp,
        })
        .await
    }
    pub async fn record_event(&self, event: LogicalEvent) -> Result<(), AppError> {
        let event = self.scrub_value(
            serde_json::to_value(event)
                .map_err(|_| AppError::SessionLog("serialize event failed".into()))?,
        );
        let event = serde_json::from_value(event)
            .map_err(|_| AppError::SessionLog("sanitize event failed".into()))?;
        self.mutate(|doc| doc.events.push(event)).await
    }
    pub async fn record_provenance(&self, provenance: TurnProvenance) -> Result<(), AppError> {
        let value = self.scrub_value(
            serde_json::to_value(provenance)
                .map_err(|_| AppError::SessionLog("serialize provenance failed".into()))?,
        );
        let provenance = serde_json::from_value(value)
            .map_err(|_| AppError::SessionLog("sanitize provenance failed".into()))?;
        self.mutate(|doc| doc.provenance.push(provenance)).await
    }

    pub async fn record_search_activity(&self, activity: SearchActivity) -> Result<(), AppError> {
        match activity {
            SearchActivity::QueryStarted { query } => {
                self.record_event(LogicalEvent::User {
                    timestamp: timestamp(),
                    content: query,
                })
                .await
            }
            SearchActivity::ProviderStarted { provider } => {
                self.record_event(LogicalEvent::Provider {
                    timestamp: timestamp(),
                    provider,
                    state: crate::contracts::session::ProviderState::Started,
                    elapsed_ms: 0,
                    retry_count: 0,
                    error: None,
                    hit_count: 0,
                })
                .await
            }
            SearchActivity::ProviderResult {
                provider,
                elapsed_ms,
                hits,
                retry_count,
                normalized_hits,
            } => {
                let results = normalized_hits
                    .iter()
                    .map(|h| ResultSummary {
                        title: h.title.clone(),
                        snippet: h.snippet.clone(),
                        url: h.url.clone(),
                    })
                    .collect();
                let mut args = Map::new();
                args.insert("provider".into(), Value::String(provider.clone()));
                if retry_count.is_none() {
                    args.insert("retry_count_unknown".into(), Value::Bool(true));
                }
                args.insert(
                    "hits".into(),
                    Value::Array(normalized_hits.iter().map(hit_value).collect()),
                );
                self.record_event(LogicalEvent::Tool {
                    timestamp: timestamp(),
                    name: "search".into(),
                    arguments: Value::Object(args),
                    elapsed_ms,
                    retry_count: retry_count.unwrap_or(UNKNOWN_RETRY_COUNT),
                    error: None,
                    result: Some(ToolResultSummary {
                        hit_count: hits,
                        results,
                    }),
                })
                .await
            }
            SearchActivity::ProviderError {
                provider,
                elapsed_ms,
                error,
                retry_count,
            } => {
                self.record_event(LogicalEvent::Provider {
                    timestamp: timestamp(),
                    provider,
                    state: crate::contracts::session::ProviderState::Failed,
                    elapsed_ms,
                    retry_count: retry_count.unwrap_or(UNKNOWN_RETRY_COUNT),
                    error: Some(error),
                    hit_count: 0,
                })
                .await
            }
            SearchActivity::RankingStarted { candidates } => {
                self.record_event(LogicalEvent::Ranking {
                    timestamp: timestamp(),
                    query: format!("ranking_started candidates={candidates}"),
                    elapsed_ms: 0,
                    decisions: Vec::new(),
                    error: None,
                })
                .await
            }
            SearchActivity::RankingCompleted {
                elapsed_ms,
                selected,
                decisions,
            } => {
                self.record_event(LogicalEvent::Ranking {
                    timestamp: timestamp(),
                    query: format!("ranking_completed selected={selected}"),
                    elapsed_ms,
                    decisions,
                    error: None,
                })
                .await
            }
            SearchActivity::RankingFailed { elapsed_ms, error } => {
                self.record_event(LogicalEvent::Ranking {
                    timestamp: timestamp(),
                    query: String::new(),
                    elapsed_ms,
                    decisions: Vec::new(),
                    error: Some(error),
                })
                .await
            }
        }
    }

    async fn mutate<F>(&self, update: F) -> Result<(), AppError>
    where
        F: FnOnce(&mut SessionDocument),
    {
        let mut doc = self.document.lock().await;
        update(&mut doc);
        doc.metadata.updated_at = timestamp();
        self.write_document(&doc)
    }

    fn scrub(&self, input: &str) -> String {
        scrub_text(input, &self.secrets)
    }
    fn scrub_value(&self, value: Value) -> Value {
        match value {
            Value::String(s) => Value::String(self.scrub(&s)),
            Value::Array(a) => Value::Array(a.into_iter().map(|v| self.scrub_value(v)).collect()),
            Value::Object(o) => Value::Object(
                o.into_iter()
                    .map(|(k, v)| {
                        let value = if sensitive_key(&k) {
                            Value::String("[REDACTED]".into())
                        } else {
                            self.scrub_value(v)
                        };
                        (k, value)
                    })
                    .collect(),
            ),
            v => v,
        }
    }
    fn write_document(&self, doc: &SessionDocument) -> Result<(), AppError> {
        let mut clean = serde_json::to_value(doc)
            .map_err(|_| AppError::SessionLog("serialize session failed".into()))?;
        clean = self.scrub_value(clean);
        let bytes = serde_json::to_vec_pretty(&clean)
            .map_err(|_| AppError::SessionLog("serialize session failed".into()))?;
        atomic_write(self.path(), &bytes).map_err(|e| session_error("write session", e))
    }
}

fn hit_value(hit: &SearchHit) -> Value {
    serde_json::to_value(hit).unwrap_or(Value::Null)
}
fn timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| format!("{}.{:09}Z", d.as_secs(), d.subsec_nanos()))
        .unwrap_or_else(|_| "0Z".into())
}
fn unique_id() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}
fn session_error(action: &str, error: io::Error) -> AppError {
    AppError::SessionLog(format!("{action} failed: {error}"))
}

fn scrub_text(input: &str, secrets: &[String]) -> String {
    let mut out = input.to_owned();
    for secret in secrets {
        if !secret.is_empty() {
            out = out.replace(secret, "[REDACTED]");
        }
    }
    for key in [
        "authorization",
        "x-subscription-token",
        "api-key",
        "api_key",
        "apikey",
        "access-token",
        "access_token",
        "accesstoken",
        "refresh-token",
        "refresh_token",
        "refreshtoken",
        "client-secret",
        "client_secret",
        "clientsecret",
        "private-key",
        "private_key",
        "signing-key",
        "signing_key",
        "encryption-key",
        "encryption_key",
        "password",
        "passwd",
        "credential",
        "credentials",
        "secret",
        "token",
    ] {
        let mut start = 0;
        while start < out.len() {
            let lower = out.to_ascii_lowercase();
            let Some(relative) = lower[start..].find(key) else {
                break;
            };
            let key_start = start + relative;
            let key_end = key_start + key.len();
            let boundary_before = key_start == 0
                || !out.as_bytes()[key_start - 1].is_ascii_alphanumeric()
                || (out.as_bytes()[key_start - 1] == b'\''
                    || out.as_bytes()[key_start - 1] == b'"');
            let boundary_after =
                key_end == out.len() || !out.as_bytes()[key_end].is_ascii_alphanumeric();
            if !boundary_before || !boundary_after {
                start = key_end;
                continue;
            }
            let mut cursor = key_end;
            let quote = out
                .as_bytes()
                .get(cursor)
                .copied()
                .filter(|b| *b == b'\'' || *b == b'"');
            if quote.is_some() {
                cursor += 1;
            }
            while out
                .as_bytes()
                .get(cursor)
                .is_some_and(|byte| byte.is_ascii_whitespace())
            {
                cursor += 1;
            }
            let Some(&delimiter) = out.as_bytes().get(cursor) else {
                break;
            };
            if !matches!(delimiter, b':' | b'=') {
                start = key_end;
                continue;
            }
            cursor += 1;
            while out
                .as_bytes()
                .get(cursor)
                .is_some_and(|byte| byte.is_ascii_whitespace())
            {
                cursor += 1;
            }
            let value_quote = out
                .as_bytes()
                .get(cursor)
                .copied()
                .filter(|b| *b == b'\'' || *b == b'"');
            let value_start = cursor + usize::from(value_quote.is_some());
            let is_header = key == "authorization" || key == "x-subscription-token";
            let mut value_end = if let Some(quote) = value_quote {
                out.as_bytes()[value_start..]
                    .iter()
                    .position(|byte| *byte == quote)
                    .map_or(value_start, |end| value_start + end)
            } else {
                let mut value_end = value_start;
                while let Some(&byte) = out.as_bytes().get(value_end) {
                    if byte.is_ascii_whitespace()
                        || matches!(byte, b'&' | b',' | b'"' | b'\'' | b'\n' | b'\r')
                    {
                        break;
                    }
                    value_end += 1;
                }
                value_end
            };
            // Authorization commonly has a scheme followed by a credential.
            if is_header
                && value_quote.is_none()
                && out
                    .as_bytes()
                    .get(value_end)
                    .is_some_and(|byte| byte.is_ascii_whitespace())
            {
                let mut second = value_end;
                while out
                    .as_bytes()
                    .get(second)
                    .is_some_and(|byte| byte.is_ascii_whitespace())
                {
                    second += 1;
                }
                let mut second_end = second;
                while let Some(&byte) = out.as_bytes().get(second_end) {
                    if byte.is_ascii_whitespace()
                        || matches!(byte, b'&' | b',' | b'"' | b'\'' | b'\n' | b'\r')
                    {
                        break;
                    }
                    second_end += 1;
                }
                if second > value_start {
                    value_end = second_end;
                }
            }
            if value_end > value_start {
                out.replace_range(value_start..value_end, "[REDACTED]");
            }
            start = value_start + "[REDACTED]".len();
        }
    }
    // Also cover a bearer token presented without an explicit JSON/header key.
    let mut start = 0;
    loop {
        let lower = out.to_ascii_lowercase();
        let Some(relative) = lower[start..].find("bearer") else {
            break;
        };
        let bearer = start + relative;
        let end = bearer + "bearer".len();
        if (bearer == 0 || !out.as_bytes()[bearer - 1].is_ascii_alphanumeric())
            && (end == out.len() || !out.as_bytes()[end].is_ascii_alphanumeric())
        {
            let mut value_start = end;
            while out
                .as_bytes()
                .get(value_start)
                .is_some_and(|b| b.is_ascii_whitespace())
            {
                value_start += 1;
            }
            let mut value_end = value_start;
            while out.as_bytes().get(value_end).is_some_and(|b| {
                !b.is_ascii_whitespace() && *b != b',' && *b != b'"' && *b != b'\''
            }) {
                value_end += 1;
            }
            if value_end > value_start {
                out.replace_range(value_start..value_end, "[REDACTED]");
                start = value_start + "[REDACTED]".len();
                continue;
            }
        }
        start = end;
    }
    out
}

fn sensitive_key(key: &str) -> bool {
    let mut normalized = String::with_capacity(key.len());
    for (index, character) in key.chars().enumerate() {
        if character.is_ascii_uppercase() && index != 0 {
            normalized.push('_');
        }
        normalized.push(character.to_ascii_lowercase());
    }
    let key = normalized.replace('-', "_");
    matches!(
        key.as_str(),
        "authorization"
            | "auth"
            | "api_key"
            | "apikey"
            | "access_token"
            | "refresh_token"
            | "id_token"
            | "client_secret"
            | "password"
            | "passwd"
            | "credential"
            | "credentials"
            | "secret"
            | "token"
            | "x_subscription_token"
            | "private_key"
            | "signing_key"
            | "encryption_key"
    ) || key.starts_with("auth_")
        || key.starts_with("credential_")
        || key.ends_with("_token")
        || key.ends_with("_secret")
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    atomic_write_inner(path, bytes, false)
}

fn atomic_write_inner(path: &Path, bytes: &[u8], fail_before_commit: bool) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "session path has no parent"))?;
    let mut temp = NamedTempFile::new_in(parent)?;
    temp.write_all(bytes)?;
    temp.as_file().sync_all()?;
    if fail_before_commit {
        return Err(io::Error::other("injected pre-commit failure"));
    }
    #[cfg(unix)]
    {
        temp.persist(path).map_err(|e| e.error)?;
    }
    #[cfg(windows)]
    {
        // Drop the open handle before asking Win32 to move or replace it.
        let temp_path = temp.into_temp_path();
        windows_commit(&temp_path, path)?;
        // ReplaceFileW/MoveFileExW owns the destination now. TempPath's drop
        // only attempts cleanup of the old temporary name.
    }
    Ok(())
}

#[cfg(windows)]
/// The only unsafe boundary: Windows has no rename-over-existing equivalent
/// with the required atomic visibility semantics. No directory fsync is claimed.
fn windows_commit(temp: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_WRITE_THROUGH, MoveFileExW, ReplaceFileW,
    };
    fn wide(p: &Path) -> Vec<u16> {
        p.as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }
    let t = wide(temp);
    let d = wide(destination);
    unsafe {
        if destination.exists()
            && ReplaceFileW(
                d.as_ptr(),
                t.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ) == 0
        {
            return Err(io::Error::last_os_error());
        }
        if !destination.exists()
            && MoveFileExW(t.as_ptr(), d.as_ptr(), MOVEFILE_WRITE_THROUGH) == 0
            && ReplaceFileW(
                d.as_ptr(),
                t.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ) == 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::provenance::{EvidenceEntry, TurnProvenance};
    use crate::contracts::session::{LogicalEvent, RankingDecision};
    use crate::contracts::types::{BackendKind, SearchHit};
    use std::collections::{BTreeMap, BTreeSet};
    use tempfile::tempdir;

    fn logger() -> (tempfile::TempDir, SessionLogger) {
        let dir = tempdir().unwrap();
        let config = SessionConfig {
            log_directory: dir.path().to_path_buf(),
            redact_credentials: true,
            redact_auth_headers: true,
        };
        let logger = SessionLogger::new(config, std::iter::empty::<String>()).unwrap();
        (dir, logger)
    }

    fn read_document(logger: &SessionLogger) -> SessionDocument {
        serde_json::from_str(&std::fs::read_to_string(logger.path()).unwrap()).unwrap()
    }

    #[test]
    fn initial_missing_file_is_valid_json() {
        let (_dir, logger) = logger();
        assert!(logger.path().is_file());
        let document = read_document(&logger);
        assert_eq!(document.metadata.format, FORMAT);
        assert_eq!(document.metadata.version, VERSION);
    }

    #[tokio::test]
    async fn repeated_replacements_remain_parseable_and_latest() {
        let (_dir, logger) = logger();
        for index in 0..8 {
            logger
                .record_user(format!("t{index}"), format!("message-{index}"))
                .await
                .unwrap();
            let document = read_document(&logger);
            assert_eq!(
                document.transcript.last().unwrap().content,
                format!("message-{index}")
            );
        }
    }

    #[tokio::test]
    async fn concurrent_records_retain_every_event() {
        let (_dir, logger) = logger();
        let mut tasks = Vec::new();
        for index in 0..32 {
            let clone = logger.clone();
            tasks.push(tokio::spawn(async move {
                clone
                    .record_user(index.to_string(), format!("event-{index}"))
                    .await
            }));
        }
        for task in tasks {
            task.await.unwrap().unwrap();
        }
        let document = logger.snapshot().await;
        assert_eq!(document.transcript.len(), 32);
        for index in 0..32 {
            assert!(
                document
                    .transcript
                    .iter()
                    .any(|entry| entry.content == format!("event-{index}"))
            );
        }
    }

    #[test]
    fn scrub_removes_assignments_without_redacting_ordinary_words() {
        let secrets = vec!["exact-secret".to_string()];
        let scrubbed = scrub_text(
            "Authorization: Bearer abc X-Subscription-Token: sub api_key=key token=tok exact-secret tokenization token ordinary",
            &secrets,
        );
        assert!(!scrubbed.contains("abc"));
        assert!(!scrubbed.contains("sub "));
        assert!(!scrubbed.contains("key "));
        assert!(!scrubbed.contains("token=tok"));
        assert!(scrubbed.contains("[REDACTED]"));
        assert!(scrubbed.contains("tokenization"));
        assert!(scrubbed.contains("token ordinary"));
        assert!(!scrubbed.contains("exact-secret"));
    }

    #[test]
    fn scrub_value_redacts_sensitive_keys_recursively() {
        let (_dir, logger) = logger();
        let value = serde_json::json!({
            "profile": {
                "api_key": "nested-api-secret",
                "items": [{"accessToken": "nested-token-secret", "label": "ordinary tokenization"}],
                "description": "ordinary content"
            }
        });
        let scrubbed = logger.scrub_value(value);
        assert_eq!(scrubbed["profile"]["api_key"], "[REDACTED]");
        assert_eq!(scrubbed["profile"]["items"][0]["accessToken"], "[REDACTED]");
        assert_eq!(
            scrubbed["profile"]["items"][0]["label"],
            "ordinary tokenization"
        );
        assert_eq!(scrubbed["profile"]["description"], "ordinary content");
    }

    #[test]
    fn scrub_text_handles_quoted_assignments_and_authorization_forms() {
        let scrubbed = scrub_text(
            r#"{"api_key":"third-party-secret", 'client_secret' = 'another-secret', Authorization: Bearer header-secret, "access_token" = "quoted-token"} Bearer standalone-secret"#,
            &[],
        );
        for secret in [
            "third-party-secret",
            "another-secret",
            "header-secret",
            "quoted-token",
            "standalone-secret",
        ] {
            assert!(!scrubbed.contains(secret), "secret leaked: {secret}");
        }
        assert!(scrubbed.matches("[REDACTED]").count() >= 5);
    }

    #[test]
    fn scrub_text_retains_configured_secret_redaction() {
        let scrubbed = scrub_text(
            r#"configured-secret ordinary tokenization {"value":"configured-secret"}"#,
            &["configured-secret".into()],
        );
        assert!(!scrubbed.contains("configured-secret"));
        assert!(scrubbed.contains("ordinary tokenization"));
    }

    #[tokio::test]
    async fn secrets_are_absent_but_ordinary_content_remains() {
        let dir = tempdir().unwrap();
        let config = SessionConfig {
            log_directory: dir.path().into(),
            redact_credentials: true,
            redact_auth_headers: true,
        };
        let logger = SessionLogger::new(config, ["credential-123".to_string()]).unwrap();
        logger
            .record_user(
                "now".into(),
                "ordinary tokenization credential-123 Authorization: Bearer abc token=xyz".into(),
            )
            .await
            .unwrap();
        let snapshot = serde_json::to_string(&logger.snapshot().await).unwrap();
        let file = std::fs::read_to_string(logger.path()).unwrap();
        for text in [&snapshot, &file] {
            assert!(!text.contains("credential-123"));
            assert!(!text.contains("Bearer abc"));
            assert!(!text.contains("xyz"));
            assert!(text.contains("ordinary tokenization"));
        }
    }

    #[tokio::test]
    async fn provider_result_keeps_all_hit_details() {
        let (_dir, logger) = logger();
        let mut metadata = BTreeMap::new();
        metadata.insert("published".into(), "2026-07-18".into());
        logger
            .record_search_activity(SearchActivity::ProviderResult {
                provider: "fixture".into(),
                elapsed_ms: 4,
                hits: 1,
                retry_count: None,
                normalized_hits: vec![SearchHit {
                    title: "title".into(),
                    url: "https://example.test".into(),
                    snippet: "snippet".into(),
                    published: None,
                    native_rank: Some(1),
                    native_score: None,
                    provider: "fixture".into(),
                    backend_kind: BackendKind::Json,
                    source_subtype: None,
                    metadata,
                }],
            })
            .await
            .unwrap();
        let json = std::fs::read_to_string(logger.path()).unwrap();
        for value in [
            "title",
            "snippet",
            "https://example.test",
            "2026-07-18",
            "retry_count_unknown",
        ] {
            assert!(json.contains(value), "missing {value}");
        }
    }

    #[tokio::test]
    async fn ranking_decisions_are_structured_and_start_is_not_an_error() {
        let (_dir, logger) = logger();
        logger
            .record_search_activity(SearchActivity::RankingStarted { candidates: 2 })
            .await
            .unwrap();
        let decision = RankingDecision {
            source_id: "source".into(),
            normalized_url: "https://example.test".into(),
            selected: true,
            decision: "keep".into(),
            score: Some(0.9),
        };
        logger
            .record_search_activity(SearchActivity::RankingCompleted {
                elapsed_ms: 9,
                selected: 1,
                decisions: vec![decision],
            })
            .await
            .unwrap();
        let document = logger.snapshot().await;
        assert!(matches!(
            &document.events[0],
            LogicalEvent::Ranking { error: None, .. }
        ));
        assert!(
            matches!(&document.events[1], LogicalEvent::Ranking { decisions, error: None, .. } if decisions.len() == 1)
        );
    }

    #[tokio::test]
    async fn transcript_and_provenance_round_trip() {
        let (_dir, logger) = logger();
        logger
            .record_assistant("a".into(), "answer".into())
            .await
            .unwrap();
        let provenance = TurnProvenance {
            turn_id: "turn".into(),
            entries: vec![EvidenceEntry {
                source_id: "s".into(),
                normalized_url: "https://example.test".into(),
                title: "T".into(),
                url: "https://example.test".into(),
                supporting_snippet: "S".into(),
                rank_decision: Some("keep".into()),
                provider_labels: BTreeSet::from(["fixture".into()]),
                source_subtypes: BTreeSet::new(),
            }],
        };
        logger.record_provenance(provenance.clone()).await.unwrap();
        let parsed = read_document(&logger);
        assert_eq!(parsed.transcript[0].content, "answer");
        assert_eq!(parsed.provenance, vec![provenance]);
    }

    #[test]
    fn injected_pre_commit_failure_keeps_previous_destination() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.json");
        atomic_write(&path, br#"{"old":true}"#).unwrap();
        let before = std::fs::read(&path).unwrap();
        assert!(atomic_write_inner(&path, br#"{"new":true}"#, true).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    // --- Characterization tests pinning scrub_text secret-redaction
    // invariants before any simplification. Each assertion locks a concrete
    // observable rule; behavior must remain identical through refactoring. ---

    #[test]
    fn scrub_text_pins_per_key_value_redaction() {
        // Each sensitive key form, in `key=value` shape, redacts its value.
        // (Value token is chosen to not collide with any key name or marker.)
        let value = "leakmark-77";
        for key in [
            "api_key",
            "apikey",
            "api-key",
            "access_token",
            "refresh_token",
            "client_secret",
            "password",
            "passwd",
            "credential",
            "credentials",
            "secret",
            "token",
            "authorization",
            "private_key",
            "encryption_key",
        ] {
            let input = format!("{key}={value}");
            let out = scrub_text(&input, &[]);
            assert!(!out.contains(value), "value leaked for key {key:?}: {out}");
            assert!(
                out.contains("[REDACTED]"),
                "no redaction marker for key {key:?}: {out}"
            );
        }
    }

    #[test]
    fn scrub_text_pins_quote_and_delimiter_forms() {
        // Quoted JSON form.
        assert!(scrub_text(r#"{"api_key":"sek"}"#, &[]).contains(r#""api_key":"[REDACTED]""#));
        // Single-quote key with '=' delimiter.
        assert!(scrub_text(r#"'api_key'='sek'"#, &[]).contains("[REDACTED]"));
        // Colon delimiter with surrounding whitespace.
        assert!(scrub_text("password : sek", &[]).contains("[REDACTED]"));
        // Whitespace, '&', ',', newline all terminate an unquoted value.
        for sep in ["&", ",", "\n"] {
            let input = format!("token=sek{sep}more");
            let out = scrub_text(&input, &[]);
            assert!(!out.contains("sek"), "value leaked past {sep:?}: {out}");
            assert!(
                out.contains("more"),
                "trailing content dropped past {sep:?}: {out}"
            );
        }
    }

    #[test]
    fn scrub_text_pins_boundary_and_case_invariants() {
        // Key glued to alphanumerics on either side is NOT a sensitive key.
        let out = scrub_text("myapi_key sek and api_keyed sek2", &[]);
        assert!(
            out.contains("myapi_key"),
            "prefixed key should be left intact: {out}"
        );
        assert!(
            out.contains("api_keyed"),
            "suffixed key should be left intact: {out}"
        );
        // Keys are matched case-insensitively.
        let mixed = scrub_text("API-KEY=sek TOKEN=sek2", &[]);
        assert!(
            !mixed.contains("sek") && !mixed.contains("sek2"),
            "case-insensitive match failed: {mixed}"
        );
        // The standalone word "tokenization"/"credentials-report" behavior:
        // "tokenization" must NOT be redacted as a "token" key (suffix boundary).
        let word = scrub_text("tokenization rules", &[]);
        assert!(
            word.contains("tokenization"),
            "ordinary word clobbered: {word}"
        );
    }

    #[test]
    fn scrub_text_pins_bearer_and_empty_secret_invariants() {
        // Bare Bearer scheme with no JSON/header key still redacts the credential.
        let bearer = scrub_text("Authorization: Bearer abc123def", &[]);
        assert!(
            !bearer.contains("abc123def"),
            "bearer credential leaked: {bearer}"
        );
        // Standalone bearer (no preceding key).
        let standalone = scrub_text("bearer tok-456", &[]);
        assert!(
            !standalone.contains("tok-456"),
            "standalone bearer leaked: {standalone}"
        );
        // Empty configured secrets are skipped entirely: they must NOT cause a
        // replace-all that inserts [REDACTED] between every character, and plain
        // text with no sensitive key is left byte-for-byte intact.
        let plain = scrub_text("plain text here", &["".to_string()]);
        assert_eq!(
            plain, "plain text here",
            "empty secret should be a no-op: {plain}"
        );
        // Key-based redaction still applies even when an empty secret is present.
        let with_key = scrub_text("api_key=sek", &["".to_string()]);
        assert!(
            !with_key.contains("sek"),
            "key redaction skipped: {with_key}"
        );
        assert!(
            with_key.contains("[REDACTED]"),
            "no redaction marker: {with_key}"
        );
    }
}
