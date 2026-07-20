// Tests for src/paths.rs (env-var-driven path resolution + config loading)
// and src/journal.rs (the append-only NDJSON journal writer).

mod common;

use answerbot::journal::{journal, journal_event, unix_ts, write_journal_line};
use answerbot::paths::{
    config_path, env_path, journal_path, load_config, DEFAULT_CONFIG_PATH, DEFAULT_JOURNAL_PATH,
};
use common::{journal_lines, temp_path};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// `ANSWERBOT_CONFIG` / `ANSWERBOT_JOURNAL` are process-global; serialize
/// tests that set/restore them so parallel runs cannot interleave.
static PATH_ENV_LOCK: Mutex<()> = Mutex::new(());

// -- env_path: generic env-var-with-default helper -------------------------
//
// Each test below uses its own uniquely-named env var so parallel tests in
// this binary never race on the same variable.

#[test]
fn env_path_returns_default_when_var_unset() {
    let var = "ANSWERBOT_TEST_ENV_PATH_UNSET";
    std::env::remove_var(var);
    assert_eq!(env_path(var, "fallback"), "fallback");
}

#[test]
fn env_path_returns_override_when_var_set() {
    let var = "ANSWERBOT_TEST_ENV_PATH_SET";
    std::env::set_var(var, "custom-value");
    assert_eq!(env_path(var, "fallback"), "custom-value");
    std::env::remove_var(var);
}

#[test]
fn env_path_empty_override_is_returned_verbatim() {
    let var = "ANSWERBOT_TEST_ENV_PATH_EMPTY";
    std::env::set_var(var, "");
    assert_eq!(env_path(var, "fallback"), "");
    std::env::remove_var(var);
}

// -- config_path / journal_path: default + override, via ANSWERBOT_CONFIG /
// ANSWERBOT_JOURNAL. Each is exercised by exactly one test (no other test in
// this binary touches these two specific env vars), so the temporary
// set/restore below cannot race with a concurrent reader.

#[test]
fn config_path_default_then_override() {
    let _guard = PATH_ENV_LOCK.lock().unwrap();
    let backup = std::env::var("ANSWERBOT_CONFIG").ok();
    std::env::remove_var("ANSWERBOT_CONFIG");
    assert_eq!(config_path(), DEFAULT_CONFIG_PATH);
    std::env::set_var("ANSWERBOT_CONFIG", "custom-config.toml");
    assert_eq!(config_path(), "custom-config.toml");
    match backup {
        Some(v) => std::env::set_var("ANSWERBOT_CONFIG", v),
        None => std::env::remove_var("ANSWERBOT_CONFIG"),
    }
}

#[test]
fn journal_path_default_then_override() {
    let _guard = PATH_ENV_LOCK.lock().unwrap();
    let backup = std::env::var("ANSWERBOT_JOURNAL").ok();
    std::env::remove_var("ANSWERBOT_JOURNAL");
    assert_eq!(journal_path(), DEFAULT_JOURNAL_PATH);
    std::env::set_var("ANSWERBOT_JOURNAL", "custom-journal.jsonl");
    assert_eq!(journal_path(), "custom-journal.jsonl");
    match backup {
        Some(v) => std::env::set_var("ANSWERBOT_JOURNAL", v),
        None => std::env::remove_var("ANSWERBOT_JOURNAL"),
    }
}

// -- load_config: ANSWERBOT_CONFIG-driven load, delegates to load_config_from

#[test]
fn load_config_reads_from_configured_path() {
    let _guard = PATH_ENV_LOCK.lock().unwrap();
    let backup = std::env::var("ANSWERBOT_CONFIG").ok();
    let path = temp_path("load-config-ok");
    std::fs::write(&path, "model = \"m\"\n").unwrap();
    std::env::set_var("ANSWERBOT_CONFIG", path.to_str().unwrap());
    let config = load_config().expect("configured path must load");
    assert_eq!(config.model, "m");
    let _ = std::fs::remove_file(&path);
    match backup {
        Some(v) => std::env::set_var("ANSWERBOT_CONFIG", v),
        None => std::env::remove_var("ANSWERBOT_CONFIG"),
    }
}

#[test]
fn load_config_missing_configured_path_is_error() {
    let _guard = PATH_ENV_LOCK.lock().unwrap();
    let backup = std::env::var("ANSWERBOT_CONFIG").ok();
    let path = temp_path("load-config-missing");
    let _ = std::fs::remove_file(&path);
    std::env::set_var("ANSWERBOT_CONFIG", path.to_str().unwrap());
    let Err(err) = load_config() else {
        panic!("missing configured path must error");
    };
    assert!(err.to_string().contains("failed to read"));
    match backup {
        Some(v) => std::env::set_var("ANSWERBOT_CONFIG", v),
        None => std::env::remove_var("ANSWERBOT_CONFIG"),
    }
}

// -- unix_ts: SystemTime -> seconds, saturating to 0 before the epoch ------

#[test]
fn unix_ts_at_epoch_is_zero() {
    assert_eq!(unix_ts(UNIX_EPOCH), 0);
}

#[test]
fn unix_ts_before_epoch_saturates_to_zero() {
    let before_epoch = UNIX_EPOCH - Duration::from_secs(1);
    assert_eq!(unix_ts(before_epoch), 0);
}

#[test]
fn unix_ts_after_epoch_matches_duration_since() {
    let t = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    assert_eq!(unix_ts(t), 1_700_000_000);
}

#[test]
fn unix_ts_now_is_plausible() {
    // Sanity bound: some time after this comment was written, well before
    // any plausible clock error could push it back to zero.
    assert!(unix_ts(SystemTime::now()) > 1_700_000_000);
}

// -- journal_event / journal: NDJSON append ---------------------------------

#[test]
fn journal_event_writes_ts_and_fields() {
    let path = temp_path("journal-event");
    journal_event(
        path.to_str().unwrap(),
        1_234,
        serde_json::json!({ "event": "question", "text": "hi" }),
    );
    let lines = journal_lines(&path);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["event"], "question");
    assert_eq!(lines[0]["text"], "hi");
    assert_eq!(lines[0]["ts"], 1_234);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn journal_event_appends_multiple_lines() {
    let path = temp_path("journal-event-append");
    journal_event(path.to_str().unwrap(), 1, serde_json::json!({ "n": 1 }));
    journal_event(path.to_str().unwrap(), 2, serde_json::json!({ "n": 2 }));
    let lines = journal_lines(&path);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["n"], 1);
    assert_eq!(lines[1]["n"], 2);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn journal_uses_current_time_for_ts() {
    let path = temp_path("journal-now");
    journal(
        path.to_str().unwrap(),
        serde_json::json!({ "event": "answer" }),
    );
    let lines = journal_lines(&path);
    assert_eq!(lines.len(), 1);
    let ts = lines[0]["ts"].as_u64().expect("ts must be a u64");
    assert!(ts > 1_700_000_000, "ts should be a recent unix timestamp");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn journal_event_open_failure_does_not_panic() {
    // Parent path is a regular file, so open(create+append) cannot succeed.
    // Must print a warning to stderr and return, not panic.
    let blocker = temp_path("journal-open-blocker");
    std::fs::write(&blocker, b"not-a-directory").unwrap();
    let nested = blocker.join("child.jsonl");
    journal_event(
        nested.to_str().unwrap(),
        1,
        serde_json::json!({ "event": "question" }),
    );
    let _ = std::fs::remove_file(&blocker);
}

#[test]
fn write_journal_line_failure_does_not_panic() {
    struct FailWriter;
    impl std::io::Write for FailWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("forced write fail"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    write_journal_line(&mut FailWriter, "fake-path.jsonl", r#"{"event":"x"}"#);
}
