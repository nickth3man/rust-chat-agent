// Shared test fixtures for the integration tests under tests/.
// Loaded via `mod common;` from each sibling .rs file — NOT compiled as a
// separate test target.
//
// Each `tests/*.rs` file is its own crate, so `mod common;` is recompiled
// per test binary and only a subset of these helpers is used by any given
// binary. `dead_code` is allowed at module scope for that reason (this is
// the standard shared-fixtures pattern for Rust integration tests, not a
// clippy pedantic lint covered by the `[lints.clippy]` allow-with-comment
// policy in AGENTS.md).
#![allow(dead_code)]

use answerbot::Source;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// A fresh, not-yet-existing file path in the OS temp dir. Unique per call
/// (process id + monotonic counter) so parallel tests in the same binary —
/// or across binaries — never collide on a journal/config fixture path.
pub fn temp_path(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    std::env::temp_dir().join(format!(
        "answerbot-test-{tag}-{}-{n}.jsonl",
        std::process::id()
    ))
}

/// Parse a journal file's NDJSON lines into `serde_json::Value`s. A missing
/// file (nothing was ever journaled) returns an empty vec rather than erroring.
pub fn journal_lines(path: &std::path::Path) -> Vec<serde_json::Value> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    contents
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("journal line must be valid JSON"))
        .collect()
}

/// Minimal Source with only the `id` set. Use for tests that only care about
/// ID handling (citation validation, registry lookup, ID generation).
pub fn src(id: &str) -> Source {
    Source {
        id: id.into(),
        url: String::new(),
        title: String::new(),
        content: String::new(),
    }
}

/// Fully-populated Source. Use when the test asserts on title/url/content.
pub fn full_src(id: &str, title: &str, url: &str, content: &str) -> Source {
    Source {
        id: id.into(),
        title: title.into(),
        url: url.into(),
        content: content.into(),
    }
}

/// Build a `reqwest::Client` with a short timeout suitable for mock-server tests.
pub fn test_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("test client")
}

/// Minimal non-reasoning config used by orchestration / LLM tests.
pub fn test_config() -> answerbot::Config {
    answerbot::Config {
        model: "test-model".into(),
        temperature: 0.0,
        reasoning: false,
    }
}

/// Reasoning-enabled config (journals chain-of-thought when present).
pub fn reasoning_config() -> answerbot::Config {
    answerbot::Config {
        model: "test-model".into(),
        temperature: 0.0,
        reasoning: true,
    }
}
