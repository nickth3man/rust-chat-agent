// Filesystem path resolution for the two on-disk inputs/outputs: config.toml
// (read) and journal.jsonl (appended). Overrides come from `ANSWERBOT_CONFIG`
// / `ANSWERBOT_JOURNAL`; paths are relative to the process CWD unless
// absolute. Kept separate from `run.rs` so the env-var-reading path helpers
// can be exercised without pulling in the orchestration.

use crate::{load_config_from, Config};
use anyhow::Result;

pub const DEFAULT_CONFIG_PATH: &str = "config.toml";
pub const DEFAULT_JOURNAL_PATH: &str = "journal.jsonl";

/// Read an env var, falling back to `default` when unset (or invalid Unicode).
pub fn env_path(var: &str, default: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| default.to_string())
}

/// Resolve the config.toml path: `ANSWERBOT_CONFIG` or `config.toml`.
pub fn config_path() -> String {
    env_path("ANSWERBOT_CONFIG", DEFAULT_CONFIG_PATH)
}

/// Resolve the journal path: `ANSWERBOT_JOURNAL` or `journal.jsonl`.
pub fn journal_path() -> String {
    env_path("ANSWERBOT_JOURNAL", DEFAULT_JOURNAL_PATH)
}

/// Load and parse the config at `config_path()`.
pub fn load_config() -> Result<Config> {
    load_config_from(config_path())
}
