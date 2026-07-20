// Binary-entry tests: exercise `src/main.rs` → `run_cli()` under a temp CWD
// with wiremock upstreams so the instrumented binary is covered without
// billed API calls.

mod common;

use serde_json::json;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

struct SeqRespond {
    responses: Mutex<VecDeque<ResponseTemplate>>,
}

impl Respond for SeqRespond {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("mock exhausted")
    }
}

fn seq(responses: Vec<ResponseTemplate>) -> SeqRespond {
    SeqRespond {
        responses: Mutex::new(responses.into()),
    }
}

fn work_dir() -> PathBuf {
    // Windows SystemTime often has ~15ms resolution, so parallel tokio tests
    // can collide if we key only on pid + "nanos". Match `common::temp_path`.
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("answerbot-cli-{}-{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.toml"),
        "model = \"test-model\"\nreasoning = false\n",
    )
    .unwrap();
    dir
}

/// Build a `Command` for the instrumented binary. Avoid `env_clear()` — on
/// Windows that drops vars the Winsock stack needs (e.g. `SystemRoot`) and
/// breaks outbound HTTP to wiremock. Instead remove only the secrets/path
/// overrides we care about, and keep LLVM coverage vars intact.
fn bin_cmd(bin: &str, dir: &Path) -> Command {
    let mut cmd = Command::new(bin);
    cmd.current_dir(dir)
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("FIRECRAWL_API_KEY")
        .env_remove("ANSWERBOT_CONFIG")
        .env_remove("ANSWERBOT_JOURNAL")
        .env_remove("ANSWERBOT_OPENROUTER_URL")
        .env_remove("ANSWERBOT_FIRECRAWL_URL");
    cmd
}

/// Owned path string for `Command::env` (avoid temporary `Cow` footguns).
fn path_env(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[tokio::test]
async fn binary_happy_path_covers_main() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "function": { "arguments": "{\"query\":\"capital of France\"}" }
                        }]
                    }
                }]
            })),
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "content": "Paris is the capital [S1]." } }]
            })),
        ]))
        .mount(&or)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "web": [{
                    "url": "https://example.com/paris",
                    "title": "Paris",
                    "markdown": "Paris is the capital of France."
                }]
            }
        })))
        .mount(&fc)
        .await;

    let dir = work_dir();
    let journal = dir.join("journal.jsonl");
    let config = dir.join("config.toml");
    let bin = env!("CARGO_BIN_EXE_answerbot");
    let output = bin_cmd(bin, &dir)
        .env("OPENROUTER_API_KEY", "or-test")
        .env("FIRECRAWL_API_KEY", "fc-test")
        .env("ANSWERBOT_OPENROUTER_URL", or.uri())
        .env("ANSWERBOT_FIRECRAWL_URL", fc.uri())
        .env("ANSWERBOT_CONFIG", path_env(&config))
        .env("ANSWERBOT_JOURNAL", path_env(&journal))
        .arg("What is the capital of France?")
        .output()
        .expect("spawn answerbot");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "status={:?}\nstdout={stdout}\nstderr={stderr}",
        output.status
    );
    assert!(stdout.contains("Paris"), "stdout={stdout}");
    assert!(stdout.contains("Sources:"), "stdout={stdout}");
    assert!(journal.exists(), "journal should be written");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn binary_usage_error_without_args() {
    let dir = work_dir();
    // Prefer relative paths (CWD = work_dir) so we don't depend on absolute
    // ANSWERBOT_CONFIG encoding across Windows shells.
    let journal = dir.join("journal.jsonl");
    let config = dir.join("config.toml");
    assert!(config.is_file(), "fixture missing at {}", config.display());
    let bin = env!("CARGO_BIN_EXE_answerbot");
    let output = bin_cmd(bin, &dir)
        .env("OPENROUTER_API_KEY", "or-test")
        .env("FIRECRAWL_API_KEY", "fc-test")
        .env("ANSWERBOT_JOURNAL", "journal.jsonl")
        .output()
        .expect("spawn answerbot");
    assert!(!output.status.success());
    let err = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(err.contains("usage: answerbot"), "{err}");
    assert!(!journal.exists(), "empty question should not journal");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn binary_missing_firecrawl_key_errors() {
    let dir = work_dir();
    let config = dir.join("config.toml");
    assert!(
        config.is_file(),
        "fixture config must exist at {}",
        config.display()
    );
    let bin = env!("CARGO_BIN_EXE_answerbot");
    let output = bin_cmd(bin, &dir)
        .env("OPENROUTER_API_KEY", "or-test")
        .env("ANSWERBOT_CONFIG", path_env(&config))
        .env("ANSWERBOT_JOURNAL", path_env(&dir.join("j.jsonl")))
        .arg("question")
        .output()
        .expect("spawn answerbot");
    assert!(!output.status.success());
    let err = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(err.contains("FIRECRAWL_API_KEY"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn binary_missing_openrouter_key_errors() {
    let dir = work_dir();
    let config = dir.join("config.toml");
    let bin = env!("CARGO_BIN_EXE_answerbot");
    let output = bin_cmd(bin, &dir)
        .env("FIRECRAWL_API_KEY", "fc-test")
        .env("ANSWERBOT_CONFIG", path_env(&config))
        .env("ANSWERBOT_JOURNAL", path_env(&dir.join("j.jsonl")))
        .arg("question")
        .output()
        .expect("spawn answerbot");
    assert!(!output.status.success());
    let err = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(err.contains("OPENROUTER_API_KEY"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn binary_missing_config_errors() {
    let dir = work_dir();
    let missing = dir.join("no-such-config.toml");
    let bin = env!("CARGO_BIN_EXE_answerbot");
    let output = bin_cmd(bin, &dir)
        .env("OPENROUTER_API_KEY", "or-test")
        .env("FIRECRAWL_API_KEY", "fc-test")
        .env("ANSWERBOT_CONFIG", path_env(&missing))
        .env("ANSWERBOT_JOURNAL", path_env(&dir.join("j.jsonl")))
        .arg("question")
        .output()
        .expect("spawn answerbot");
    assert!(!output.status.success());
    let err = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(err.contains("failed to read"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn binary_malformed_dotenv_errors() {
    let dir = work_dir();
    std::fs::write(dir.join(".env"), [0xff, 0xfe, 0xfd, b'\n']).unwrap();
    let bin = env!("CARGO_BIN_EXE_answerbot");
    let output = bin_cmd(bin, &dir)
        .env("OPENROUTER_API_KEY", "or-test")
        .env("FIRECRAWL_API_KEY", "fc-test")
        .env("ANSWERBOT_CONFIG", path_env(&dir.join("config.toml")))
        .env("ANSWERBOT_JOURNAL", path_env(&dir.join("j.jsonl")))
        .arg("question")
        .output()
        .expect("spawn answerbot");
    assert!(!output.status.success());
    let err = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(err.contains("failed to load .env"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
