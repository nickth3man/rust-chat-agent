// End-to-end orchestration tests for `answerbot::run` against wiremock
// OpenRouter + Firecrawl servers. No billed API calls.

mod common;

use answerbot::run::{run, AttemptCaps, Endpoints, RunInput};
use answerbot::{load_dotenv, Config};
use common::{journal_lines, temp_path, test_client, test_config};
use serde_json::json;
use std::collections::VecDeque;
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
            .expect("openrouter mock exhausted")
    }
}

fn seq(responses: Vec<ResponseTemplate>) -> SeqRespond {
    SeqRespond {
        responses: Mutex::new(responses.into()),
    }
}

fn tool_query(q: &str) -> serde_json::Value {
    json!({
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": { "arguments": format!(r#"{{"query":"{q}"}}"#) }
                }]
            }
        }]
    })
}

fn content(c: &str) -> serde_json::Value {
    json!({ "choices": [{ "message": { "content": c } }] })
}

fn firecrawl_ok() -> serde_json::Value {
    json!({
        "data": {
            "web": [{
                "url": "https://example.com/paris",
                "title": "Paris",
                "markdown": "Paris is the capital of France."
            }]
        }
    })
}

fn firecrawl_empty() -> serde_json::Value {
    json!({ "data": { "web": [] } })
}

async fn mount_firecrawl(server: &MockServer, body: serde_json::Value) {
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

fn run_input(
    question: &str,
    openrouter: &str,
    firecrawl: &str,
    journal: &str,
    config: Config,
) -> RunInput {
    RunInput {
        question: question.into(),
        config,
        today: "2026-07-20".into(),
        client: test_client(),
        openrouter_key: "or-key".into(),
        firecrawl_key: "fc-key".into(),
        endpoints: Endpoints {
            openrouter: openrouter.into(),
            firecrawl: firecrawl.into(),
        },
        journal_path: journal.into(),
        caps: AttemptCaps {
            network: 3,
            rewrite: 3,
            llm: 3,
        },
        skip_sleep: true,
    }
}

#[tokio::test]
async fn run_empty_question_bails() {
    let journal = temp_path("run-empty-q");
    let err = run(run_input(
        "",
        "http://127.0.0.1:9",
        "http://127.0.0.1:9",
        journal.to_str().unwrap(),
        test_config(),
    ))
    .await
    .unwrap_err();
    assert!(err.to_string().contains("usage: answerbot"), "{err}");
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_happy_path_journals_and_answers() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(tool_query("capital of France")),
            ResponseTemplate::new(200).set_body_json(content("Paris is the capital [S1].")),
        ]))
        .mount(&or)
        .await;
    mount_firecrawl(&fc, firecrawl_ok()).await;

    let journal = temp_path("run-happy");
    run(run_input(
        "What is the capital of France?",
        &or.uri(),
        &fc.uri(),
        journal.to_str().unwrap(),
        test_config(),
    ))
    .await
    .unwrap();

    let lines = journal_lines(&journal);
    let events: Vec<_> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert!(events.contains(&"question"));
    assert!(events.contains(&"query"));
    assert!(events.contains(&"source"));
    assert!(events.contains(&"answer"));
    assert!(!events.contains(&"requery"));
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_anchors_temporal_question() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(tool_query("rust release")),
            ResponseTemplate::new(200).set_body_json(content("Latest is 1.x [S1].")),
        ]))
        .mount(&or)
        .await;
    mount_firecrawl(&fc, firecrawl_ok()).await;

    let journal = temp_path("run-anchor");
    run(run_input(
        "What is the latest Rust release?",
        &or.uri(),
        &fc.uri(),
        journal.to_str().unwrap(),
        test_config(),
    ))
    .await
    .unwrap();

    let lines = journal_lines(&journal);
    assert!(lines.iter().any(|l| l["event"] == "anchor"));
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_one_requery_then_answer() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(tool_query("first query")),
            ResponseTemplate::new(200).set_body_json(content("SEARCH: better query")),
            ResponseTemplate::new(200).set_body_json(content("Final answer [S1].")),
        ]))
        .mount(&or)
        .await;
    // Two searches: initial + requery
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(firecrawl_ok()))
        .mount(&fc)
        .await;

    let journal = temp_path("run-requery");
    run(run_input(
        "Q?",
        &or.uri(),
        &fc.uri(),
        journal.to_str().unwrap(),
        test_config(),
    ))
    .await
    .unwrap();

    let lines = journal_lines(&journal);
    assert!(lines.iter().any(|l| l["event"] == "requery"));
    assert!(lines.iter().any(|l| l["event"] == "answer"));
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_late_requery_rejected() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(tool_query("q")),
            ResponseTemplate::new(200).set_body_json(content("SEARCH: again")),
            ResponseTemplate::new(200).set_body_json(content("SEARCH: still more")),
        ]))
        .mount(&or)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(firecrawl_ok()))
        .mount(&fc)
        .await;

    let journal = temp_path("run-late");
    let err = run(run_input(
        "Q?",
        &or.uri(),
        &fc.uri(),
        journal.to_str().unwrap(),
        test_config(),
    ))
    .await
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("another search after the one allowed"),
        "{err}"
    );
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_zero_citation_retry_then_accept() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(tool_query("q")),
            ResponseTemplate::new(200).set_body_json(content("Paris is the capital.")),
            ResponseTemplate::new(200).set_body_json(content("Paris is the capital [S1].")),
        ]))
        .mount(&or)
        .await;
    mount_firecrawl(&fc, firecrawl_ok()).await;

    let journal = temp_path("run-cite-retry");
    run(run_input(
        "Q?",
        &or.uri(),
        &fc.uri(),
        journal.to_str().unwrap(),
        test_config(),
    ))
    .await
    .unwrap();

    let lines = journal_lines(&journal);
    assert!(lines.iter().any(|l| l["event"] == "no_citations_retry"));
    assert!(lines.iter().any(|l| l["event"] == "answer"));
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_citation_retry_late_requery_rejected() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(tool_query("q")),
            ResponseTemplate::new(200).set_body_json(content("no cites")),
            ResponseTemplate::new(200).set_body_json(content("SEARCH: too late")),
        ]))
        .mount(&or)
        .await;
    mount_firecrawl(&fc, firecrawl_ok()).await;

    let journal = temp_path("run-cite-late");
    let err = run(run_input(
        "Q?",
        &or.uri(),
        &fc.uri(),
        journal.to_str().unwrap(),
        test_config(),
    ))
    .await
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("another search after the one allowed"),
        "{err}"
    );
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_rewrite_http_failure() {
    let journal = temp_path("run-rw-fail");
    let err = run(run_input(
        "Q?",
        "http://127.0.0.1:9",
        "http://127.0.0.1:9",
        journal.to_str().unwrap(),
        test_config(),
    ))
    .await
    .unwrap_err();
    assert!(!err.to_string().is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_search_http_failure_after_rewrite() {
    let or = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_query("q")))
        .mount(&or)
        .await;
    let journal = temp_path("run-search-fail");
    let err = run(run_input(
        "Q?",
        &or.uri(),
        "http://127.0.0.1:9",
        journal.to_str().unwrap(),
        test_config(),
    ))
    .await
    .unwrap_err();
    assert!(!err.to_string().is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_requery_second_llm_failure() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(tool_query("q")),
            ResponseTemplate::new(200).set_body_json(content("SEARCH: better")),
            // Second answer (after requery) fails
            ResponseTemplate::new(500).set_body_string("down"),
        ]))
        .mount(&or)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(firecrawl_ok()))
        .mount(&fc)
        .await;

    let journal = temp_path("run-requery-llm-fail");
    let err = run(RunInput {
        caps: AttemptCaps {
            network: 1,
            rewrite: 3,
            llm: 1,
        },
        ..run_input(
            "Q?",
            &or.uri(),
            &fc.uri(),
            journal.to_str().unwrap(),
            test_config(),
        )
    })
    .await
    .unwrap_err();
    assert!(!err.to_string().is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_requery_search_failure() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(tool_query("q")),
            ResponseTemplate::new(200).set_body_json(content("SEARCH: better")),
        ]))
        .mount(&or)
        .await;
    // First search ok; second (requery) fails.
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(firecrawl_ok()),
            ResponseTemplate::new(500).set_body_string("down"),
            ResponseTemplate::new(500).set_body_string("down"),
            ResponseTemplate::new(500).set_body_string("down"),
        ]))
        .mount(&fc)
        .await;

    let journal = temp_path("run-requery-fail");
    let err = run(RunInput {
        caps: AttemptCaps {
            network: 1,
            rewrite: 3,
            llm: 3,
        },
        ..run_input(
            "Q?",
            &or.uri(),
            &fc.uri(),
            journal.to_str().unwrap(),
            test_config(),
        )
    })
    .await
    .unwrap_err();
    assert!(!err.to_string().is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_answer_llm_failure() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(tool_query("q")),
            // Answer call fails
            ResponseTemplate::new(500).set_body_string("down"),
            ResponseTemplate::new(500).set_body_string("down"),
            ResponseTemplate::new(500).set_body_string("down"),
        ]))
        .mount(&or)
        .await;
    mount_firecrawl(&fc, firecrawl_ok()).await;

    let journal = temp_path("run-answer-fail");
    let err = run(RunInput {
        caps: AttemptCaps {
            network: 1,
            rewrite: 3,
            llm: 1,
        },
        ..run_input(
            "Q?",
            &or.uri(),
            &fc.uri(),
            journal.to_str().unwrap(),
            test_config(),
        )
    })
    .await
    .unwrap_err();
    assert!(!err.to_string().is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_citation_retry_llm_failure() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(tool_query("q")),
            ResponseTemplate::new(200).set_body_json(content("no cites here")),
            ResponseTemplate::new(500).set_body_string("down"),
        ]))
        .mount(&or)
        .await;
    mount_firecrawl(&fc, firecrawl_ok()).await;

    let journal = temp_path("run-cite-fail");
    let err = run(RunInput {
        caps: AttemptCaps {
            network: 1,
            rewrite: 3,
            llm: 1,
        },
        ..run_input(
            "Q?",
            &or.uri(),
            &fc.uri(),
            journal.to_str().unwrap(),
            test_config(),
        )
    })
    .await
    .unwrap_err();
    assert!(!err.to_string().is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn run_empty_search_bails() {
    let or = MockServer::start().await;
    let fc = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_query("q")))
        .mount(&or)
        .await;
    mount_firecrawl(&fc, firecrawl_empty()).await;

    let journal = temp_path("run-no-pages");
    let err = run(run_input(
        "Q?",
        &or.uri(),
        &fc.uri(),
        journal.to_str().unwrap(),
        test_config(),
    ))
    .await
    .unwrap_err();
    assert!(err.to_string().contains("no usable pages"), "{err}");
    let _ = std::fs::remove_file(&journal);
}

#[test]
fn endpoints_default_and_from_env() {
    let d = Endpoints::default();
    assert!(d.openrouter.contains("openrouter.ai"));
    assert!(d.firecrawl.contains("firecrawl.dev"));

    let backup_or = std::env::var("ANSWERBOT_OPENROUTER_URL").ok();
    let backup_fc = std::env::var("ANSWERBOT_FIRECRAWL_URL").ok();
    std::env::set_var("ANSWERBOT_OPENROUTER_URL", "http://or.test/v1");
    std::env::set_var("ANSWERBOT_FIRECRAWL_URL", "http://fc.test/v2");
    let e = Endpoints::from_env();
    assert_eq!(e.openrouter, "http://or.test/v1");
    assert_eq!(e.firecrawl, "http://fc.test/v2");
    match backup_or {
        Some(v) => std::env::set_var("ANSWERBOT_OPENROUTER_URL", v),
        None => std::env::remove_var("ANSWERBOT_OPENROUTER_URL"),
    }
    match backup_fc {
        Some(v) => std::env::set_var("ANSWERBOT_FIRECRAWL_URL", v),
        None => std::env::remove_var("ANSWERBOT_FIRECRAWL_URL"),
    }
}

#[test]
fn load_dotenv_missing_is_ok() {
    // Running from a temp dir with no .env must succeed.
    // Process CWD is global — serialize against sibling dotenv tests.
    let _cwd = CWD_LOCK.lock().unwrap();
    let dir = tempfile_dir();
    let prev = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    let result = load_dotenv();
    std::env::set_current_dir(prev).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    result.expect("missing .env is fine");
}

#[test]
fn load_dotenv_valid_file_is_ok() {
    let _cwd = CWD_LOCK.lock().unwrap();
    let dir = tempfile_dir();
    // Unique var so we don't clobber developer secrets; covers the
    // `dotenvy::dotenv() -> Ok` branch in load_dotenv.
    std::fs::write(dir.join(".env"), "ANSWERBOT_TEST_DOTENV_OK=1\n").unwrap();
    let prev = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    let result = load_dotenv();
    std::env::set_current_dir(prev).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    result.expect("valid .env must load");
    assert_eq!(
        std::env::var("ANSWERBOT_TEST_DOTENV_OK").ok().as_deref(),
        Some("1")
    );
    std::env::remove_var("ANSWERBOT_TEST_DOTENV_OK");
}

#[test]
fn load_dotenv_malformed_is_error() {
    let _cwd = CWD_LOCK.lock().unwrap();
    let dir = tempfile_dir();
    // Invalid UTF-8 contents make dotenvy's reader fail with a non-NotFound error.
    std::fs::write(dir.join(".env"), [0xff, 0xfe, 0xfd, b'\n']).unwrap();
    let prev = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    let result = load_dotenv();
    std::env::set_current_dir(prev).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        result.is_err(),
        "unreadable .env must error, got {result:?}"
    );
}

/// `std::env::set_current_dir` is process-global; parallel dotenv tests must
/// not interleave (restoring CWD to a dir another test already deleted).
static CWD_LOCK: Mutex<()> = Mutex::new(());

fn tempfile_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("answerbot-dotenv-{}-{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
