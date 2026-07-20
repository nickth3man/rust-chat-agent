// Wiremock-backed tests for HTTP retry, OpenRouter LLM helpers, and Firecrawl
// search — no billed upstream calls.

mod common;

use answerbot::http::{
    backoff_duration, is_retryable_post_json_error, journal_retry_and_sleep, post_json,
    post_json_retry_reason, sleep_backoff, try_post_json,
};
use answerbot::llm::{
    chat_body, first_message, journal_reasoning, llm, openrouter_call, reasoning_text, rewrite_llm,
    rewrite_query_from_response, ChatChoice, ChatMessage, ChatResponse, OpenRouterCtx, ToolCall,
    ToolFunction,
};
use answerbot::search::{search, SearchCtx};
use answerbot::Source;
use common::{journal_lines, reasoning_config, temp_path, test_client, test_config};
use serde_json::json;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Sequential responder: each request pops the next template.
struct SeqRespond {
    responses: std::sync::Mutex<std::collections::VecDeque<ResponseTemplate>>,
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
        responses: std::sync::Mutex::new(responses.into()),
    }
}

fn openrouter_ctx<'a>(
    client: &'a reqwest::Client,
    url: &'a str,
    journal: &'a str,
) -> OpenRouterCtx<'a> {
    OpenRouterCtx {
        client,
        url,
        api_key: "test-key",
        journal_path: journal,
        network_max_attempts: 3,
        skip_sleep: true,
    }
}

fn chat_content(content: &str) -> serde_json::Value {
    json!({
        "choices": [{ "message": { "content": content } }]
    })
}

fn chat_tool_query(query: &str) -> serde_json::Value {
    json!({
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": { "arguments": format!(r#"{{"query":"{query}"}}"#) }
                }]
            }
        }]
    })
}

fn chat_reasoning_only(text: &str) -> serde_json::Value {
    json!({
        "choices": [{ "message": { "content": null, "reasoning": text } }]
    })
}

fn firecrawl_web(url: &str, title: &str, markdown: &str) -> serde_json::Value {
    json!({
        "data": {
            "web": [{
                "url": url,
                "title": title,
                "markdown": markdown
            }]
        }
    })
}

// -- pure helpers (no network) ---------------------------------------------

#[test]
fn chat_body_omits_reasoning_when_disabled() {
    let body = chat_body(&test_config(), 100, "sys", "user");
    assert_eq!(body["model"], "test-model");
    assert!(body.get("reasoning").is_none());
}

#[test]
fn chat_body_adds_reasoning_when_enabled() {
    let body = chat_body(&reasoning_config(), 100, "sys", "user");
    assert_eq!(body["reasoning"], json!({}));
}

#[test]
fn first_message_and_reasoning_text_helpers() {
    let empty = ChatResponse { choices: vec![] };
    assert!(first_message(&empty).is_none());
    assert!(reasoning_text(&empty).is_none());

    let resp = ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                content: Some("hi".into()),
                reasoning: Some("  think  ".into()),
                tool_calls: None,
            },
        }],
    };
    assert_eq!(first_message(&resp).unwrap().content.as_deref(), Some("hi"));
    assert_eq!(reasoning_text(&resp), Some("think"));

    let blank_reason = ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                content: None,
                reasoning: Some("   ".into()),
                tool_calls: None,
            },
        }],
    };
    assert!(reasoning_text(&blank_reason).is_none());
}

#[test]
fn rewrite_query_from_response_paths() {
    let ok = ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                content: None,
                reasoning: None,
                tool_calls: Some(vec![ToolCall {
                    function: ToolFunction {
                        arguments: r#"{"query":"capital of France"}"#.into(),
                    },
                }]),
            },
        }],
    };
    assert_eq!(
        rewrite_query_from_response(&ok).unwrap(),
        "capital of France"
    );

    let none = ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                content: Some("prose".into()),
                reasoning: None,
                tool_calls: None,
            },
        }],
    };
    assert!(rewrite_query_from_response(&none).is_err());
}

#[test]
fn journal_reasoning_respects_config_flag() {
    let path = temp_path("journal-reasoning");
    let resp = ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                content: Some("a".into()),
                reasoning: Some("chain".into()),
                tool_calls: None,
            },
        }],
    };
    journal_reasoning(&resp, &test_config(), path.to_str().unwrap());
    assert!(journal_lines(&path).is_empty());

    journal_reasoning(&resp, &reasoning_config(), path.to_str().unwrap());
    let lines = journal_lines(&path);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["event"], "reasoning");
    assert_eq!(lines[0]["text"], "chain");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn backoff_duration_matches_backoff_ms() {
    assert_eq!(
        backoff_duration(0),
        std::time::Duration::from_millis(answerbot::backoff_ms(0))
    );
}

#[test]
fn post_json_retry_reason_unknown_without_reqwest() {
    let err = anyhow::anyhow!("plain error");
    assert_eq!(post_json_retry_reason(&err), "unknown");
    assert!(!is_retryable_post_json_error(&err));
}

#[tokio::test]
async fn sleep_backoff_and_journal_retry_skip_sleep() {
    let path = temp_path("retry-sleep");
    sleep_backoff(0, true).await;
    journal_retry_and_sleep(
        path.to_str().unwrap(),
        "rewrite_retry",
        1,
        "empty-query",
        true,
    )
    .await;
    let lines = journal_lines(&path);
    assert_eq!(lines[0]["event"], "rewrite_retry");
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn sleep_backoff_actually_sleeps_when_not_skipped() {
    // Hits the non-zero Duration branch (250ms for attempt 0). Keep attempt
    // at 0 so the test stays fast.
    sleep_backoff(0, false).await;
}

#[tokio::test]
async fn post_json_retry_reason_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(std::time::Duration::from_secs(3))
                .set_body_json(json!({"ok": true})),
        )
        .mount(&server)
        .await;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(50))
        .build()
        .unwrap();
    let uri = server.uri();
    let err = try_post_json(&client, &uri, "k", &json!({}))
        .await
        .unwrap_err();
    assert_eq!(post_json_retry_reason(&err), "timeout");
    assert!(is_retryable_post_json_error(&err));
}

#[tokio::test]
async fn is_retryable_builder_or_request_error_is_false() {
    // An unusable URL yields a reqwest error that is neither timeout/connect/
    // decode nor an HTTP status — the final `false` arm of the classifier.
    let client = test_client();
    let err = try_post_json(&client, "http://", "k", &json!({}))
        .await
        .unwrap_err();
    assert!(!is_retryable_post_json_error(&err));
    assert_eq!(post_json_retry_reason(&err), "unknown");
}

// -- post_json / try_post_json ---------------------------------------------

#[tokio::test]
async fn post_json_happy_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(&server)
        .await;
    let journal = temp_path("post-ok");
    let client = test_client();
    let v = post_json(
        &client,
        &server.uri(),
        "key",
        &json!({}),
        journal.to_str().unwrap(),
        3,
        true,
    )
    .await
    .unwrap();
    assert_eq!(v["ok"], true);
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn post_json_max_attempts_zero_bails() {
    let journal = temp_path("post-zero");
    let client = test_client();
    let err = post_json(
        &client,
        "http://127.0.0.1:9",
        "key",
        &json!({}),
        journal.to_str().unwrap(),
        0,
        true,
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("internal: network retry"), "{err}");
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn post_json_non_retryable_4xx() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad"))
        .mount(&server)
        .await;
    let journal = temp_path("post-400");
    let client = test_client();
    let err = post_json(
        &client,
        &server.uri(),
        "key",
        &json!({}),
        journal.to_str().unwrap(),
        3,
        true,
    )
    .await
    .unwrap_err();
    assert!(!is_retryable_post_json_error(&err));
    assert!(journal_lines(&journal).is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn post_json_retries_429_then_ok() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(429).set_body_string("slow down"),
            ResponseTemplate::new(200).set_body_json(json!({"ok": true})),
        ]))
        .mount(&server)
        .await;
    let journal = temp_path("post-429");
    let client = test_client();
    let v = post_json(
        &client,
        &server.uri(),
        "key",
        &json!({}),
        journal.to_str().unwrap(),
        3,
        true,
    )
    .await
    .unwrap();
    assert_eq!(v["ok"], true);
    let lines = journal_lines(&journal);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["event"], "network_retry");
    assert!(lines[0]["reason"].as_str().unwrap().contains("status 429"));
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn post_json_exhausts_5xx_retries() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).set_body_string("down"))
        .mount(&server)
        .await;
    let journal = temp_path("post-503");
    let client = test_client();
    let err = post_json(
        &client,
        &server.uri(),
        "key",
        &json!({}),
        journal.to_str().unwrap(),
        2,
        true,
    )
    .await
    .unwrap_err();
    assert!(format!("{err:#}").contains("after 2 attempts"), "{err:#}");
    let lines = journal_lines(&journal);
    assert_eq!(lines.len(), 1); // one retry between attempt 1 and final
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn post_json_decode_failure_is_not_retried() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not-json"))
        .mount(&server)
        .await;
    let journal = temp_path("post-decode");
    let client = test_client();
    let err = post_json(
        &client,
        &server.uri(),
        "key",
        &json!({}),
        journal.to_str().unwrap(),
        3,
        true,
    )
    .await
    .unwrap_err();
    assert!(!is_retryable_post_json_error(&err));
    assert!(journal_lines(&journal).is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn post_json_connect_failure_retries_then_exhausts() {
    let journal = temp_path("post-connect");
    // Closed port with a generous timeout so reqwest classifies this as
    // connect (not timeout). Port 9 is typically closed on loopback.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let err = post_json(
        &client,
        "http://127.0.0.1:9/",
        "key",
        &json!({}),
        journal.to_str().unwrap(),
        2,
        true,
    )
    .await
    .unwrap_err();
    assert!(format!("{err:#}").contains("after 2 attempts"), "{err:#}");
    let reason = journal_lines(&journal)[0]["reason"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        reason == "connect" || reason == "timeout",
        "unexpected reason {reason}"
    );
    // Also exercise the reason helper directly on the final error.
    let direct = post_json_retry_reason(&err);
    assert!(
        direct == "connect" || direct == "timeout" || direct == "unknown",
        "{direct}"
    );
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn try_post_json_ok() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"a": 1})))
        .mount(&server)
        .await;
    let v = try_post_json(&test_client(), &server.uri(), "k", &json!({}))
        .await
        .unwrap();
    assert_eq!(v["a"], 1);
}

// -- llm / rewrite_llm / openrouter_call -----------------------------------

#[tokio::test]
async fn llm_returns_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_content("Paris [S1]")))
        .mount(&server)
        .await;
    let journal = temp_path("llm-ok");
    let client = test_client();
    let uri = server.uri();
    let jpath = journal.to_str().unwrap();
    let ctx = openrouter_ctx(&client, &uri, jpath);
    let text = llm(&ctx, &test_config(), "sys", "user", 3).await.unwrap();
    assert_eq!(text, "Paris [S1]");
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn llm_retries_reasoning_only_then_ok() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(chat_reasoning_only("thinking")),
            ResponseTemplate::new(200).set_body_json(chat_content("done")),
        ]))
        .mount(&server)
        .await;
    let journal = temp_path("llm-reason");
    let client = test_client();
    let uri = server.uri();
    let jpath = journal.to_str().unwrap();
    let ctx = openrouter_ctx(&client, &uri, jpath);
    let text = llm(&ctx, &reasoning_config(), "sys", "user", 3)
        .await
        .unwrap();
    assert_eq!(text, "done");
    let lines = journal_lines(&journal);
    assert!(lines
        .iter()
        .any(|l| l["event"] == "empty_answer_retry" && l["reason"] == "reasoning-only"));
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn llm_exhausts_empty_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_content("   ")))
        .mount(&server)
        .await;
    let journal = temp_path("llm-empty");
    let client = test_client();
    let uri = server.uri();
    let jpath = journal.to_str().unwrap();
    let ctx = openrouter_ctx(&client, &uri, jpath);
    let err = llm(&ctx, &test_config(), "sys", "user", 2)
        .await
        .unwrap_err();
    assert!(format!("{err:#}").contains("after 2 attempts"), "{err:#}");
    assert!(journal_lines(&journal)
        .iter()
        .any(|l| l["reason"] == "empty"));
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn llm_max_attempts_zero_bails() {
    let journal = temp_path("llm-zero");
    let client = test_client();
    let ctx = openrouter_ctx(&client, "http://127.0.0.1:9", journal.to_str().unwrap());
    let err = llm(&ctx, &test_config(), "sys", "user", 0)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("internal: llm retry"), "{err}");
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn rewrite_llm_happy_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_tool_query("rust version")))
        .mount(&server)
        .await;
    let journal = temp_path("rewrite-ok");
    let client = test_client();
    let uri = server.uri();
    let jpath = journal.to_str().unwrap();
    let ctx = openrouter_ctx(&client, &uri, jpath);
    let q = rewrite_llm(&ctx, &test_config(), "sys", "user", 3)
        .await
        .unwrap();
    assert_eq!(q, "rust version");
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn rewrite_llm_retries_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(seq(vec![
            ResponseTemplate::new(200).set_body_json(chat_content("no tool")),
            ResponseTemplate::new(200).set_body_json(chat_tool_query("ok query")),
        ]))
        .mount(&server)
        .await;
    let journal = temp_path("rewrite-retry");
    let client = test_client();
    let uri = server.uri();
    let jpath = journal.to_str().unwrap();
    let ctx = openrouter_ctx(&client, &uri, jpath);
    let q = rewrite_llm(&ctx, &test_config(), "sys", "user", 3)
        .await
        .unwrap();
    assert_eq!(q, "ok query");
    assert!(journal_lines(&journal)
        .iter()
        .any(|l| l["event"] == "rewrite_retry"));
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn rewrite_llm_exhausts_into_final_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_content("nope")))
        .mount(&server)
        .await;
    let journal = temp_path("rewrite-exh");
    let client = test_client();
    let uri = server.uri();
    let jpath = journal.to_str().unwrap();
    let ctx = openrouter_ctx(&client, &uri, jpath);
    let err = rewrite_llm(&ctx, &test_config(), "sys", "user", 2)
        .await
        .unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("after 2 attempts"), "{msg}");
    assert!(msg.contains("tool call"), "{msg}");
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn rewrite_llm_max_attempts_zero_bails() {
    let journal = temp_path("rewrite-zero");
    let client = test_client();
    let ctx = openrouter_ctx(&client, "http://127.0.0.1:9", journal.to_str().unwrap());
    let err = rewrite_llm(&ctx, &test_config(), "sys", "user", 0)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("internal: rewrite retry"), "{err}");
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn openrouter_call_http_error_propagates() {
    let journal = temp_path("or-http-err");
    let client = test_client();
    let ctx = openrouter_ctx(&client, "http://127.0.0.1:9", journal.to_str().unwrap());
    let err = openrouter_call(&ctx, &json!({"model": "m"}))
        .await
        .unwrap_err();
    assert!(!err.to_string().is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn llm_http_error_propagates() {
    let journal = temp_path("llm-http-err");
    let client = test_client();
    let ctx = openrouter_ctx(&client, "http://127.0.0.1:9", journal.to_str().unwrap());
    let err = llm(&ctx, &test_config(), "sys", "user", 1)
        .await
        .unwrap_err();
    assert!(!err.to_string().is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn rewrite_llm_http_error_propagates() {
    let journal = temp_path("rw-http-err");
    let client = test_client();
    let ctx = openrouter_ctx(&client, "http://127.0.0.1:9", journal.to_str().unwrap());
    let err = rewrite_llm(&ctx, &test_config(), "sys", "user", 1)
        .await
        .unwrap_err();
    assert!(!err.to_string().is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn search_http_error_propagates() {
    let journal = temp_path("search-http-err");
    let client = test_client();
    let ctx = SearchCtx {
        client: &client,
        url: "http://127.0.0.1:9",
        api_key: "fk",
        journal_path: journal.to_str().unwrap(),
        network_max_attempts: 1,
        skip_sleep: true,
    };
    let mut registry = Vec::new();
    let err = search(&ctx, "q", &mut registry).await.unwrap_err();
    assert!(!err.to_string().is_empty());
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn openrouter_call_bad_shape_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"choices": "nope"})))
        .mount(&server)
        .await;
    let journal = temp_path("or-bad");
    let client = test_client();
    let uri = server.uri();
    let jpath = journal.to_str().unwrap();
    let ctx = openrouter_ctx(&client, &uri, jpath);
    let err = openrouter_call(&ctx, &json!({})).await.unwrap_err();
    assert!(
        err.to_string().contains("failed to parse OpenRouter"),
        "{err}"
    );
    let _ = std::fs::remove_file(&journal);
}

// -- search ----------------------------------------------------------------

#[tokio::test]
async fn search_ingests_and_journals_sources() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(firecrawl_web(
            "https://a.com",
            "A",
            "body a",
        )))
        .mount(&server)
        .await;
    let journal = temp_path("search-ok");
    let client = test_client();
    let uri = server.uri();
    let jpath = journal.to_str().unwrap();
    let ctx = SearchCtx {
        client: &client,
        url: &uri,
        api_key: "fk",
        journal_path: jpath,
        network_max_attempts: 3,
        skip_sleep: true,
    };
    let mut registry = Vec::new();
    search(&ctx, "q", &mut registry).await.unwrap();
    assert_eq!(registry.len(), 1);
    assert_eq!(registry[0].id, "S1");
    assert_eq!(registry[0].url, "https://a.com");
    let lines = journal_lines(&journal);
    assert!(lines
        .iter()
        .any(|l| l["event"] == "source" && l["id"] == "S1"));
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn search_parse_error_propagates() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": {}})))
        .mount(&server)
        .await;
    let journal = temp_path("search-bad");
    let client = test_client();
    let uri = server.uri();
    let jpath = journal.to_str().unwrap();
    let ctx = SearchCtx {
        client: &client,
        url: &uri,
        api_key: "fk",
        journal_path: jpath,
        network_max_attempts: 3,
        skip_sleep: true,
    };
    let mut registry: Vec<Source> = Vec::new();
    let err = search(&ctx, "q", &mut registry).await.unwrap_err();
    assert!(
        err.to_string().contains("missing or null data.web"),
        "{err}"
    );
    let _ = std::fs::remove_file(&journal);
}

#[tokio::test]
async fn search_empty_web_leaves_registry_empty() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": {"web": []}})))
        .mount(&server)
        .await;
    let journal = temp_path("search-empty");
    let client = test_client();
    let uri = server.uri();
    let jpath = journal.to_str().unwrap();
    let ctx = SearchCtx {
        client: &client,
        url: &uri,
        api_key: "fk",
        journal_path: jpath,
        network_max_attempts: 3,
        skip_sleep: true,
    };
    let mut registry = Vec::new();
    search(&ctx, "q", &mut registry).await.unwrap();
    assert!(registry.is_empty());
    let _ = std::fs::remove_file(&journal);
}
