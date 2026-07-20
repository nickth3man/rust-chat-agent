// answerbot — research agent CLI.
//
// The orchestration (env loading, LLM/Firecrawl calls, journaling, printing)
// lives here. The pure LLM-facing helpers (Source, prompt formatting, citation
// validation) live in `src/lib.rs` and are exercised by integration tests in
// `tests/`.
//
// Flow (matches the diagram in README.md):
//   1. You ask a question
//   2. The AI rewrites it into one good search query   (LLM call #1)
//   3. Firecrawl searches AND reads the top pages in one trip
//   4. The AI answers with [S1] [S2] citations — or asks
//      for exactly ONE more search if something is missing (LLM call #2, maybe #3)
//   5. Citations pointing at sources that don't exist are stripped
//   6. Every step is appended to journal.jsonl
//
// Run:  cargo run -- "your question"
// Config: API keys are read from a .env file at the project root at startup
// (OPENROUTER_API_KEY, FIRECRAWL_API_KEY). The model itself is selected in
// config/models.json (parsed by load_config() below).

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::Write;
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::sleep;

use answerbot::{
    answer_prompt, answer_system_prompt, extract_answer_text, has_citations, is_retryable_status,
    next_source_id, parse_config, parse_firecrawl_web, parse_requery, parse_rewrite_query_arg,
    query_system_prompt, registry_contains_url, rewrite_with_anchor, should_reject_late_requery,
    strip_invalid_citations, truncate_content, Config, RewriteQueryReject, Source,
};

const MAX_SOURCES_PER_SEARCH: usize = 4; // pages Firecrawl reads per trip
const MAX_CHARS_PER_SOURCE: usize = 8_000; // truncate long pages ("safe limits")
const REQUEST_TIMEOUT_SECS: u64 = 30;
/// Bounded retry caps. Each is the TOTAL number of attempts (1 = no retry).
/// `NETWORK_MAX_ATTEMPTS` covers transient HTTP failures (timeout, connect,
/// 429, 5xx) inside `post_json`. `REWRITE_MAX_ATTEMPTS` covers the rewrite
/// step emitting a malformed tool call (Class A: a stochastic small-model
/// failure). `LLM_MAX_ATTEMPTS` covers the answering step returning no text
/// (Class B: reasoning-only or empty turn from a reasoning model).
const NETWORK_MAX_ATTEMPTS: u32 = 3;
const REWRITE_MAX_ATTEMPTS: u32 = 5;
const LLM_MAX_ATTEMPTS: u32 = 3;

const DEFAULT_CONFIG_PATH: &str = "config/models.json";
const DEFAULT_JOURNAL_PATH: &str = "journal.jsonl";

// ---------------------------------------------------------------------------
// Model configuration — loaded from config/models.json at startup.
// Path override: ANSWERBOT_CONFIG. The Config struct and parsing live in
// src/lib.rs for testability.
// ---------------------------------------------------------------------------
fn env_path(var: &str, default: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| default.to_string())
}

fn config_path() -> String {
    env_path("ANSWERBOT_CONFIG", DEFAULT_CONFIG_PATH)
}

fn journal_path() -> String {
    env_path("ANSWERBOT_JOURNAL", DEFAULT_JOURNAL_PATH)
}

fn load_config() -> Result<Config> {
    let path = config_path();
    let contents = std::fs::read_to_string(&path).with_context(|| {
        format!("failed to read {path} (set ANSWERBOT_CONFIG or run from the repo root)")
    })?;
    parse_config(&contents)
}

/// Today's date in the user's local timezone as YYYY-MM-DD.
static TODAY: LazyLock<String> =
    LazyLock::new(|| chrono::Local::now().format("%Y-%m-%d").to_string());

// ---------------------------------------------------------------------------
// The journal: one JSON line per event. This single file is the audit trail,
// the dedup record, and the citation registry's paper trail all at once.
//
// We use `std::fs` (synchronous) rather than `tokio::fs` here and in
// `load_config()`: these are tiny files (a few hundred bytes each) written a
// handful of times per run. The blocking duration is negligible compared to
// the network calls that dominate the runtime, and `tokio::fs` would add
// noise without measurable benefit. The tokio runtime is multi-threaded, so
// these brief synchronous calls do not stall other tasks.
// ---------------------------------------------------------------------------
fn journal(mut event: Value) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    event["ts"] = json!(ts);
    // AGENTS.md declares "everything journaled" a design constraint. Surface
    // open/write failures on stderr so silent journal loss is at least
    // observable to the operator; do not abort the run.
    let path = journal_path();
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{event}") {
                eprintln!("warning: journal write failed ({path}): {e}");
            }
        }
        Err(e) => eprintln!("warning: could not open {path}: {e}"),
    }
}

/// Journal a retry event and sleep for the backoff of the prior attempt.
/// `attempt` is 1-based (matches journal fields). Used by the LLM/rewrite
/// loops; `post_json` journals an extra `url` field then calls `sleep_backoff`.
async fn journal_retry_and_sleep(event: &str, attempt: u32, reason: &str) {
    journal(json!({
        "event": event,
        "attempt": attempt,
        "reason": reason,
    }));
    sleep_backoff(attempt.saturating_sub(1)).await;
}

async fn sleep_backoff(attempt_0based: u32) {
    sleep(backoff_duration(attempt_0based)).await;
}

// ---------------------------------------------------------------------------
// OpenRouter chat-completions helpers (OpenAI-compatible).
// Docs: https://openrouter.ai/docs/api-reference/overview
// ---------------------------------------------------------------------------

// --- OpenRouter response types ---

/// `OpenRouter` (OpenAI-compatible) chat-completions response. Only the
/// fields we read are typed; the rest is ignored by serde.
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize)]
struct ToolCall {
    function: ToolFunction,
}

#[derive(Deserialize)]
struct ToolFunction {
    /// JSON-encoded arguments string (per `OpenAI` tool-call spec).
    arguments: String,
}

// --- Retry plumbing ------------------------------------------------------

/// Exponential backoff: 250ms, 500ms, 1000ms, 2000ms, 4000ms (capped at 4s).
/// `attempt` is 0-indexed (0 = first retry). Pure arithmetic; sleep happens
/// at call site. Delegates to `answerbot::backoff_ms` so the schedule lives
/// in the pure lib (and is covered by the integration tests).
fn backoff_duration(attempt: u32) -> std::time::Duration {
    std::time::Duration::from_millis(answerbot::backoff_ms(attempt))
}

/// One POST attempt: send → check status → parse JSON. Wrapped by `post_json`
/// in a bounded retry loop. Kept separate so the retry classification can
/// inspect a single error in isolation.
///
/// Note: HTTP `Retry-After` on 429 responses is NOT honored — by the time
/// `error_for_status()` converts the response to an error, the headers are
/// consumed and unavailable. We retry with fixed exponential backoff instead.
/// Acceptable for current usage (single query per invocation, generous upstream
/// limits); revisit if 429 storms become operational.
async fn try_post_json(
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
fn find_reqwest_error(err: &anyhow::Error) -> Option<&reqwest::Error> {
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
///
/// Note: not unit-tested. `reqwest::Error` has no public constructor, so these
/// classifiers are exercised end-to-end via real `post_json` retries (see
/// `network_retry` events in `journal.jsonl`). The status-code classification
/// they reduce to is unit-tested via `is_retryable_status` in `tests/retry.rs`.
fn is_retryable_post_json_error(err: &anyhow::Error) -> bool {
    find_reqwest_error(err).is_some_and(is_retryable_reqwest_error)
}

fn is_retryable_reqwest_error(rerr: &reqwest::Error) -> bool {
    if rerr.is_timeout() || rerr.is_connect() {
        return true;
    }
    if rerr.is_decode() {
        return false;
    }
    if let Some(status) = rerr.status() {
        return is_retryable_status(status.as_u16());
    }
    false
}

/// Short label for the journal entry describing why a `post_json` attempt
/// is being retried. Prefers timeout/connect/status labels when present.
fn post_json_retry_reason(err: &anyhow::Error) -> String {
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

/// POST a JSON body with bearer auth, return the parsed JSON response.
/// Shared by the `OpenRouter` chat-completions call and the Firecrawl
/// search call so HTTP-status and timeout handling lives in one place.
/// Retries transient failures (timeout, connect, 429, 5xx) with exponential
/// backoff up to `NETWORK_MAX_ATTEMPTS`. Non-retryable errors (other 4xx,
/// body-decode failures) propagate immediately.
async fn post_json(
    client: &reqwest::Client,
    url: &str,
    bearer_key: &str,
    body: &Value,
) -> Result<Value> {
    for attempt in 0..NETWORK_MAX_ATTEMPTS {
        match try_post_json(client, url, bearer_key, body).await {
            Ok(v) => return Ok(v),
            // Non-retryable failures (deterministic 4xx, body-decode) go
            // through unchanged — we never wasted a billed retry on them.
            Err(e) if !is_retryable_post_json_error(&e) => return Err(e),
            Err(e) if attempt + 1 >= NETWORK_MAX_ATTEMPTS => {
                return Err(e).context(format!("after {} attempts", attempt + 1));
            }
            Err(e) => {
                journal(json!({
                    "event": "network_retry",
                    "attempt": attempt + 1,
                    "url": url,
                    "reason": post_json_retry_reason(&e),
                }));
                sleep_backoff(attempt).await;
            }
        }
    }
    unreachable!("loop returns or continues on every path")
}

/// POST a chat-completions request body to `OpenRouter` and return the parsed
/// response. Delegates HTTP plumbing to `post_json` and deserializes the
/// body into a typed `ChatResponse`.
async fn openrouter_call(client: &reqwest::Client, body: &Value) -> Result<ChatResponse> {
    let key = std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;
    let v = post_json(
        client,
        "https://openrouter.ai/api/v1/chat/completions",
        &key,
        body,
    )
    .await?;
    serde_json::from_value(v).context("failed to parse OpenRouter chat response")
}

fn first_message(resp: &ChatResponse) -> Option<&ChatMessage> {
    resp.choices.first().map(|c| &c.message)
}

/// Non-empty trimmed reasoning text from the first choice, if any.
fn reasoning_text(resp: &ChatResponse) -> Option<&str> {
    first_message(resp)
        .and_then(|m| m.reasoning.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Reasoning/thinking models return their chain-of-thought in the `reasoning`
/// field. When the model is configured as reasoning, journal that text if the
/// provider supplied one. No-op for non-reasoning models.
fn journal_reasoning(resp: &ChatResponse, config: &Config) {
    if !config.reasoning {
        return;
    }
    if let Some(text) = reasoning_text(resp) {
        journal(json!({ "event": "reasoning", "text": text }));
    }
}

/// Common chat-completions body (model, temperature, `max_tokens`, messages)
/// shared by every LLM call. Adds the empty `reasoning` object that
/// reasoning-capable models expect when configured.
fn chat_body(config: &Config, max_tokens: u64, system: &str, user: &str) -> Value {
    let mut body = json!({
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": max_tokens,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ]
    });
    if config.reasoning {
        body["reasoning"] = json!({});
    }
    body
}

/// Answering LLM call: returns the text reply from a plain chat completion.
/// Retries on empty `content` (Class B) — common with reasoning models that
/// occasionally finish their chain-of-thought but emit no final answer, or
/// transiently return an entirely empty turn. Bounded by `LLM_MAX_ATTEMPTS`.
async fn llm(
    client: &reqwest::Client,
    config: &Config,
    system: &str,
    user: &str,
) -> Result<String> {
    for attempt in 0..LLM_MAX_ATTEMPTS {
        let resp: ChatResponse =
            openrouter_call(client, &chat_body(config, 1500, system, user)).await?;
        if let Some(text) =
            extract_answer_text(first_message(&resp).and_then(|m| m.content.as_deref()))
        {
            journal_reasoning(&resp, config);
            return Ok(text);
        }
        // Content is empty/None. Distinguish "reasoning-only" (reasoning is
        // present, content is empty — common stochastic failure mode) from a
        // fully empty turn (both empty — could be a transient blank response,
        // worth one or two retries).
        let reason = if reasoning_text(&resp).is_some() {
            "reasoning-only"
        } else {
            "empty"
        };
        let n = attempt + 1;
        if n >= LLM_MAX_ATTEMPTS {
            return Err(anyhow::Error::msg("no text in LLM response"))
                .context(format!("after {n} attempts"));
        }
        journal_retry_and_sleep("empty_answer_retry", n, reason).await;
    }
    unreachable!("loop returns or continues on every path")
}

// ---------------------------------------------------------------------------
// Tool schema for the query rewrite step. Forces the model to respond with a
// structured tool call instead of free-form prose.
// ---------------------------------------------------------------------------
const REWRITE_TOOL: &str = r#"{
  "type": "function",
  "function": {
    "name": "generate_search_query",
    "description": "Generate a short web search query from the user's question",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query (plain natural language, 1-6 words)"
        }
      },
      "required": ["query"]
    }
  }
}"#;

/// Shape failures from a forced rewrite tool call. Carries enough to journal
/// a retry reason and to build the final error with the original messages.
enum RewriteFail {
    NoToolCall,
    InvalidArgs(serde_json::Error),
    MissingQuery,
    EmptyQuery,
}

impl RewriteFail {
    fn reason(&self) -> &'static str {
        match self {
            Self::NoToolCall => "no-tool-call",
            Self::InvalidArgs(_) => "invalid-tool-args",
            Self::MissingQuery => "missing-query-field",
            Self::EmptyQuery => "empty-query",
        }
    }

    fn into_final_error(self, attempts: u32) -> anyhow::Error {
        match self {
            Self::NoToolCall => anyhow::Error::msg("rewrite: model did not return a tool call")
                .context(format!("after {attempts} attempts")),
            Self::InvalidArgs(e) => anyhow::Error::from(e).context(format!(
                "rewrite: tool call missing valid arguments JSON after {attempts} attempts"
            )),
            Self::MissingQuery => {
                anyhow::Error::msg("rewrite: tool call arguments missing 'query' field")
                    .context(format!("after {attempts} attempts"))
            }
            Self::EmptyQuery => {
                anyhow::Error::msg("rewrite: tool call arguments had empty 'query' field")
                    .context(format!("after {attempts} attempts"))
            }
        }
    }
}

impl From<RewriteQueryReject> for RewriteFail {
    fn from(reject: RewriteQueryReject) -> Self {
        match reject {
            RewriteQueryReject::Missing => Self::MissingQuery,
            RewriteQueryReject::Empty => Self::EmptyQuery,
        }
    }
}

/// Pull the search query out of a forced `generate_search_query` tool call.
fn rewrite_query_from_response(resp: &ChatResponse) -> Result<String, RewriteFail> {
    let Some(tc) = first_message(resp)
        .and_then(|m| m.tool_calls.as_ref())
        .and_then(|tcs| tcs.first())
    else {
        return Err(RewriteFail::NoToolCall);
    };
    let args: Value =
        serde_json::from_str(&tc.function.arguments).map_err(RewriteFail::InvalidArgs)?;
    parse_rewrite_query_arg(&args).map_err(RewriteFail::from)
}

/// Forced-tool-call LLM for query rewriting. Sends `tool_choice: "required"`
/// so the model must respond with a `generate_search_query` tool call.
/// Retries on any tool-call-shape failure (Class A) — no tool call at all,
/// unparseable arguments, missing `query`, or blank `query` — bounded by
/// `REWRITE_MAX_ATTEMPTS`. Reasoning-capable models sometimes ignore
/// `tool_choice: "required"` and answer in prose instead.
async fn rewrite_llm(
    client: &reqwest::Client,
    config: &Config,
    system: &str,
    user: &str,
) -> Result<String> {
    let tool_schema: Value = serde_json::from_str(REWRITE_TOOL)?;
    let mut body = chat_body(config, 250, system, user);
    body["tools"] = json!([tool_schema]);
    body["tool_choice"] = json!("required");

    for attempt in 0..REWRITE_MAX_ATTEMPTS {
        let resp = openrouter_call(client, &body).await?;
        match rewrite_query_from_response(&resp) {
            Ok(query) => {
                journal_reasoning(&resp, config);
                return Ok(query);
            }
            Err(fail) => {
                let n = attempt + 1;
                if n >= REWRITE_MAX_ATTEMPTS {
                    return Err(fail.into_final_error(n));
                }
                journal_retry_and_sleep("rewrite_retry", n, fail.reason()).await;
            }
        }
    }
    unreachable!("loop returns or continues on every path")
}

// ---------------------------------------------------------------------------
// Firecrawl search: ONE call that both finds the top results and returns
// each page's full text as markdown. This replaces the old search + dedupe
// + rank + fetch stages. Docs: https://docs.firecrawl.dev
// Response shape parsing lives in `answerbot::parse_firecrawl_web`.
// ---------------------------------------------------------------------------
async fn search(client: &reqwest::Client, query: &str, registry: &mut Vec<Source>) -> Result<()> {
    let key = std::env::var("FIRECRAWL_API_KEY").context("FIRECRAWL_API_KEY not set")?;
    let body = json!({
        "query": query,
        "limit": MAX_SOURCES_PER_SEARCH,
        "sources": [{ "type": "web" }],
        "scrapeOptions": { "formats": ["markdown"], "onlyMainContent": true }
    });
    let resp: Value = post_json(client, "https://api.firecrawl.dev/v2/search", &key, &body).await?;

    let results = parse_firecrawl_web(&resp)?;

    for r in results {
        // Dedup: skip anything we've already registered (the registry IS the set).
        if registry_contains_url(registry, &r.url) {
            continue;
        }
        // Prefer full page text; fall back to the snippet if scraping failed.
        let mut content = r.markdown.or(r.description).unwrap_or_default();
        truncate_content(&mut content, MAX_CHARS_PER_SOURCE);
        if content.is_empty() {
            continue;
        }
        let id = next_source_id(registry);
        journal(json!({ "event": "source", "id": id, "url": r.url, "query": query }));
        registry.push(Source {
            id,
            url: r.url,
            title: r.title,
            content,
        });
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env from the project root into the process environment. Missing
    // file is fine (variables may already be set in the shell); a present but
    // malformed file is a real error.
    if let Err(e) = dotenvy::dotenv() {
        if !e.not_found() {
            return Err(e).context("failed to load .env");
        }
    }

    let config = load_config()?;

    // 1. You ask -----------------------------------------------------------
    let question: String = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if question.is_empty() {
        bail!("usage: answerbot \"your question\"");
    }
    journal(json!({ "event": "question", "text": question }));
    let anchored = rewrite_with_anchor(&question, &TODAY);
    if anchored != question {
        journal(json!({
            "event": "anchor",
            "original": question,
            "rewritten": anchored,
            "today": *TODAY,
        }));
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()?;

    // 2. Rewrite the question into one good search query -------------------
    let query = rewrite_llm(&client, &config, &query_system_prompt(&TODAY), &anchored).await?;
    eprintln!("searching: {query}");
    journal(json!({ "event": "query", "text": query }));

    // 3. One trip to Firecrawl (find + read pages) --------------------------
    let mut registry: Vec<Source> = Vec::new();
    search(&client, &query, &mut registry).await?;
    if registry.is_empty() {
        bail!("search returned no usable pages — try rephrasing the question");
    }

    // 4. Answer — with exactly one re-search allowed ------------------------
    let system = answer_system_prompt(&TODAY);
    let mut answer = llm(
        &client,
        &config,
        &system,
        &answer_prompt(&anchored, &registry, false),
    )
    .await?;

    if let Some(new_query) = parse_requery(&answer) {
        eprintln!("searching again: {new_query}");
        journal(json!({ "event": "requery", "text": new_query }));
        search(&client, &new_query, &mut registry).await?;
        answer = llm(
            &client,
            &config,
            &system,
            &answer_prompt(&anchored, &registry, true),
        )
        .await?;
    }

    // 5. Honest citations: strip any [Sn] that isn't a real source ----------
    let mut clean = strip_invalid_citations(&answer, &registry);

    // After the one allowed re-search, a further SEARCH: must not be printed
    // — and must not burn a citation-retry LLM call first.
    if should_reject_late_requery(&clean) {
        bail!("model requested another search after the one allowed re-search");
    }

    // 5b. One retry if the model produced a correct-looking answer with zero
    // citations (small models sometimes ignore the citation rule).
    if !has_citations(&clean) && !registry.is_empty() {
        journal(json!({ "event": "no_citations_retry" }));
        eprintln!("retry: previous answer had no citations");
        let retry_prompt = format!(
            "{}\n\nIMPORTANT: Your previous answer contained zero source citations. \
             Every factual claim must end with [Sn] matching a source above. \
             Rewrite your answer now with citations.",
            answer_prompt(&anchored, &registry, true),
        );
        let retry = llm(&client, &config, &system, &retry_prompt).await?;
        clean = strip_invalid_citations(&retry, &registry);
        // Citation retry also insists; reject a SEARCH: from that path too.
        if should_reject_late_requery(&clean) {
            bail!("model requested another search after the one allowed re-search");
        }
    }

    // 6. Print the answer + a source list built from the real registry ------
    println!("\n{clean}\n\nSources:");
    for s in &registry {
        let tag = format!("[{}]", s.id);
        if clean.contains(&tag) {
            println!("  {tag} {} — {}", s.title, s.url);
        }
    }
    journal(json!({ "event": "answer", "text": clean }));
    Ok(())
}
