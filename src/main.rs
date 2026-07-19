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

use answerbot::{
    answer_prompt, answer_system_prompt, has_citations, next_source_id, parse_config,
    parse_requery, registry_contains_url, rewrite_with_anchor, strip_invalid_citations,
    truncate_content, Config, Source,
};

const MAX_SOURCES_PER_SEARCH: usize = 4; // pages Firecrawl reads per trip
const MAX_CHARS_PER_SOURCE: usize = 8_000; // truncate long pages ("safe limits")
const REQUEST_TIMEOUT_SECS: u64 = 30;

// ---------------------------------------------------------------------------
// Model configuration — loaded from config/models.json at startup.
// The Config struct and parsing live in src/lib.rs for testability.
// ---------------------------------------------------------------------------
fn load_config() -> Result<Config> {
    let path = "config/models.json";
    let contents =
        std::fs::read_to_string(path).with_context(|| format!("failed to read {path}"))?;
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
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("journal.jsonl")
    {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{event}") {
                eprintln!("warning: journal write failed: {e}");
            }
        }
        Err(e) => eprintln!("warning: could not open journal.jsonl: {e}"),
    }
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

/// POST a JSON body with bearer auth, return the parsed JSON response.
/// Shared by the `OpenRouter` chat-completions call and the Firecrawl
/// search call so HTTP-status and timeout handling lives in one place.
async fn post_json(
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

/// Reasoning/thinking models return their chain-of-thought in the `reasoning`
/// field. When the model is configured as reasoning, journal that text if the
/// provider supplied one. No-op for non-reasoning models.
fn journal_reasoning(resp: &ChatResponse, config: &Config) {
    if !config.reasoning {
        return;
    }
    if let Some(text) = resp
        .choices
        .first()
        .and_then(|c| c.message.reasoning.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
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
async fn llm(
    client: &reqwest::Client,
    config: &Config,
    system: &str,
    user: &str,
) -> Result<String> {
    let resp: ChatResponse =
        openrouter_call(client, &chat_body(config, 1500, system, user)).await?;
    let text = resp
        .choices
        .first()
        .and_then(|c| c.message.content.as_deref())
        .context("no text in LLM response")?
        .trim()
        .to_string();
    journal_reasoning(&resp, config);
    Ok(text)
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

/// Forced-tool-call LLM for query rewriting. Sends `tool_choice: "required"`
/// so the model must respond with a `generate_search_query` tool call.
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
    let resp: ChatResponse = openrouter_call(client, &body).await?;
    let tc = resp
        .choices
        .first()
        .and_then(|c| c.message.tool_calls.as_ref())
        .and_then(|tcs| tcs.first())
        .context("rewrite: model did not return a tool call")?;
    let args: Value = serde_json::from_str(&tc.function.arguments)
        .context("rewrite: tool call missing valid arguments JSON")?;
    journal_reasoning(&resp, config);
    args["query"]
        .as_str()
        .map(|s| s.trim().to_string())
        .context("rewrite: tool call arguments missing 'query' field")
}

// ---------------------------------------------------------------------------
// Firecrawl search: ONE call that both finds the top results and returns
// each page's full text as markdown. This replaces the old search + dedupe
// + rank + fetch stages. Docs: https://docs.firecrawl.dev
// ---------------------------------------------------------------------------
#[derive(Deserialize)]
struct FcResult {
    url: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    markdown: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

async fn search(client: &reqwest::Client, query: &str, registry: &mut Vec<Source>) -> Result<()> {
    let key = std::env::var("FIRECRAWL_API_KEY").context("FIRECRAWL_API_KEY not set")?;
    let body = json!({
        "query": query,
        "limit": MAX_SOURCES_PER_SEARCH,
        "sources": [{ "type": "web" }],
        "scrapeOptions": { "formats": ["markdown"], "onlyMainContent": true }
    });
    let resp: Value = post_json(client, "https://api.firecrawl.dev/v2/search", &key, &body).await?;

    // Surface response-shape drift as an explicit error instead of masking it
    // as "no usable pages". Firecrawl's /v2/search contract is documented to
    // return { data: { web: [ { url, title, markdown, ... }, ... ] } }; if
    // the field is missing or the wrong type, propagate a context-rich error
    // so the cause is diagnosable instead of looking like empty content.
    let web_value = resp
        .get("data")
        .and_then(|d| d.get("web"))
        .filter(|v| !v.is_null())
        .context("Firecrawl /v2/search: missing or null data.web field")?;
    let results: Vec<FcResult> = serde_json::from_value(web_value.clone())
        .context("Firecrawl /v2/search: data.web entries failed to deserialize")?;

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

const QUERY_SYSTEM: &str = "You are a search query generator. Rewrite the \
user's question as a short, effective web search query, using the \
`generate_search_query` tool.\n\n\
Rules (follow every rule):\n\
- Keep the query concise \u{2014} 1\u{2013}6 words for best results.\n\
- Write plain natural language phrases only.\n\
- Do NOT use search operator syntax: site:, filetype:, inurl:, \
  intitle:, OR, AND, NOT, -, or quotation marks.\n\
- Do NOT write sentences \u{2014} write only the query.\n\
- Include the year for questions about recent events or dates.\n\n\
Examples:\n\
  Question: What is the capital of France?\n\
  Query: capital of France\n\n\
  Question: Latest news on the Rust Foundation\n\
  Query: Rust Foundation news 2026\n\n\
  Question: What is the latest version of Rust?\n\
  Query: latest Rust version 2026";

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
    let query = rewrite_llm(&client, &config, QUERY_SYSTEM, &anchored).await?;
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

    // Treat an empty requery (the model emits `SEARCH:` with no following
    // text) as "no requery" so we don't fire a billed Firecrawl call with an
    // empty query string. See audit H-02.
    if let Some(new_query) = parse_requery(&answer).filter(|q| !q.is_empty()) {
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
