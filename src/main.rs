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
// Config: values are read from a .env file at the project root at startup
// (OPENROUTER_API_KEY, OPENROUTER_MODEL, FIRECRAWL_API_KEY).

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::Write;
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use answerbot::{
    answer_prompt, answer_system_prompt, rewrite_with_anchor, strip_invalid_citations, Source,
};

const MAX_SOURCES_PER_SEARCH: usize = 4; // pages Firecrawl reads per trip
const MAX_CHARS_PER_SOURCE: usize = 8_000; // truncate long pages ("safe limits")
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// Today's date in the user's local timezone as YYYY-MM-DD. Computed once per
/// process — this binary is single-shot per `cargo run`, so per-call refresh
/// buys nothing and would invalidate prompt-cache prefixes that the LLM
/// provider may otherwise share across sessions.
static TODAY: LazyLock<String> =
    LazyLock::new(|| chrono::Local::now().format("%Y-%m-%d").to_string());

// ---------------------------------------------------------------------------
// The journal: one JSON line per event. This single file is the audit trail,
// the dedup record, and the citation registry's paper trail all at once.
// ---------------------------------------------------------------------------
fn journal(mut event: Value) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    event["ts"] = json!(ts);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("journal.jsonl")
    {
        let _ = writeln!(f, "{event}");
    }
}

// ---------------------------------------------------------------------------
// LLM call: OpenRouter chat-completions request (OpenAI-compatible), returns
// the text reply. Docs: https://openrouter.ai/docs/api-reference/overview
// ---------------------------------------------------------------------------
async fn llm(client: &reqwest::Client, system: &str, user: &str) -> Result<String> {
    let key = std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;
    let model = std::env::var("OPENROUTER_MODEL").context("OPENROUTER_MODEL not set")?;
    let body = json!({
        "model": model,
        "max_tokens": 1500,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ]
    });
    let resp: Value = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(key)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let text = resp["choices"][0]["message"]["content"]
        .as_str()
        .context("no text in LLM response")?
        .trim()
        .to_string();
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
/// No sentinel, no fallback \u{2014} the model returns a query or the call fails.
async fn rewrite_llm(client: &reqwest::Client, system: &str, user: &str) -> Result<String> {
    let key = std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;
    let model = std::env::var("OPENROUTER_MODEL").context("OPENROUTER_MODEL not set")?;
    let tool_schema: Value = serde_json::from_str(REWRITE_TOOL)?;
    let body = json!({
        "model": model,
        "max_tokens": 250,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ],
        "tools": [tool_schema],
        "tool_choice": "required"
    });
    let resp: Value = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(key)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let tc = resp["choices"][0]["message"]["tool_calls"]
        .as_array()
        .and_then(|a| a.first())
        .context("rewrite: model did not return a tool call")?;
    let args: Value = serde_json::from_str(
        tc["function"]["arguments"]
            .as_str()
            .context("rewrite: tool call missing arguments")?,
    )?;
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
    let resp: Value = client
        .post("https://api.firecrawl.dev/v2/search")
        .bearer_auth(key)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let results: Vec<FcResult> =
        serde_json::from_value(resp["data"]["web"].clone()).unwrap_or_default();

    for r in results {
        // Dedup: skip anything we've already registered (the registry IS the set).
        if registry.iter().any(|s| s.url == r.url) {
            continue;
        }
        // Prefer full page text; fall back to the snippet if scraping failed.
        let mut content = r.markdown.or(r.description).unwrap_or_default();
        content.truncate(MAX_CHARS_PER_SOURCE);
        if content.is_empty() {
            continue;
        }
        let id = format!("S{}", registry.len() + 1);
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
    let query = rewrite_llm(&client, QUERY_SYSTEM, &anchored).await?;
    eprintln!("searching: {query}");
    journal(json!({ "event": "query", "text": query }));

    // 3. One trip to Firecrawl (find + read pages) --------------------------
    let mut registry: Vec<Source> = Vec::new();
    search(&client, &query, &mut registry).await?;
    if registry.is_empty() {
        bail!("search returned no usable pages — try rephrasing the question");
    }

    // 4. Answer — with exactly one re-search allowed ------------------------
    let mut answer = llm(
        &client,
        &answer_system_prompt(&TODAY),
        &answer_prompt(&anchored, &registry, false),
    )
    .await?;

    if let Some(new_query) = answer.strip_prefix("SEARCH:") {
        let new_query = new_query.trim();
        eprintln!("searching again: {new_query}");
        journal(json!({ "event": "requery", "text": new_query }));
        search(&client, new_query, &mut registry).await?;
        answer = llm(
            &client,
            &answer_system_prompt(&TODAY),
            &answer_prompt(&anchored, &registry, true),
        )
        .await?;
    }

    // 5. Honest citations: strip any [Sn] that isn't a real source ----------
    let clean = strip_invalid_citations(&answer, &registry)?;

    // 6. Print the answer + a source list built from the real registry ------
    println!("\n{clean}\n\nSources:");
    for s in &registry {
        if clean.contains(&format!("[{}]", s.id)) {
            println!("  [{}] {} — {}", s.id, s.title, s.url);
        }
    }
    journal(json!({ "event": "answer", "text": clean }));
    Ok(())
}
