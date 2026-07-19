// answerbot — the whole system in one file.
//
// Flow (matches the diagram):
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
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_SOURCES_PER_SEARCH: usize = 4; // pages Firecrawl reads per trip
const MAX_CHARS_PER_SOURCE: usize = 8_000; // truncate long pages ("safe limits")
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// One fetched page. The `id` (S1, S2, ...) is what the answer cites.
struct Source {
    id: String,
    url: String,
    title: String,
    content: String,
}

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

/// Format the registry into the evidence block the AI reads before answering.
fn evidence_block(registry: &[Source]) -> String {
    registry
        .iter()
        .map(|s| format!("[{}] {} ({})\n{}\n", s.id, s.title, s.url, s.content))
        .collect::<Vec<_>>()
        .join("\n---\n")
}

/// Build the prompt sent to the answering LLM. `insist` adds the "you must
/// answer now" suffix used after a re-search.
fn answer_prompt(question: &str, registry: &[Source], insist: bool) -> String {
    let suffix = if insist {
        "\n\nYou must answer now using the sources above. Do not request another search."
    } else {
        ""
    };
    format!(
        "Question: {question}\n\nSources:\n{}{suffix}",
        evidence_block(registry)
    )
}

const ANSWER_SYSTEM: &str = "You are a research assistant. Answer the user's question \
using ONLY the provided sources. Cite every factual claim with its source id in \
brackets, e.g. [S1]. If — and only if — the sources genuinely cannot answer the \
question, reply with exactly one line: SEARCH: <a better search query>. Otherwise, \
answer directly. Never invent sources.";

const QUERY_SYSTEM: &str = "Rewrite the user's question as a short, effective web \
search query. Reply with the query only — no quotes, no explanation.";

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

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()?;

    // 2. Rewrite the question into one good search query -------------------
    let query = llm(&client, QUERY_SYSTEM, &question).await?;
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
        ANSWER_SYSTEM,
        &answer_prompt(&question, &registry, false),
    )
    .await?;

    if let Some(new_query) = answer.strip_prefix("SEARCH:") {
        let new_query = new_query.trim();
        eprintln!("searching again: {new_query}");
        journal(json!({ "event": "requery", "text": new_query }));
        search(&client, new_query, &mut registry).await?;
        answer = llm(
            &client,
            ANSWER_SYSTEM,
            &answer_prompt(&question, &registry, true),
        )
        .await?;
    }

    // 5. Honest citations: strip any [Sn] that isn't a real source ----------
    let cite_re = Regex::new(r"\[S\d+\]")?;
    let valid: HashSet<&str> = registry.iter().map(|s| s.id.as_str()).collect();
    let clean = cite_re.replace_all(&answer, |c: &regex::Captures| {
        let tag = &c[0];
        let id = &tag[1..tag.len() - 1]; // "[S1]" -> "S1"
        if valid.contains(id) {
            tag.to_string()
        } else {
            String::new()
        }
    });

    // 6. Print the answer + a source list built from the real registry ------
    println!("\n{clean}\n\nSources:");
    for s in &registry {
        if clean.contains(&format!("[{}]", s.id)) {
            println!("  [{}] {} — {}", s.id, s.title, s.url);
        }
    }
    journal(json!({ "event": "answer", "text": clean.to_string() }));
    Ok(())
}
