// Firecrawl search: ONE call that both finds the top results and returns
// each page's full text as markdown. This replaces the old search + dedupe
// + rank + fetch stages. Docs: https://docs.firecrawl.dev
// Response shape parsing lives in `crate::parse_firecrawl_web`.

use crate::http::post_json;
use crate::journal::journal;
use crate::{ingest_web_results, parse_firecrawl_web, Source};
use anyhow::Result;
use serde_json::{json, Value};

pub const MAX_SOURCES_PER_SEARCH: usize = 4; // pages Firecrawl reads per trip
pub const MAX_CHARS_PER_SOURCE: usize = 8_000; // truncate long pages ("safe limits")

/// Everything a Firecrawl search call needs, bundled for the same reason as
/// `llm::OpenRouterCtx`: keeps call sites under clippy's argument-count lint
/// and lets tests point `url`/`api_key`/`journal_path` at fixtures.
pub struct SearchCtx<'a> {
    pub client: &'a reqwest::Client,
    pub url: &'a str,
    pub api_key: &'a str,
    pub journal_path: &'a str,
    pub network_max_attempts: u32,
    pub skip_sleep: bool,
}

/// Search + read pages for `query`, appending newly discovered sources to
/// `registry` (deduped by URL) and journaling one `source` event per
/// addition.
pub async fn search(ctx: &SearchCtx<'_>, query: &str, registry: &mut Vec<Source>) -> Result<()> {
    let body = json!({
        "query": query,
        "limit": MAX_SOURCES_PER_SEARCH,
        "sources": [{ "type": "web" }],
        "scrapeOptions": { "formats": ["markdown"], "onlyMainContent": true }
    });
    let resp: Value = post_json(
        ctx.client,
        ctx.url,
        ctx.api_key,
        &body,
        ctx.journal_path,
        ctx.network_max_attempts,
        ctx.skip_sleep,
    )
    .await?;

    let results = parse_firecrawl_web(&resp)?;
    let before = registry.len();
    let added = ingest_web_results(registry, results, MAX_CHARS_PER_SOURCE);
    for s in &registry[before..before + added] {
        journal(
            ctx.journal_path,
            json!({ "event": "source", "id": s.id, "url": s.url, "query": query }),
        );
    }
    Ok(())
}
