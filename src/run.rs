// The orchestration: env loading, LLM/Firecrawl calls, journaling, printing.
// `run()` is the pure-async body (fully injectable — no env access, no
// hardcoded URLs) so it can be driven from tests against wiremock servers.
// `run_cli()` is the thin env-reading/runtime-building wrapper the real
// binary calls; it is the only place in the crate that reads
// `OPENROUTER_API_KEY` / `FIRECRAWL_API_KEY` / argv / the system clock.

use crate::journal::journal;
use crate::llm::{llm, rewrite_llm, OpenRouterCtx};
use crate::paths::{journal_path, load_config};
use crate::search::{search, SearchCtx};
use crate::{
    answer_prompt, answer_system_prompt, cited_sources, first_answer_decision,
    post_answer_decision, query_system_prompt, rewrite_with_anchor, strip_invalid_citations,
    Config, FirstAnswerDecision, PostAnswerDecision, Source,
};
use anyhow::{bail, Context, Result};
use serde_json::json;
use std::time::Duration;

/// Hard cap per network call. Not the whole run's timeout — see README.
const REQUEST_TIMEOUT_SECS: u64 = 30;

const DEFAULT_OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_FIRECRAWL_URL: &str = "https://api.firecrawl.dev/v2/search";

/// Base URLs for the two upstream services. Overridable via
/// `ANSWERBOT_OPENROUTER_URL` / `ANSWERBOT_FIRECRAWL_URL` so tests (and any
/// future self-hosted proxy) can point at a different host without touching
/// call sites.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoints {
    pub openrouter: String,
    pub firecrawl: String,
}

impl Default for Endpoints {
    fn default() -> Self {
        Self {
            openrouter: DEFAULT_OPENROUTER_URL.to_string(),
            firecrawl: DEFAULT_FIRECRAWL_URL.to_string(),
        }
    }
}

impl Endpoints {
    /// Read overrides from `ANSWERBOT_OPENROUTER_URL` / `ANSWERBOT_FIRECRAWL_URL`,
    /// falling back to the real upstream hosts.
    pub fn from_env() -> Self {
        let default = Self::default();
        Self {
            openrouter: std::env::var("ANSWERBOT_OPENROUTER_URL").unwrap_or(default.openrouter),
            firecrawl: std::env::var("ANSWERBOT_FIRECRAWL_URL").unwrap_or(default.firecrawl),
        }
    }
}

/// Bounded retry caps. Each is the TOTAL number of attempts (1 = no retry).
/// `network` covers transient HTTP failures (timeout, connect, 429, 5xx)
/// inside `post_json`. `rewrite` covers the rewrite step emitting a
/// malformed tool call (Class A: a stochastic small-model failure). `llm`
/// covers the answering step returning no text (Class B: reasoning-only or
/// empty turn from a reasoning model).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttemptCaps {
    pub network: u32,
    pub rewrite: u32,
    pub llm: u32,
}

impl Default for AttemptCaps {
    fn default() -> Self {
        Self {
            network: 3,
            rewrite: 5,
            llm: 3,
        }
    }
}

/// Everything `run()` needs to answer one question. Fully injectable: no
/// field is read from the environment inside `run()` itself — that is
/// `run_cli()`'s job.
pub struct RunInput {
    pub question: String,
    pub config: Config,
    pub today: String,
    pub client: reqwest::Client,
    pub openrouter_key: String,
    pub firecrawl_key: String,
    pub endpoints: Endpoints,
    pub journal_path: String,
    pub caps: AttemptCaps,
    /// When true, retry loops sleep for `Duration::ZERO` instead of the
    /// exponential backoff schedule. Tests only.
    pub skip_sleep: bool,
}

/// Bundle of per-question knobs shared by `answer_with_one_requery` and
/// `resolve_citations`, kept off the argument list so both stay under
/// clippy's argument-count lint.
struct AnswerCtx<'a> {
    config: &'a Config,
    system: &'a str,
    anchored: &'a str,
    caps: AttemptCaps,
    journal_path: &'a str,
}

/// Step 4: answer from `registry`, allowing exactly one `SEARCH:` re-search.
/// Split out of `run()` to keep it under clippy's function-length lint.
async fn answer_with_one_requery(
    openrouter_ctx: &OpenRouterCtx<'_>,
    search_ctx: &SearchCtx<'_>,
    ctx: &AnswerCtx<'_>,
    registry: &mut Vec<Source>,
) -> Result<String> {
    let mut answer = llm(
        openrouter_ctx,
        ctx.config,
        ctx.system,
        &answer_prompt(ctx.anchored, registry, false),
        ctx.caps.llm,
    )
    .await?;

    if let FirstAnswerDecision::Requery(new_query) = first_answer_decision(&answer) {
        eprintln!("searching again: {new_query}");
        journal(
            ctx.journal_path,
            json!({ "event": "requery", "text": new_query }),
        );
        search(search_ctx, &new_query, registry).await?;
        answer = llm(
            openrouter_ctx,
            ctx.config,
            ctx.system,
            &answer_prompt(ctx.anchored, registry, true),
            ctx.caps.llm,
        )
        .await?;
    }
    Ok(answer)
}

/// Steps 5 (strip invalid citations) and the zero-citation retry / late-
/// requery rejection that follows. Split out of `run()` for the same reason
/// as `answer_with_one_requery`.
async fn resolve_citations(
    openrouter_ctx: &OpenRouterCtx<'_>,
    ctx: &AnswerCtx<'_>,
    answer: &str,
    registry: &[Source],
) -> Result<String> {
    let mut clean = strip_invalid_citations(answer, registry);

    // After the one allowed re-search, a further SEARCH: must not be printed
    // — and must not burn a citation-retry LLM call first.
    match post_answer_decision(&clean, registry) {
        PostAnswerDecision::RejectLateRequery => {
            bail!("model requested another search after the one allowed re-search");
        }
        PostAnswerDecision::RetryForCitations => {
            journal(ctx.journal_path, json!({ "event": "no_citations_retry" }));
            eprintln!("retry: previous answer had no citations");
            let retry_prompt = format!(
                "{}\n\nIMPORTANT: Your previous answer contained zero source citations. \
                 Every factual claim must end with [Sn] matching a source above. \
                 Rewrite your answer now with citations.",
                answer_prompt(ctx.anchored, registry, true),
            );
            let retry = llm(
                openrouter_ctx,
                ctx.config,
                ctx.system,
                &retry_prompt,
                ctx.caps.llm,
            )
            .await?;
            clean = strip_invalid_citations(&retry, registry);
            // Citation retry also insists; reject a SEARCH: from that path too.
            if post_answer_decision(&clean, registry) == PostAnswerDecision::RejectLateRequery {
                bail!("model requested another search after the one allowed re-search");
            }
        }
        PostAnswerDecision::Accept => {}
    }
    Ok(clean)
}

/// Answer `input.question`, printing the final answer + sources to stdout
/// and journaling every step to `input.journal_path`.
///
/// Flow (matches the diagram in README.md):
///   1. You ask a question
///   2. The AI rewrites it into one good search query   (LLM call #1)
///   3. Firecrawl searches AND reads the top pages in one trip
///   4. The AI answers with [S1] [S2] citations — or asks for exactly ONE
///      more search if something is missing (LLM call #2, maybe #3)
///   5. Citations pointing at sources that don't exist are stripped
///   6. Every step is appended to the journal
pub async fn run(input: RunInput) -> Result<()> {
    let RunInput {
        question,
        config,
        today,
        client,
        openrouter_key,
        firecrawl_key,
        endpoints,
        journal_path,
        caps,
        skip_sleep,
    } = input;

    // 1. You ask -----------------------------------------------------------
    if question.is_empty() {
        bail!("usage: answerbot \"your question\"");
    }
    journal(
        &journal_path,
        json!({ "event": "question", "text": question }),
    );
    let anchored = rewrite_with_anchor(&question, &today);
    if anchored != question {
        journal(
            &journal_path,
            json!({
                "event": "anchor",
                "original": question,
                "rewritten": anchored,
                "today": today,
            }),
        );
    }

    let openrouter_ctx = OpenRouterCtx {
        client: &client,
        url: &endpoints.openrouter,
        api_key: &openrouter_key,
        journal_path: &journal_path,
        network_max_attempts: caps.network,
        skip_sleep,
    };
    let search_ctx = SearchCtx {
        client: &client,
        url: &endpoints.firecrawl,
        api_key: &firecrawl_key,
        journal_path: &journal_path,
        network_max_attempts: caps.network,
        skip_sleep,
    };

    // 2. Rewrite the question into one good search query -------------------
    let query = rewrite_llm(
        &openrouter_ctx,
        &config,
        &query_system_prompt(&today),
        &anchored,
        caps.rewrite,
    )
    .await?;
    eprintln!("searching: {query}");
    journal(&journal_path, json!({ "event": "query", "text": query }));

    // 3. One trip to Firecrawl (find + read pages) --------------------------
    let mut registry: Vec<Source> = Vec::new();
    search(&search_ctx, &query, &mut registry).await?;
    if registry.is_empty() {
        bail!("search returned no usable pages — try rephrasing the question");
    }

    // 4. Answer — with exactly one re-search allowed ------------------------
    let system = answer_system_prompt(&today);
    let answer_ctx = AnswerCtx {
        config: &config,
        system: &system,
        anchored: &anchored,
        caps,
        journal_path: &journal_path,
    };
    let answer =
        answer_with_one_requery(&openrouter_ctx, &search_ctx, &answer_ctx, &mut registry).await?;

    // 5. Honest citations, zero-citation retry, late-requery rejection ------
    let clean = resolve_citations(&openrouter_ctx, &answer_ctx, &answer, &registry).await?;

    // 6. Print the answer + a source list built from the real registry ------
    println!("\n{clean}\n\nSources:");
    for s in cited_sources(&clean, &registry) {
        println!("  [{}] {} — {}", s.id, s.title, s.url);
    }
    journal(&journal_path, json!({ "event": "answer", "text": clean }));
    Ok(())
}

/// Load `.env` from the process CWD into the process environment. A missing
/// file is fine (variables may already be set in the shell); a present but
/// malformed file is a real error.
pub fn load_dotenv() -> Result<()> {
    if let Err(e) = dotenvy::dotenv() {
        if !e.not_found() {
            return Err(e).context("failed to load .env");
        }
    }
    Ok(())
}

/// The real binary's entry point: load `.env` + `config.toml`, read secrets
/// and argv from the environment, and run the async orchestration on a
/// fresh multi-thread tokio runtime. `main.rs` calls only this.
///
/// # Panics
///
/// Panics if the reqwest client or the tokio multi-thread runtime cannot be
/// built. Those builders only fail under exotic system conditions that are
/// not worth modeling as `Result` for coverage (unreachable `?` regions).
pub fn run_cli() -> Result<()> {
    load_dotenv()?;

    let config = load_config()?;

    let question: String = std::env::args().skip(1).collect::<Vec<_>>().join(" ");

    // Builder defaults we use never fail to construct a client or runtime in
    // practice; use `expect` so LLVM coverage is not blocked by unreachable
    // `?` error regions on these two calls.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()
        .expect("reqwest client with timeout must build");

    let openrouter_key =
        std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;
    let firecrawl_key = std::env::var("FIRECRAWL_API_KEY").context("FIRECRAWL_API_KEY not set")?;
    let endpoints = Endpoints::from_env();
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();

    let input = RunInput {
        question,
        config,
        today,
        client,
        openrouter_key,
        firecrawl_key,
        endpoints,
        journal_path: journal_path(),
        caps: AttemptCaps::default(),
        skip_sleep: false,
    };

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio multi-thread runtime must build")
        .block_on(run(input))
}
