// answerbot — the pure LLM-facing helpers extracted from the binary.
//
// These types and functions have no I/O, no env access, and no network. They
// live in the library crate so the integration tests in `tests/` can
// exercise them without standing up the full binary. The binary in
// `src/main.rs` imports them and owns the orchestration: env loading, LLM
// calls, Firecrawl calls, journaling, and printing.

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::LazyLock;

/// One fetched page. The `id` (S1, S2, ...) is what the answer cites.
pub struct Source {
    pub id: String,
    pub url: String,
    pub title: String,
    pub content: String,
}

/// Regex matching `[Sn]` citation markers in either case (e.g. `[S1]`, `[s1]`).
/// Used by `strip_invalid_citations`. Lowercase variants are matched so they
/// can be removed when (always) found to be invalid — registry IDs are always
/// capital-S (`S1`, `S2`, …), so `[s1]` never names a real source and is
/// always stripped.
static STRIP_CITE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[[Ss]\d+\]").expect("hardcoded citation regex must compile"));

/// Regex matching well-formed capital-S citations `[Sn]`. Used by
/// `has_citations` as the zero-citation retry gate so it matches exactly what
/// `strip_invalid_citations` keeps and what the `Sources:` footer emits.
static HAS_CITATIONS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[S\d+\]").expect("hardcoded citation regex must compile"));

/// Format the registry into the evidence block the AI reads before answering.
pub fn evidence_block(registry: &[Source]) -> String {
    registry
        .iter()
        .map(|s| format!("[{}] {} ({})\n{}\n", s.id, s.title, s.url, s.content))
        .collect::<Vec<_>>()
        .join("\n---\n")
}

/// Strip any `[Sn]`/`[sn]` citation that doesn't point to a real source in
/// the registry. This is the "honest citations" guarantee: the printed
/// answer can only cite pages that were actually fetched. Lowercase markers
/// like `[s1]` are always stripped because registry IDs are capital-S only.
///
/// The regex is a `LazyLock` static (compiled once per process) rather than
/// recompiled per call, so this function returns `String` instead of
/// `Result<String>` — there is no regex-compilation error path left.
pub fn strip_invalid_citations(answer: &str, registry: &[Source]) -> String {
    let valid: HashSet<&str> = registry.iter().map(|s| s.id.as_str()).collect();
    STRIP_CITE_RE
        .replace_all(answer, |c: &regex::Captures| {
            let tag = &c[0];
            let id = &tag[1..tag.len() - 1]; // "[S1]" -> "S1"
            if valid.contains(id) {
                tag.to_string()
            } else {
                String::new()
            }
        })
        .into_owned()
}

/// Build the prompt sent to the answering LLM. `insist` adds the "you must
/// answer now" suffix used after a re-search.
pub fn answer_prompt(question: &str, registry: &[Source], insist: bool) -> String {
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

// ---------------------------------------------------------------------------
// Temporal anchor helpers
// ---------------------------------------------------------------------------

/// Phrases that imply a relative-time anchor. When any are present and the
/// question does not already contain "as of", `rewrite_with_anchor` suffixes
/// the question with `(as of <today>)`.
const ANCHOR_PHRASES: &[&str] = &[
    "latest",
    "recent",
    "recently",
    "today",
    "yesterday",
    "tomorrow",
    "this year",
    "this month",
    "this week",
    "this quarter",
];

/// Render the answering system prompt with the current date injected into
/// the `{{current_date}}` placeholder.
pub fn answer_system_prompt(today: &str) -> String {
    ANSWER_SYSTEM_TEMPLATE.replace("{{current_date}}", today)
}

/// Suffix `"(as of <today>)"` to a question only when (a) the question uses
/// a relative-time phrase, and (b) it does not already pin its own date
/// with "as of". Pure: returns the input unchanged otherwise.
pub fn rewrite_with_anchor(question: &str, today: &str) -> String {
    let lower = question.to_ascii_lowercase();
    if lower.contains("as of") {
        return question.to_string();
    }
    let needs_anchor = ANCHOR_PHRASES.iter().any(|p| lower.contains(p));
    if needs_anchor {
        format!("{question} (as of {today})")
    } else {
        question.to_string()
    }
}

// ---------------------------------------------------------------------------
// Orchestration helpers
// ---------------------------------------------------------------------------

/// Runtime model configuration loaded from config/models.json.
#[derive(Deserialize)]
pub struct Config {
    pub model: String,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Whether the model supports reasoning/thinking (chain-of-thought).
    #[serde(default = "default_true")]
    pub reasoning: bool,
}

fn default_temperature() -> f64 {
    0.7
}

fn default_true() -> bool {
    true
}

/// Parse a Config from raw JSON string contents.
pub fn parse_config(contents: &str) -> Result<Config> {
    serde_json::from_str(contents).context("failed to parse config")
}

/// Extract a re-search query from an answer that starts with `SEARCH:`.
/// Returns `None` when the answer is not a requery request.
pub fn parse_requery(answer: &str) -> Option<String> {
    answer.strip_prefix("SEARCH:").map(|s| s.trim().to_string())
}

/// Check whether a URL is already in the source registry (dedup helper).
pub fn registry_contains_url(registry: &[Source], url: &str) -> bool {
    registry.iter().any(|s| s.url == url)
}

/// Generate the next source ID (`S1`, `S2`, …).
///
/// # Invariant
///
/// The registry MUST be append-only with contiguous IDs starting at `S1`:
/// `registry[i].id == "S{i+1}"`. This function relies on `registry.len()+1`
/// being the next free ID, which only holds if no source is ever removed,
/// re-ranked, or assigned a non-sequential ID. The orchestration in
/// `src/main.rs` preserves this invariant by deduping *before* assigning an
/// ID (see `registry_contains_url`) and never removing entries.
///
/// A `debug_assert!` enforces the invariant so a future regression surfaces
/// in debug builds instead of silently producing colliding IDs.
pub fn next_source_id(registry: &[Source]) -> String {
    debug_assert!(
        registry
            .iter()
            .enumerate()
            .all(|(i, s)| s.id == format!("S{}", i + 1)),
        "registry invariant violated: source IDs must be contiguous S1, S2, ..."
    );
    format!("S{}", registry.len() + 1)
}

/// Check whether an answer contains at least one well-formed `[Sn]` citation
/// (capital S, digits, closing bracket). Used for the zero-citation retry
/// gate; matches exactly what `strip_invalid_citations` keeps and what the
/// `Sources:` footer emits, so lowercase `[s1]` or partial `[S` prose can no
/// longer suppress the retry.
pub fn has_citations(answer: &str) -> bool {
    HAS_CITATIONS_RE.is_match(answer)
}

/// Truncate content to at most `max` bytes, safely rounding down to the
/// nearest UTF-8 character boundary to avoid the panic that
/// `String::truncate` would produce when `max` lands mid-character.
/// No-op when `max >= content.len()`.
pub fn truncate_content(content: &mut String, max: usize) {
    if max < content.len() {
        let mut byte_idx = max;
        while !content.is_char_boundary(byte_idx) {
            byte_idx -= 1;
        }
        content.truncate(byte_idx);
    }
}

const ANSWER_SYSTEM_TEMPLATE: &str = "\
# ROLE
You are a research assistant. Answer ONLY from the sources provided below.

# HARD RULES (mandatory)
1. Every factual claim must end with [Sn] where n matches a source id above.
2. Never invent a source. Never cite a source that was not provided.
3. If the sources cannot answer the question, reply with EXACTLY one line:
       SEARCH: <a better search query>
4. Otherwise, answer in 1-3 short paragraphs. No preamble, no closing.

# EXAMPLE
Q: What is the capital of France?
Sources: [S1] Wikipedia — Paris (https://en.wikipedia.org/wiki/Paris)
A: The capital of France is Paris [S1]. It is in the Île-de-France region [S1].

# REMINDER (read before answering)
Before your final answer, check: did every factual claim get a [Sn] citation?
If any claim is uncited, add the citation now. If no source supports a claim,
return SEARCH: <query> instead.

The current date is {{current_date}}.";
