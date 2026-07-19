// answerbot — the pure LLM-facing helpers extracted from the binary.
//
// These types and functions have no I/O, no env access, and no network. They
// live in the library crate so the integration tests in `tests/` can
// exercise them without standing up the full binary. The binary in
// `src/main.rs` imports them and owns the orchestration: env loading, LLM
// calls, Firecrawl calls, journaling, and printing.

use anyhow::Result;
use regex::Regex;
use std::collections::HashSet;

/// One fetched page. The `id` (S1, S2, ...) is what the answer cites.
pub struct Source {
    pub id: String,
    pub url: String,
    pub title: String,
    pub content: String,
}

/// Format the registry into the evidence block the AI reads before answering.
pub fn evidence_block(registry: &[Source]) -> String {
    registry
        .iter()
        .map(|s| format!("[{}] {} ({})\n{}\n", s.id, s.title, s.url, s.content))
        .collect::<Vec<_>>()
        .join("\n---\n")
}

/// Strip any `[Sn]` citation that doesn't point to a real source in the
/// registry. This is the "honest citations" guarantee: the printed answer
/// can only cite pages that were actually fetched.
pub fn strip_invalid_citations(answer: &str, registry: &[Source]) -> Result<String> {
    let cite_re = Regex::new(r"\[S\d+\]")?;
    let valid: HashSet<&str> = registry.iter().map(|s| s.id.as_str()).collect();
    Ok(cite_re
        .replace_all(answer, |c: &regex::Captures| {
            let tag = &c[0];
            let id = &tag[1..tag.len() - 1]; // "[S1]" -> "S1"
            if valid.contains(id) {
                tag.to_string()
            } else {
                String::new()
            }
        })
        .into_owned())
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
// Temporal anchor helpers (Option #6 implementation). See the plan in
// the conversation or the README for design rationale.
// ---------------------------------------------------------------------------

/// Phrases that imply a relative-time anchor. When any are present
/// (case-insensitive) and the question does not already contain "as of",
/// `rewrite_with_anchor` suffixes the question with `(as of <today>)` so the
/// search-rewrite step has a concrete date to relativize "latest", "recent",
/// etc. against. Conservative list — common false positives ("current" as in
/// "current directory", "this week's weather", etc.) are deliberately omitted.
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
/// the `{{current_date}}` placeholder. Mirrors the `FastChat` convention:
/// per-render substitution of a placeholder in a static template.
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

const ANSWER_SYSTEM_TEMPLATE: &str = "You are a research assistant. Answer the user's question \
using ONLY the provided sources. Cite every factual claim with its source id in \
brackets, e.g. [S1]. If — and only if — the sources genuinely cannot answer the \
question, reply with exactly one line: SEARCH: <a better search query>. Otherwise, \
answer directly. Never invent sources.\n\nThe current date is {{current_date}}.";
