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
