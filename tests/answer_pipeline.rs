// Integration tests for the LLM-facing helpers in src/lib.rs:
//   - evidence_block: formats the source registry for the prompt
//   - answer_prompt: builds the question + sources prompt with optional insist
//   - strip_invalid_citations: validates [Sn] tags against the actual registry

use answerbot::{answer_prompt, evidence_block, strip_invalid_citations, Source};

fn src(id: &str) -> Source {
    Source {
        id: id.into(),
        url: String::new(),
        title: String::new(),
        content: String::new(),
    }
}

/// Run `strip_invalid_citations` and panic on regex failure. The pattern is
/// a hardcoded literal, so this branch is unreachable in practice.
fn must_strip(answer: &str, registry: &[Source]) -> String {
    match strip_invalid_citations(answer, registry) {
        Ok(s) => s,
        Err(e) => panic!("hardcoded citation regex must compile: {e}"),
    }
}

// -- evidence_block: registry composition ------------------------------

#[test]
fn evidence_block_empty_registry_is_empty() {
    assert_eq!(evidence_block(&[]), "");
}

#[test]
fn evidence_block_single_source_has_no_separator() {
    let reg = [src("S1")];
    let out = evidence_block(&reg);
    assert!(out.contains("[S1]"), "missing id tag in {out:?}");
    assert!(!out.contains("---"), "no separator expected in {out:?}");
}

#[test]
fn evidence_block_separates_sources_with_dashes() {
    let reg = [src("S1"), src("S2"), src("S3")];
    let out = evidence_block(&reg);
    assert!(out.contains("[S1]") && out.contains("[S2]") && out.contains("[S3]"));
    assert!(out.contains("\n---\n"), "expected --- separator in {out:?}");
}

// -- answer_prompt: insist toggle --------------------------------------

#[test]
fn answer_prompt_omits_insist_suffix_by_default() {
    let reg = [src("S1")];
    let out = answer_prompt("What?", &reg, false);
    assert!(out.contains("Question: What?"));
    assert!(!out.contains("You must answer now"));
}

#[test]
fn answer_prompt_adds_insist_suffix_when_requested() {
    let reg = [src("S1")];
    let out = answer_prompt("What?", &reg, true);
    assert!(out.contains("Question: What?"));
    assert!(out.contains("You must answer now"));
    assert!(out.contains("Do not request another search"));
}

// -- strip_invalid_citations: registry-membership guarantee -------------

#[test]
fn strip_invalid_citations_keeps_only_registered_ids() {
    let reg = [src("S1"), src("S2")];
    let out = must_strip("[S1] real [S99] fake [S2] real", &reg);
    assert_eq!(out, "[S1] real  fake [S2] real");
}

#[test]
fn strip_invalid_citations_is_case_sensitive() {
    // The regex is \[S\d+\]; lowercase tags don't match and pass through.
    let reg = [src("S1")];
    let out = must_strip("[s1] lowercase [S1] real", &reg);
    assert_eq!(out, "[s1] lowercase [S1] real");
}

#[test]
fn strip_invalid_citations_supports_multi_digit_ids() {
    let reg = [src("S10"), src("S11")];
    let out = must_strip("[S10] a [S11] b [S9] c", &reg);
    assert_eq!(out, "[S10] a [S11] b  c");
}
