// Integration tests for the LLM-facing helpers in src/lib.rs:
//   - evidence_block: formats the source registry for the prompt
//   - answer_prompt: builds the question + sources prompt with optional insist
//   - strip_invalid_citations: validates [Sn] tags against the actual registry

use answerbot::{
    answer_prompt, answer_system_prompt, evidence_block, rewrite_with_anchor,
    strip_invalid_citations, Source,
};
use regex::Regex;

fn src(id: &str) -> Source {
    Source {
        id: id.into(),
        url: String::new(),
        title: String::new(),
        content: String::new(),
    }
}

fn full_src(id: &str, title: &str, url: &str, content: &str) -> Source {
    Source {
        id: id.into(),
        title: title.into(),
        url: url.into(),
        content: content.into(),
    }
}

/// Run `strip_invalid_citations` and panic on regex failure. The pattern is
/// a hardcoded literal, so this branch is unreachable in practice.
fn must_strip(answer: &str, registry: &[Source]) -> String {
    let Ok(s) = strip_invalid_citations(answer, registry) else {
        panic!("hardcoded citation regex must compile");
    };
    s
}

fn citation_regex() -> Regex {
    let Ok(r) = Regex::new(r"\[S\d+\]") else {
        panic!("hardcoded citation regex must compile");
    };
    r
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

#[test]
fn evidence_block_contains_all_fields() {
    let s = full_src("S1", "My Title", "https://example.com", "Hello world");
    let out = evidence_block(&[s]);
    assert!(out.contains("[S1]"), "id missing in {out:?}");
    assert!(out.contains("My Title"), "title missing in {out:?}");
    assert!(
        out.contains("https://example.com"),
        "url missing in {out:?}"
    );
    assert!(out.contains("Hello world"), "content missing in {out:?}");
}

#[test]
fn evidence_block_preserves_order() {
    let reg = [src("S2"), src("S1"), src("S3")];
    let out = evidence_block(&reg);
    let Some(pos_s2) = out.find("[S2]") else {
        panic!("S2 not found in {out:?}");
    };
    let Some(pos_s1) = out.find("[S1]") else {
        panic!("S1 not found in {out:?}");
    };
    let Some(pos_s3) = out.find("[S3]") else {
        panic!("S3 not found in {out:?}");
    };
    assert!(pos_s2 < pos_s1, "S2 should appear before S1");
    assert!(pos_s1 < pos_s3, "S1 should appear before S3");
}

#[test]
fn evidence_block_separator_count_is_n_minus_one() {
    assert_eq!(
        evidence_block(&[src("S1")]).matches("\n---\n").count(),
        0,
        "single source should have 0 separators"
    );
    assert_eq!(
        evidence_block(&[src("S1"), src("S2")])
            .matches("\n---\n")
            .count(),
        1
    );
    assert_eq!(
        evidence_block(&[src("S1"), src("S2"), src("S3")])
            .matches("\n---\n")
            .count(),
        2
    );
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

#[test]
fn answer_prompt_embeds_evidence_block() {
    let reg = [full_src("S1", "Title", "https://a.com", "content here")];
    let ev = evidence_block(&reg);
    let prompt = answer_prompt("What?", &reg, false);
    assert!(
        prompt.contains(&ev),
        "prompt should embed evidence_block output"
    );
}

#[test]
fn answer_prompt_handles_empty_registry() {
    let reg: [Source; 0] = [];
    let out = answer_prompt("What?", &reg, false);
    assert!(out.contains("Question: What?"));
    assert!(out.contains("Sources:"));
    assert!(
        !out.contains("[S"),
        "empty registry should produce no [S tags"
    );
}

#[test]
fn answer_prompt_orders_evidence_before_insist_suffix() {
    let reg = [src("S1")];
    let out = answer_prompt("Q?", &reg, true);
    let Some(pos_ev) = out.find("[S1]") else {
        panic!("evidence [S1] not found in {out:?}");
    };
    let Some(pos_insist) = out.find("You must answer now") else {
        panic!("insist suffix not found in {out:?}");
    };
    assert!(
        pos_ev < pos_insist,
        "evidence should appear before insist suffix"
    );
}

#[test]
fn answer_prompt_preserves_question_verbatim() {
    let q = "Line1\nLine2 with [S1] ? special: äöü";
    let reg = [src("S1")];
    let out = answer_prompt(q, &reg, false);
    assert!(
        out.contains(q),
        "question should appear verbatim in {out:?}"
    );
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

#[test]
fn strip_invalid_citations_empty_registry_strips_all() {
    let reg: [Source; 0] = [];
    let out = must_strip("[S1][S2] hello", &reg);
    assert_eq!(out, " hello");
    let out2 = must_strip("[S1][S2]", &reg);
    assert_eq!(out2, "");
}

#[test]
fn strip_invalid_citations_empty_answer_returns_empty() {
    let reg = [src("S1")];
    assert_eq!(must_strip("", &reg), "");
    let empty_reg: [Source; 0] = [];
    assert_eq!(must_strip("", &empty_reg), "");
}

#[test]
fn strip_invalid_citations_no_citations_unchanged() {
    let reg = [src("S1")];
    let input = "hello world, no tags here!";
    assert_eq!(must_strip(input, &reg), input);
}

#[test]
fn strip_invalid_citations_adjacent_valid_kept() {
    let reg = [src("S1"), src("S2")];
    assert_eq!(must_strip("[S1][S2]", &reg), "[S1][S2]");
}

#[test]
fn strip_invalid_citations_duplicate_valid_all_kept() {
    let reg = [src("S1")];
    assert_eq!(must_strip("[S1] [S1] [S1]", &reg), "[S1] [S1] [S1]");
}

#[test]
fn strip_invalid_citations_at_boundaries() {
    let reg = [src("S1")];
    assert_eq!(must_strip("[S1] foo", &reg), "[S1] foo");
    assert_eq!(must_strip("foo [S1]", &reg), "foo [S1]");
    assert_eq!(must_strip("[S1]", &reg), "[S1]");
}

#[test]
fn strip_invalid_citations_malformed_pass_through() {
    let reg = [src("S1")];
    let cases = [
        "[S]", "[S ]", "[ S1]", "[S1 ]", "[S-1]", "[S1a]", "[S1", "S1]", "[s1]", "[s1 ]",
    ];
    for c in cases {
        let out = must_strip(c, &reg);
        assert_eq!(out, c, "malformed case {c:?} should pass through unchanged");
    }
}

#[test]
fn strip_invalid_citations_only_valid_remain_in_output() {
    let reg = [src("S1"), src("S2")];
    let input = "[S1] real [S99] fake [S2] real [S3] nope [S10] also nope";
    let out = must_strip(input, &reg);
    let re = citation_regex();
    for mat in re.find_iter(&out) {
        let tag = mat.as_str();
        let id = &tag[1..tag.len() - 1];
        assert!(
            reg.iter().any(|s| s.id == id),
            "found invalid id {id:?} in output {out:?}"
        );
    }
    assert!(!out.contains("[S99]"));
    assert!(!out.contains("[S3]"));
    assert!(!out.contains("[S10]"));
    assert!(out.contains("[S1]"));
    assert!(out.contains("[S2]"));
}

// -- answer_system_prompt: placeholder substitution ----------------------

#[test]
fn answer_system_prompt_substitutes_today_marker() {
    let out = answer_system_prompt("2026-07-18");
    assert!(
        out.contains("The current date is 2026-07-18"),
        "rendered prompt missing date line: {out:?}"
    );
}

#[test]
fn answer_system_prompt_substitutes_exactly_once() {
    let out = answer_system_prompt("2026-07-18");
    assert_eq!(
        out.matches("2026-07-18").count(),
        1,
        "expected exactly one substitution"
    );
}

#[test]
fn answer_system_prompt_leaks_no_placeholder() {
    let out = answer_system_prompt("2026-07-18");
    assert!(
        !out.contains("{{current_date}}"),
        "placeholder should be substituted, got {out:?}"
    );
}

#[test]
fn answer_system_prompt_date_can_be_any_iso_string() {
    assert!(answer_system_prompt("1999-12-31").contains("1999-12-31"));
    assert!(answer_system_prompt("2026-01-01").contains("2026-01-01"));
}

// -- rewrite_with_anchor: relative-time phrase detection -----------------

#[test]
fn rewrite_with_anchor_passthrough_when_no_phrase() {
    assert_eq!(
        rewrite_with_anchor("What is Rust?", "2026-07-18"),
        "What is Rust?"
    );
    assert_eq!(
        rewrite_with_anchor("Who created Python?", "2026-07-18"),
        "Who created Python?"
    );
}

#[test]
fn rewrite_with_anchor_appends_for_latest() {
    assert_eq!(
        rewrite_with_anchor("What is the latest Rust release?", "2026-07-18"),
        "What is the latest Rust release? (as of 2026-07-18)"
    );
}

#[test]
fn rewrite_with_anchor_appends_for_today_and_recent() {
    assert_eq!(
        rewrite_with_anchor("Today's headlines?", "2026-07-18"),
        "Today's headlines? (as of 2026-07-18)"
    );
    assert_eq!(
        rewrite_with_anchor("Recent fusion breakthroughs?", "2026-07-18"),
        "Recent fusion breakthroughs? (as of 2026-07-18)"
    );
}

#[test]
fn rewrite_with_anchor_appends_for_this_year() {
    assert_eq!(
        rewrite_with_anchor("Top languages this year?", "2026-07-18"),
        "Top languages this year? (as of 2026-07-18)"
    );
}

#[test]
fn rewrite_with_anchor_is_case_insensitive() {
    assert_eq!(
        rewrite_with_anchor("LATEST version of Rust", "2026-07-18"),
        "LATEST version of Rust (as of 2026-07-18)"
    );
    assert_eq!(
        rewrite_with_anchor("latest version of Rust", "2026-07-18"),
        "latest version of Rust (as of 2026-07-18)"
    );
}

#[test]
fn rewrite_with_anchor_skips_already_dated_questions() {
    assert_eq!(
        rewrite_with_anchor("What was the latest as of 2024?", "2026-07-18"),
        "What was the latest as of 2024?"
    );
    assert_eq!(
        rewrite_with_anchor("Revenue as of Q1?", "2026-07-18"),
        "Revenue as of Q1?"
    );
}

#[test]
fn rewrite_with_anchor_preserves_question_whitespace() {
    let q = "What is the\nlatest\tRust release?";
    assert_eq!(
        rewrite_with_anchor(q, "2026-07-18"),
        "What is the\nlatest\tRust release? (as of 2026-07-18)"
    );
}

#[test]
fn rewrite_with_anchor_empty_passthrough() {
    assert_eq!(rewrite_with_anchor("", "2026-07-18"), "");
}
