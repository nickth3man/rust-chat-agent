// Tests for the pure orchestration helpers in src/lib.rs: parse_config,
// parse_requery, registry_contains_url, next_source_id, has_citations,
// truncate_content.

use answerbot::{
    extract_answer_text, has_citations, next_source_id, parse_config, parse_requery,
    registry_contains_url, truncate_content,
};
mod common;

use common::{full_src, src};

// -- extract_answer_text: content-trimming helper for LLM retry ---------

#[test]
fn extract_answer_text_returns_trimmed_content() {
    assert_eq!(
        extract_answer_text(Some("  Paris\n")),
        Some("Paris".to_string())
    );
}

#[test]
fn extract_answer_text_returns_typical_content() {
    assert_eq!(
        extract_answer_text(Some("The capital is Paris.")),
        Some("The capital is Paris.".to_string())
    );
}

#[test]
fn extract_answer_text_none_when_missing() {
    assert_eq!(extract_answer_text(None), None);
}

#[test]
fn extract_answer_text_none_when_empty() {
    assert_eq!(extract_answer_text(Some("")), None);
}

#[test]
fn extract_answer_text_none_when_whitespace_only() {
    // Regression: previously, Some("   ") bypassed the LLM retry loop and
    // was returned as Ok(""). It must now classify as None so the existing
    // empty/reasoning-only retry classification runs.
    assert_eq!(extract_answer_text(Some("   \n\t ")), None);
}

/// Compare two f64 values within machine epsilon. clippy denies `assert_eq!` on floats.
fn assert_f64_eq(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= f64::EPSILON,
        "assertion failed: `(left ≈ right)`\n  left: `{a}`\n right: `{b}`",
    );
}

// -- parse_config: JSON config parsing / error paths / defaults ------------

#[test]
fn parse_config_valid_json() {
    let config = parse_config(r#"{"model": "test-model"}"#).unwrap();
    assert_eq!(config.model, "test-model");
    assert_f64_eq(config.temperature, 0.7); // default
    assert!(config.reasoning); // default true
}

#[test]
fn parse_config_all_fields() {
    let config = parse_config(r#"{"model": "m", "temperature": 0.5, "reasoning": false}"#).unwrap();
    assert_eq!(config.model, "m");
    assert_f64_eq(config.temperature, 0.5);
    assert!(!config.reasoning);
}

#[test]
fn parse_config_invalid_json_is_error() {
    let result = parse_config("not-json-at-all");
    assert!(result.is_err(), "invalid JSON must produce an error");
}

#[test]
fn parse_config_empty_string_is_error() {
    let result = parse_config("");
    assert!(result.is_err(), "empty string must produce an error");
}

#[test]
fn parse_config_extra_fields_ignored() {
    let config =
        parse_config(r#"{"model": "m", "nonexistent": "ignored", "temperature": 0.1}"#).unwrap();
    assert_eq!(config.model, "m");
    assert_f64_eq(config.temperature, 0.1);
}

#[test]
fn parse_config_missing_model_is_error() {
    let result = parse_config(r#"{"temperature": 0.5}"#);
    assert!(result.is_err(), "missing required field 'model' must error");
}

// -- parse_requery: SEARCH: prefix extraction -------------------------------

#[test]
fn parse_requery_typical() {
    assert_eq!(
        parse_requery("SEARCH: rust latest version"),
        Some("rust latest version".into()),
    );
}

#[test]
fn parse_requery_trims_whitespace() {
    assert_eq!(parse_requery("SEARCH:   rust  "), Some("rust".into()),);
}

#[test]
fn parse_requery_just_prefix_no_query() {
    // strip_prefix("SEARCH:") on "SEARCH:" returns Some(""), trimmed to ""
    assert_eq!(parse_requery("SEARCH:"), Some(String::new()));
}

#[test]
fn parse_requery_prefix_only_whitespace() {
    assert_eq!(parse_requery("SEARCH:   "), Some(String::new()));
}

#[test]
fn parse_requery_no_prefix_returns_none() {
    assert_eq!(parse_requery("I don't know"), None);
}

#[test]
fn parse_requery_empty_string_returns_none() {
    assert_eq!(parse_requery(""), None);
}

#[test]
fn parse_requery_case_sensitive() {
    // Only uppercase "SEARCH:" triggers; lower/mixed case does not.
    assert_eq!(parse_requery("search: foo"), None);
    assert_eq!(parse_requery("Search: foo"), None);
}

#[test]
fn parse_requery_embedded_does_not_match() {
    // "SEARCH:" must be at the start of the string.
    assert_eq!(parse_requery("Do SEARCH: again"), None);
}

#[test]
fn parse_requery_preserves_query_content_verbatim() {
    // The extracted query retains punctuation, Unicode, etc.
    let q = "SEARCH: What's the latest? (2026)";
    assert_eq!(parse_requery(q), Some("What's the latest? (2026)".into()),);
}

// -- registry_contains_url: dedup helper -----------------------------------

#[test]
fn registry_contains_url_empty_registry() {
    assert!(!registry_contains_url(&[], "https://example.com"));
}

#[test]
fn registry_contains_url_found() {
    let reg = [full_src("S1", "", "https://example.com/page", "")];
    assert!(registry_contains_url(&reg, "https://example.com/page"));
}

#[test]
fn registry_contains_url_not_found() {
    let reg = [full_src("S1", "", "https://example.com/page", "")];
    assert!(!registry_contains_url(&reg, "https://other.com"));
}

#[test]
fn registry_contains_url_case_sensitive() {
    let reg = [full_src("S1", "", "https://Example.com", "")];
    // Different case = no match (URLs are case-sensitive per spec).
    assert!(!registry_contains_url(&reg, "https://example.com"));
}

#[test]
fn registry_contains_url_trailing_slash_matters() {
    let reg = [full_src("S1", "", "https://example.com/page/", "")];
    assert!(!registry_contains_url(&reg, "https://example.com/page"));
}

#[test]
fn registry_contains_url_multiple_sources() {
    let reg = [
        full_src("S1", "", "https://a.com", ""),
        full_src("S2", "", "https://b.com", ""),
        full_src("S3", "", "https://c.com", ""),
    ];
    assert!(registry_contains_url(&reg, "https://b.com"));
    assert!(!registry_contains_url(&reg, "https://d.com"));
}

// -- next_source_id: sequential ID generation -----------------------------

#[test]
fn next_source_id_empty_registry() {
    assert_eq!(next_source_id(&[]), "S1");
}

#[test]
fn next_source_id_one_source() {
    let reg = [src("S1")];
    assert_eq!(next_source_id(&reg), "S2");
}

#[test]
fn next_source_id_five_sources() {
    let reg = (1..=5).map(|i| src(&format!("S{i}"))).collect::<Vec<_>>();
    assert_eq!(next_source_id(&reg), "S6");
}

#[test]
#[should_panic(expected = "registry invariant violated")]
fn next_source_id_non_contiguous_panics_in_debug() {
    // The contiguous-ID invariant is enforced by a debug_assert! (audit M-04).
    // In debug builds (the default for `cargo test`), non-contiguous input
    // must panic instead of silently producing a colliding ID. In release
    // builds the assertion is compiled out and this test would fail; that is
    // expected and the documented workflow is debug-only tests.
    let reg = [src("S1"), src("S10")];
    let _ = next_source_id(&reg);
}

// -- has_citations: zero-citation retry gate --------------------------------

#[test]
fn has_citations_empty_string() {
    assert!(!has_citations(""));
}

#[test]
fn has_citations_typical_valid() {
    assert!(has_citations("Paris is the capital [S1]"));
}

#[test]
fn has_citations_no_citation_text() {
    assert!(!has_citations("Just some text without any source markers"));
}

#[test]
fn has_citations_partial_s_at_end_does_not_match() {
    // `[S` alone (no digits, no closing bracket) is no longer recognized —
    // the retry gate matches exactly what the registry / footer emit: [Sn].
    assert!(!has_citations("some text [S"));
}

#[test]
fn has_citations_lowercase_only() {
    // Lowercase [s1] is not a valid citation format; only capital [Sn] counts.
    assert!(!has_citations("[s1] test"));
}

#[test]
fn has_citations_numeric_without_bracket() {
    // Bare "S1" without bracket should not match.
    assert!(!has_citations("source S1 is great"));
}

#[test]
fn has_citations_rejects_malformed_braces() {
    // Partial or malformed "[S..." patterns no longer trigger the gate;
    // only well-formed [Sn] citations count. Audit M-05.
    assert!(!has_citations("[S"));
    assert!(!has_citations("[Sabc"));
    assert!(!has_citations("[S ]"));
    assert!(!has_citations("[S1"));
    assert!(!has_citations("[S1a]"));
    assert!(!has_citations("S1]"));
}

#[test]
fn has_citations_well_formed_matches() {
    // Sanity: well-formed capital-S citations DO match.
    assert!(has_citations("[S1]"));
    assert!(has_citations("hello [S23] world"));
    assert!(has_citations("[S1][S2]"));
    assert!(has_citations("multi\nline [S10]\nanswer"));
}

// -- truncate_content: char-boundary-safe wrapper around String::truncate ----

#[test]
fn truncate_content_under_limit_unchanged() {
    let mut s = "hello".to_string();
    truncate_content(&mut s, 10);
    assert_eq!(s, "hello");
}

#[test]
fn truncate_content_exact_limit() {
    let mut s = "hello".to_string();
    truncate_content(&mut s, 5);
    assert_eq!(s, "hello");
}

#[test]
fn truncate_content_over_limit() {
    let mut s = "hello world".to_string();
    truncate_content(&mut s, 5);
    assert_eq!(s, "hello");
}

#[test]
fn truncate_content_empty_string() {
    let mut s = String::new();
    truncate_content(&mut s, 100);
    assert_eq!(s, "");
}

#[test]
fn truncate_content_ascii_exact_boundary() {
    let mut s = "abcdefghij".to_string();
    truncate_content(&mut s, 5);
    assert_eq!(s, "abcde");
}

/// Truncating at a non-char boundary rounds down instead of panicking.
/// "héllo" byte layout: 68 | C3 A9 | 6C 6C 6F (é = bytes 1-2).
#[test]
fn truncate_content_safe_at_non_char_boundary() {
    let mut s = "héllo".to_string();
    truncate_content(&mut s, 2); // byte 2 is in the middle of 2-byte 'é'
    assert_eq!(
        s, "h",
        "should truncate safely at char boundary (byte 0, before 'é')"
    );
}

/// Multiple multi-byte chars: truncation rounds down correctly.
#[test]
fn truncate_content_four_byte_char_boundary() {
    let mut s = "𐍈hello".to_string(); // 𐍈 is 4 bytes (F0 90 8D 88)
    truncate_content(&mut s, 3); // byte 3 is in the middle of 𐍈
    assert_eq!(s, "", "3 bytes lands inside 4-byte char, should round to 0");
    let mut s = "𐍈hello".to_string();
    truncate_content(&mut s, 4); // exactly at the 𐍈 boundary
    assert_eq!(s, "𐍈", "4 bytes is exactly the 4-byte char boundary");
}
