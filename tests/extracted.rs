// Tests for the pure orchestration helpers in src/lib.rs: parse_config (TOML),
// parse_requery, registry_contains_url, next_source_id, has_citations,
// truncate_content.

use answerbot::{
    cited_sources, extract_answer_text, first_answer_decision, has_citations, ingest_web_results,
    load_config_from, next_source_id, parse_config, parse_firecrawl_web, parse_requery,
    parse_rewrite_query_arg, parse_rewrite_tool_arguments, post_answer_decision,
    registry_contains_url, should_reject_late_requery, truncate_content, FirecrawlWebResult,
    FirstAnswerDecision, PostAnswerDecision, RewriteQueryReject, RewriteToolReject,
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

// -- parse_config: TOML config parsing / error paths / defaults ------------

#[test]
fn parse_config_valid_toml() {
    let config = parse_config(r#"model = "test-model""#).unwrap();
    assert_eq!(config.model, "test-model");
    assert_f64_eq(config.temperature, 0.7); // default
    assert!(config.reasoning); // default true
}

#[test]
fn parse_config_all_fields() {
    let config = parse_config(
        r#"
model = "m"
temperature = 0.5
reasoning = false
"#,
    )
    .unwrap();
    assert_eq!(config.model, "m");
    assert_f64_eq(config.temperature, 0.5);
    assert!(!config.reasoning);
}

#[test]
fn parse_config_invalid_toml_is_error() {
    let result = parse_config("not-toml-at-all");
    assert!(result.is_err(), "invalid TOML must produce an error");
}

#[test]
fn parse_config_empty_string_is_error() {
    let result = parse_config("");
    assert!(result.is_err(), "empty string must produce an error");
}

#[test]
fn parse_config_extra_fields_ignored() {
    let config = parse_config(
        r#"
model = "m"
nonexistent = "ignored"
temperature = 0.1
"#,
    )
    .unwrap();
    assert_eq!(config.model, "m");
    assert_f64_eq(config.temperature, 0.1);
}

#[test]
fn parse_config_missing_model_is_error() {
    let result = parse_config("temperature = 0.5");
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
    // Empty query after the prefix is not a requery (would bill Firecrawl).
    assert_eq!(parse_requery("SEARCH:"), None);
}

#[test]
fn parse_requery_prefix_only_whitespace() {
    assert_eq!(parse_requery("SEARCH:   "), None);
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

// -- should_reject_late_requery: post-insist SEARCH: guard -----------------

#[test]
fn should_reject_late_requery_true_for_valid_search() {
    assert!(should_reject_late_requery("SEARCH: better query"));
}

#[test]
fn should_reject_late_requery_false_for_normal_answer() {
    assert!(!should_reject_late_requery("Paris is the capital [S1]"));
}

#[test]
fn should_reject_late_requery_false_for_blank_search() {
    // Empty/whitespace-only requeries are not actionable SEARCH: lines.
    assert!(!should_reject_late_requery("SEARCH:"));
    assert!(!should_reject_late_requery("SEARCH:   "));
}

#[test]
fn should_reject_late_requery_false_when_embedded() {
    assert!(!should_reject_late_requery("Do SEARCH: again"));
}

// -- parse_rewrite_query_arg: tool-call query extraction -------------------

#[test]
fn parse_rewrite_query_arg_happy_path() {
    let args = serde_json::json!({ "query": "  rust edition  " });
    assert_eq!(parse_rewrite_query_arg(&args).unwrap(), "rust edition");
}

#[test]
fn parse_rewrite_query_arg_missing_field() {
    let args = serde_json::json!({ "other": "x" });
    assert_eq!(
        parse_rewrite_query_arg(&args).unwrap_err(),
        RewriteQueryReject::Missing
    );
}

#[test]
fn parse_rewrite_query_arg_null_is_missing() {
    let args = serde_json::json!({ "query": null });
    assert_eq!(
        parse_rewrite_query_arg(&args).unwrap_err(),
        RewriteQueryReject::Missing
    );
}

#[test]
fn parse_rewrite_query_arg_non_string_is_missing() {
    let args = serde_json::json!({ "query": 42 });
    assert_eq!(
        parse_rewrite_query_arg(&args).unwrap_err(),
        RewriteQueryReject::Missing
    );
}

#[test]
fn parse_rewrite_query_arg_empty_is_empty() {
    let args = serde_json::json!({ "query": "" });
    assert_eq!(
        parse_rewrite_query_arg(&args).unwrap_err(),
        RewriteQueryReject::Empty
    );
}

#[test]
fn parse_rewrite_query_arg_whitespace_is_empty() {
    let args = serde_json::json!({ "query": "   \t  " });
    assert_eq!(
        parse_rewrite_query_arg(&args).unwrap_err(),
        RewriteQueryReject::Empty
    );
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

// -- parse_firecrawl_web: /v2/search data.web shape ------------------------

#[test]
fn parse_firecrawl_web_happy_path() {
    let resp = serde_json::json!({
        "data": {
            "web": [
                {
                    "url": "https://example.com",
                    "title": "Example",
                    "markdown": "# Hello",
                    "description": "snippet"
                }
            ]
        }
    });
    let results = parse_firecrawl_web(&resp).expect("happy path must parse");
    assert_eq!(
        results,
        vec![FirecrawlWebResult {
            url: "https://example.com".into(),
            title: "Example".into(),
            markdown: Some("# Hello".into()),
            description: Some("snippet".into()),
        }]
    );
}

#[test]
fn parse_firecrawl_web_empty_array_is_ok() {
    let resp = serde_json::json!({ "data": { "web": [] } });
    let results = parse_firecrawl_web(&resp).expect("empty web array is valid");
    assert!(results.is_empty());
}

#[test]
fn parse_firecrawl_web_missing_data_web_is_error() {
    let resp = serde_json::json!({ "data": {} });
    let err = parse_firecrawl_web(&resp).expect_err("missing web must error");
    assert!(
        err.to_string().contains("missing or null data.web"),
        "unexpected error: {err}"
    );
}

#[test]
fn parse_firecrawl_web_null_web_is_error() {
    let resp = serde_json::json!({ "data": { "web": null } });
    let err = parse_firecrawl_web(&resp).expect_err("null web must error");
    assert!(
        err.to_string().contains("missing or null data.web"),
        "unexpected error: {err}"
    );
}

#[test]
fn parse_firecrawl_web_bad_entry_shape_is_error() {
    // url is required; a number is the wrong type.
    let resp = serde_json::json!({
        "data": { "web": [ { "url": 123 } ] }
    });
    let err = parse_firecrawl_web(&resp).expect_err("bad entry must error");
    assert!(
        err.to_string().contains("failed to deserialize"),
        "unexpected error: {err}"
    );
}

#[test]
fn parse_firecrawl_web_defaults_optional_fields() {
    let resp = serde_json::json!({
        "data": {
            "web": [ { "url": "https://a.com" } ]
        }
    });
    let results = parse_firecrawl_web(&resp).expect("url-only entry must parse");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].url, "https://a.com");
    assert_eq!(results[0].title, "");
    assert_eq!(results[0].markdown, None);
    assert_eq!(results[0].description, None);
}

// -- ingest_web_results: dedup / truncate / skip-empty / contiguous IDs ----

fn web(
    url: &str,
    title: &str,
    markdown: Option<&str>,
    description: Option<&str>,
) -> FirecrawlWebResult {
    FirecrawlWebResult {
        url: url.into(),
        title: title.into(),
        markdown: markdown.map(str::to_string),
        description: description.map(str::to_string),
    }
}

#[test]
fn ingest_web_results_happy_path_assigns_contiguous_ids() {
    let mut registry = Vec::new();
    let added = ingest_web_results(
        &mut registry,
        vec![
            web("https://a.com", "A", Some("body a"), None),
            web("https://b.com", "B", Some("body b"), None),
        ],
        8_000,
    );
    assert_eq!(added, 2);
    assert_eq!(registry.len(), 2);
    assert_eq!(registry[0].id, "S1");
    assert_eq!(registry[1].id, "S2");
    assert_eq!(registry[0].content, "body a");
    assert_eq!(registry[1].title, "B");
}

#[test]
fn ingest_web_results_dedups_by_url() {
    let mut registry = vec![full_src("S1", "Existing", "https://a.com", "old")];
    let added = ingest_web_results(
        &mut registry,
        vec![
            web("https://a.com", "Dup", Some("new"), None),
            web("https://b.com", "B", Some("body b"), None),
        ],
        8_000,
    );
    assert_eq!(added, 1);
    assert_eq!(registry.len(), 2);
    assert_eq!(registry[0].content, "old");
    assert_eq!(registry[1].id, "S2");
    assert_eq!(registry[1].url, "https://b.com");
}

#[test]
fn ingest_web_results_skips_empty_content() {
    let mut registry = Vec::new();
    let added = ingest_web_results(
        &mut registry,
        vec![
            web("https://empty.com", "E", None, None),
            web("https://ok.com", "O", Some("text"), None),
        ],
        8_000,
    );
    assert_eq!(added, 1);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry[0].id, "S1");
    assert_eq!(registry[0].url, "https://ok.com");
}

#[test]
fn ingest_web_results_prefers_markdown_over_description() {
    let mut registry = Vec::new();
    ingest_web_results(
        &mut registry,
        vec![web(
            "https://a.com",
            "A",
            Some("full markdown"),
            Some("snippet only"),
        )],
        8_000,
    );
    assert_eq!(registry[0].content, "full markdown");
}

#[test]
fn ingest_web_results_falls_back_to_description() {
    let mut registry = Vec::new();
    ingest_web_results(
        &mut registry,
        vec![web("https://a.com", "A", None, Some("snippet"))],
        8_000,
    );
    assert_eq!(registry[0].content, "snippet");
}

#[test]
fn ingest_web_results_truncates_to_max_chars() {
    let mut registry = Vec::new();
    ingest_web_results(
        &mut registry,
        vec![web("https://a.com", "A", Some("abcdefghij"), None)],
        5,
    );
    assert_eq!(registry[0].content, "abcde");
}

// -- first_answer_decision / post_answer_decision: SEARCH + citation gates -

#[test]
fn first_answer_decision_requery() {
    assert_eq!(
        first_answer_decision("SEARCH: better query"),
        FirstAnswerDecision::Requery("better query".into())
    );
}

#[test]
fn first_answer_decision_proceed() {
    assert_eq!(
        first_answer_decision("Paris is the capital [S1]."),
        FirstAnswerDecision::Proceed
    );
    assert_eq!(
        first_answer_decision("SEARCH:"),
        FirstAnswerDecision::Proceed
    );
}

#[test]
fn post_answer_decision_accepts_cited_answer() {
    let registry = vec![src("S1")];
    assert_eq!(
        post_answer_decision("Paris [S1].", &registry),
        PostAnswerDecision::Accept
    );
}

#[test]
fn post_answer_decision_retries_when_no_citations() {
    let registry = vec![src("S1")];
    assert_eq!(
        post_answer_decision("Paris is the capital.", &registry),
        PostAnswerDecision::RetryForCitations
    );
}

#[test]
fn post_answer_decision_late_requery_beats_citation_retry() {
    // SEARCH: answers have no [Sn] citations. Ordering must reject late
    // requery *before* classifying as a zero-citation retry.
    let registry = vec![src("S1")];
    assert_eq!(
        post_answer_decision("SEARCH: another try", &registry),
        PostAnswerDecision::RejectLateRequery
    );
}

#[test]
fn post_answer_decision_accepts_uncited_when_registry_empty() {
    assert_eq!(
        post_answer_decision("no sources available", &[]),
        PostAnswerDecision::Accept
    );
}

// -- parse_rewrite_tool_arguments: no-HTTP rewrite shape failures ----------

#[test]
fn parse_rewrite_tool_arguments_happy_path() {
    let q = parse_rewrite_tool_arguments(Some(r#"{"query":"capital of France"}"#)).unwrap();
    assert_eq!(q, "capital of France");
}

#[test]
fn parse_rewrite_tool_arguments_no_tool_call() {
    let err = parse_rewrite_tool_arguments(None).unwrap_err();
    assert_eq!(err, RewriteToolReject::NoToolCall);
    assert_eq!(err.reason(), "no-tool-call");
}

#[test]
fn parse_rewrite_tool_arguments_invalid_args() {
    let err = parse_rewrite_tool_arguments(Some("not-json")).unwrap_err();
    assert_eq!(err, RewriteToolReject::InvalidArgs);
    assert_eq!(err.reason(), "invalid-tool-args");
}

#[test]
fn parse_rewrite_tool_arguments_missing_query() {
    let err = parse_rewrite_tool_arguments(Some(r#"{"other":"x"}"#)).unwrap_err();
    assert_eq!(err, RewriteToolReject::MissingQuery);
    assert_eq!(err.reason(), "missing-query-field");
}

#[test]
fn parse_rewrite_tool_arguments_empty_query() {
    let err = parse_rewrite_tool_arguments(Some(r#"{"query":"  "}"#)).unwrap_err();
    assert_eq!(err, RewriteToolReject::EmptyQuery);
    assert_eq!(err.reason(), "empty-query");
}

// -- RewriteToolReject::into_final_error: exhaustion messages ---------------

#[test]
fn into_final_error_no_tool_call() {
    let err = RewriteToolReject::NoToolCall.into_final_error(5);
    let msg = format!("{err:#}");
    assert!(msg.contains("after 5 attempts"), "{msg}");
    assert!(msg.contains("did not return a tool call"), "{msg}");
}

#[test]
fn into_final_error_invalid_args() {
    let err = RewriteToolReject::InvalidArgs.into_final_error(3);
    let msg = format!("{err:#}");
    assert!(msg.contains("after 3 attempts"), "{msg}");
    assert!(msg.contains("valid arguments JSON"), "{msg}");
}

#[test]
fn into_final_error_missing_query() {
    let err = RewriteToolReject::MissingQuery.into_final_error(2);
    let msg = format!("{err:#}");
    assert!(msg.contains("after 2 attempts"), "{msg}");
    assert!(msg.contains("missing 'query' field"), "{msg}");
}

#[test]
fn into_final_error_empty_query() {
    let err = RewriteToolReject::EmptyQuery.into_final_error(1);
    let msg = format!("{err:#}");
    assert!(msg.contains("after 1 attempts"), "{msg}");
    assert!(msg.contains("empty 'query' field"), "{msg}");
}

// -- cited_sources: Sources footer filter ----------------------------------

#[test]
fn cited_sources_filters_to_mentioned_ids_in_registry_order() {
    let registry = vec![
        full_src("S1", "One", "https://1.com", "a"),
        full_src("S2", "Two", "https://2.com", "b"),
        full_src("S3", "Three", "https://3.com", "c"),
    ];
    let cited = cited_sources("Uses [S3] and [S1] only.", &registry);
    assert_eq!(cited.len(), 2);
    assert_eq!(cited[0].id, "S1");
    assert_eq!(cited[1].id, "S3");
}

#[test]
fn cited_sources_empty_when_none_cited() {
    let registry = vec![src("S1")];
    assert!(cited_sources("no markers here", &registry).is_empty());
}

// -- load_config_from: filesystem path (no network) ------------------------

#[test]
fn load_config_from_valid_file() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("answerbot-config-ok-{}.toml", std::process::id()));
    std::fs::write(&path, "model = \"test-model\"\ntemperature = 0.3\n").unwrap();
    let config = load_config_from(&path).expect("valid file must load");
    let _ = std::fs::remove_file(&path);
    assert_eq!(config.model, "test-model");
    assert_f64_eq(config.temperature, 0.3);
}

#[test]
fn load_config_from_missing_file_is_error() {
    let path = std::env::temp_dir().join(format!(
        "answerbot-config-missing-{}.toml",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let Err(err) = load_config_from(&path) else {
        panic!("missing file must error");
    };
    assert!(
        err.to_string().contains("failed to read"),
        "unexpected error: {err}"
    );
}
