//! Plain, dependency-free terminal rendering.
//!
//! This module deliberately only formats frozen contracts.  It does not own
//! terminal state and, in particular, never turns rendered source material
//! back into model input.

use crate::agent::{MemoryEvent, MemoryEventKind};
use crate::config::AppConfig;
use crate::contracts::TurnProvenance;
use crate::search::registry::BackendRegistry;
use crate::tools::meta_search::SearchActivity;
use std::collections::HashSet;
use std::io::{self, Write};
use std::path::Path;

/// Make untrusted text safe to place in a terminal-rendered line.
///
/// CR/LF are deliberately represented as spaces: allowing either one through
/// would let source data manufacture additional renderer output lines.  ANSI
/// sequences are consumed before the remaining control characters are
/// filtered; tabs remain useful for callers that intentionally use them for
/// indentation.
fn sanitize_terminal_line(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut pending_line_break = false;

    while let Some(character) = chars.next() {
        if character == '\u{1b}' {
            match chars.next() {
                Some(']') => {
                    // OSC sequences end at BEL or the ST sequence (ESC \\).
                    while let Some(next) = chars.next() {
                        if next == '\u{7}' {
                            break;
                        }
                        if next == '\u{1b}' && chars.next_if_eq(&'\\').is_some() {
                            break;
                        }
                    }
                }
                Some('[') => {
                    // CSI sequences end with a byte in the final range.
                    for next in chars.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
                Some(_) | None => {
                    // Consume the command character of a two-byte ESC form.
                }
            }
            continue;
        }

        if character == '\r' || character == '\n' {
            pending_line_break = true;
            continue;
        }
        if pending_line_break {
            if !output.ends_with(' ') {
                output.push(' ');
            }
            pending_line_break = false;
        }
        if character == '\t' || !character.is_control() {
            output.push(character);
        }
    }
    if pending_line_break && !output.ends_with(' ') {
        output.push(' ');
    }
    output
}

/// Sanitize free-form untrusted text for direct terminal output.
///
/// This preserves ordinary Unicode, tabs, and LF formatting. CR is dropped
/// so text cannot overwrite an existing terminal line.
pub(crate) fn sanitize_terminal_text(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(character) = chars.next() {
        if character == '\u{1b}' {
            match chars.next() {
                Some(']') => {
                    while let Some(next) = chars.next() {
                        if next == '\u{7}' {
                            break;
                        }
                        if next == '\u{1b}' && chars.next_if_eq(&'\\').is_some() {
                            break;
                        }
                    }
                }
                Some('[') => {
                    for next in chars.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
                Some(_) | None => {}
            }
            continue;
        }
        if character == '\r' {
            continue;
        }
        if character == '\n' || character == '\t' || !character.is_control() {
            output.push(character);
        }
    }
    output
}

/// Messages emitted by memory management without coupling this lane to the
/// production memory implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionNotice {
    Started {
        before_tokens: usize,
    },
    Completed {
        before_tokens: usize,
        after_tokens: usize,
    },
    Failed {
        error: String,
    },
}

pub fn render_activity(activity: &SearchActivity) -> String {
    let mut out = String::new();
    match activity {
        SearchActivity::QueryStarted { query } => {
            out.push_str(&format!(
                "Search started: {}\n",
                sanitize_terminal_line(query)
            ));
        }
        SearchActivity::ProviderStarted { provider } => {
            out.push_str(&format!(
                "Provider {}: searching\n",
                sanitize_terminal_line(provider)
            ));
        }
        SearchActivity::ProviderResult {
            provider,
            elapsed_ms,
            hits,
            retry_count,
            normalized_hits,
        } => {
            out.push_str(&format!(
                "Provider {}: {hits} hit(s) in {elapsed_ms} ms (retries: {})\n",
                sanitize_terminal_line(provider),
                retry_count.map_or_else(|| "unknown".to_owned(), |n| n.to_string())
            ));
            for (index, hit) in normalized_hits.iter().enumerate() {
                out.push_str(&format!("  Hit {}\n", index + 1));
                out.push_str(&format!(
                    "    Title: {}\n",
                    sanitize_terminal_line(&hit.title)
                ));
                out.push_str(&format!(
                    "    Snippet: {}\n",
                    sanitize_terminal_line(&hit.snippet)
                ));
                out.push_str(&format!("    URL: {}\n", sanitize_terminal_line(&hit.url)));
                if !hit.metadata.is_empty() {
                    out.push_str("    Metadata:\n");
                    for (key, value) in &hit.metadata {
                        out.push_str(&format!(
                            "      {}: {}\n",
                            sanitize_terminal_line(key),
                            sanitize_terminal_line(value)
                        ));
                    }
                }
            }
        }
        SearchActivity::ProviderError {
            provider,
            elapsed_ms,
            error,
            retry_count,
        } => {
            out.push_str(&format!(
                "Provider {}: ERROR after {elapsed_ms} ms: {} (retries: {})\n",
                sanitize_terminal_line(provider),
                sanitize_terminal_line(error),
                retry_count.map_or_else(|| "unknown".to_owned(), |n| n.to_string())
            ));
        }
        SearchActivity::RankingStarted { candidates } => {
            out.push_str(&format!("Ranking started: {candidates} candidate(s)\n"));
        }
        SearchActivity::RankingCompleted {
            elapsed_ms,
            selected,
            decisions,
        } => {
            out.push_str(&format!(
                "Ranking completed: selected {selected} in {elapsed_ms} ms\n"
            ));
            if decisions.is_empty() {
                out.push_str("  Decisions: none\n");
            } else {
                out.push_str("  Decisions:\n");
                for decision in decisions {
                    let score = decision
                        .score
                        .map_or_else(|| "unknown".to_owned(), |value| value.to_string());
                    out.push_str(&format!(
                        "    - {} | {} | selected: {} | score: {} | decision: {}\n",
                        sanitize_terminal_line(&decision.source_id),
                        sanitize_terminal_line(&decision.normalized_url),
                        decision.selected,
                        score,
                        sanitize_terminal_line(&decision.decision)
                    ));
                }
            }
        }
        SearchActivity::RankingFailed { elapsed_ms, error } => {
            out.push_str(&format!(
                "Ranking FAILED after {elapsed_ms} ms: {}\n",
                sanitize_terminal_line(error)
            ));
        }
    }
    out
}

pub fn render_activities(activities: &[SearchActivity]) -> String {
    activities.iter().map(render_activity).collect()
}

pub fn write_activity<W: Write>(writer: &mut W, activity: &SearchActivity) -> io::Result<()> {
    writer.write_all(render_activity(activity).as_bytes())?;
    writer.flush()
}

/// Render only the selected evidence for a turn.  `normalized_url` is the
/// sole identity key, so duplicate selections cannot produce duplicate lines.
pub fn render_sources(provenance: &TurnProvenance) -> String {
    let mut out = String::new();
    let mut seen = HashSet::new();
    let mut number = 1;
    for entry in &provenance.entries {
        if !seen.insert(entry.normalized_url.as_str()) {
            continue;
        }
        out.push_str(&format!(
            "{number}. {} — {}\n",
            sanitize_terminal_line(&entry.title),
            sanitize_terminal_line(&entry.url)
        ));
        number += 1;
    }
    out
}

pub fn write_sources<W: Write>(writer: &mut W, provenance: &TurnProvenance) -> io::Result<()> {
    writer.write_all(render_sources(provenance).as_bytes())?;
    writer.flush()
}

pub fn render_help() -> String {
    "Commands:\n  /help     Show this help\n  /status   Show configuration and runtime status\n  /quit     Exit\n  /clear    Clear the conversation\n  /compact  Compact conversation memory\n".into()
}

pub fn write_help<W: Write>(writer: &mut W) -> io::Result<()> {
    writer.write_all(render_help().as_bytes())?;
    writer.flush()
}

/// Format public status data only.  `AppConfig` contains no resolved
/// credentials; provider key fields are intentionally not printed either.
pub fn render_status(config: &AppConfig, registry: &BackendRegistry, config_path: &Path) -> String {
    let enabled = registry
        .enabled_names()
        .iter()
        .map(|name| sanitize_terminal_line(name))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Status\nModels:\n  chat: {}\n  rank: {}\n  summarize: {}\nProviders: {}\nSearch:\n  stage budget: {}s\n  rank timeout: {}s\n  hit cap: {}\n  model output: {} bytes\nFetch:\n  timeout: {}s\n  max bytes: {}\n  max chars: {}\n  redirect limit: {}\nSession:\n  log directory: {}\n  redact credentials: {}\n  redact auth headers: {}\nConfig path: {}\n",
        sanitize_terminal_line(&config.models.chat_id),
        sanitize_terminal_line(&config.models.rank_id),
        sanitize_terminal_line(&config.models.summarize_id),
        if enabled.is_empty() { "none" } else { &enabled },
        config.search.stage_budget_secs,
        config.search.rank_timeout_secs,
        config.search.per_backend_hit_cap,
        config.search.model_output_bytes,
        config.fetch.timeout_secs,
        config.fetch.max_bytes,
        config.fetch.max_chars,
        config.fetch.redirect_limit,
        sanitize_terminal_line(&config.session.log_directory.display().to_string()),
        config.session.redact_credentials,
        config.session.redact_auth_headers,
        sanitize_terminal_line(&config_path.display().to_string()),
    )
}

pub fn write_status<W: Write>(
    writer: &mut W,
    config: &AppConfig,
    registry: &BackendRegistry,
    config_path: &Path,
) -> io::Result<()> {
    writer.write_all(render_status(config, registry, config_path).as_bytes())?;
    writer.flush()
}

pub fn render_compaction(notice: &CompactionNotice) -> String {
    match notice {
        CompactionNotice::Started { before_tokens } => {
            format!("Compaction started ({before_tokens} tokens)\n")
        }
        CompactionNotice::Completed {
            before_tokens,
            after_tokens,
        } => format!("Compaction completed: {before_tokens} -> {after_tokens} tokens\n"),
        CompactionNotice::Failed { error } => {
            format!("Compaction failed: {}\n", sanitize_terminal_line(error))
        }
    }
}

pub fn write_compaction<W: Write>(writer: &mut W, notice: &CompactionNotice) -> io::Result<()> {
    writer.write_all(render_compaction(notice).as_bytes())?;
    writer.flush()
}

/// Format the production memory activity contract without exposing memory
/// implementation details to callers of this rendering lane.
pub fn render_memory_event(event: &MemoryEvent) -> String {
    let label = match &event.kind {
        MemoryEventKind::Started => "Compaction started",
        MemoryEventKind::Retry => "Compaction retry (summarizer)",
        MemoryEventKind::Completed => "Compaction completed",
        MemoryEventKind::Fallback => "Compaction fallback (summarizer unavailable)",
        MemoryEventKind::Cleared => "Conversation memory cleared",
    };
    format!(
        "{label} [{}]: {}\n",
        sanitize_terminal_line(&event.conversation_id),
        sanitize_terminal_line(&event.detail)
    )
}

pub fn write_memory_event<W: Write>(writer: &mut W, event: &MemoryEvent) -> io::Result<()> {
    writer.write_all(render_memory_event(event).as_bytes())?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::session::RankingDecision;
    use crate::contracts::{BackendKind, EvidenceEntry, SearchHit};
    use std::collections::BTreeMap;

    fn hit() -> SearchHit {
        SearchHit {
            title: "A title".into(),
            url: "https://example.test/a".into(),
            snippet: "A snippet".into(),
            published: None,
            native_rank: None,
            native_score: None,
            provider: "demo".into(),
            backend_kind: BackendKind::Web,
            source_subtype: None,
            metadata: BTreeMap::from([("kind".into(), "doc".into())]),
        }
    }

    #[test]
    fn provider_activity_has_detail_and_unknown_retry() {
        let text = render_activity(&SearchActivity::ProviderResult {
            provider: "demo".into(),
            elapsed_ms: 42,
            hits: 1,
            retry_count: None,
            normalized_hits: vec![hit()],
        });
        assert!(text.contains("demo") && text.contains("42 ms") && text.contains("1 hit"));
        assert!(
            text.contains("A title") && text.contains("A snippet") && text.contains("kind: doc")
        );
        assert!(text.contains("retries: unknown") && !text.contains("retries: 0"));
    }

    #[test]
    fn ranking_activity_has_structured_decisions() {
        let text = render_activity(&SearchActivity::RankingCompleted {
            elapsed_ms: 17,
            selected: 1,
            decisions: vec![RankingDecision {
                source_id: "source-1".into(),
                normalized_url: "https://example.test".into(),
                selected: true,
                decision: "strong match".into(),
                score: Some(0.875),
            }],
        });
        assert!(text.contains("17 ms") && text.contains("selected 1"));
        assert!(text.contains("source-1") && text.contains("https://example.test"));
        assert!(text.contains("selected: true") && text.contains("score: 0.875"));
        assert!(text.contains("decision: strong match"));
    }

    #[test]
    fn untrusted_activity_fields_cannot_forge_lines_or_terminal_controls() {
        let hostile = "ok\r\nFORGED\u{1b}[31m RED\u{1b}[0m\u{7}\u{8}\treadable";
        let text = render_activity(&SearchActivity::ProviderResult {
            provider: hostile.into(),
            elapsed_ms: 1,
            hits: 1,
            retry_count: Some(0),
            normalized_hits: vec![SearchHit {
                title: hostile.into(),
                url: hostile.into(),
                snippet: hostile.into(),
                metadata: BTreeMap::from([(hostile.into(), hostile.into())]),
                ..hit()
            }],
        });

        assert!(!text.contains("\u{1b}"));
        assert!(!text.contains('\u{7}'));
        assert!(!text.contains('\u{8}'));
        assert!(text.contains("ok FORGED RED\treadable"));
        assert!(!text.contains("\nFORGED"));
        assert!(text.contains('\t'));
    }

    #[test]
    fn free_form_terminal_sanitizer_preserves_formatting_and_unicode() {
        let hostile =
            "α\tline one\nline two\rOVER\u{1b}]52;c;secret\u{7} tail \u{1b}[31mred\u{1b}[0m\u{8}";
        assert_eq!(
            sanitize_terminal_text(hostile),
            "α\tline one\nline twoOVER tail red"
        );
    }

    #[test]
    fn split_escape_prefix_cannot_activate_a_sequence() {
        assert_eq!(sanitize_terminal_text("\u{1b}"), "");
        assert_eq!(sanitize_terminal_text("[31mvisible"), "[31mvisible");
    }

    #[test]
    fn untrusted_source_fields_cannot_forge_source_lines() {
        let hostile = "title\n2. forged — https://evil.test\u{1b}[2J";
        let entry = EvidenceEntry {
            source_id: "source".into(),
            normalized_url: "unique".into(),
            title: hostile.into(),
            url: hostile.into(),
            supporting_snippet: hostile.into(),
            rank_decision: Some(hostile.into()),
            provider_labels: Default::default(),
            source_subtypes: Default::default(),
        };
        let text = render_sources(&TurnProvenance {
            turn_id: "turn".into(),
            entries: vec![entry],
        });

        assert_eq!(text.lines().count(), 1);
        assert!(!text.contains('\u{1b}'));
        assert!(text.contains("title 2. forged — https://evil.test"));
    }

    #[test]
    fn sources_are_numbered_and_deduplicated() {
        let entry = |url: &str, title: &str| EvidenceEntry {
            source_id: title.into(),
            normalized_url: url.into(),
            title: title.into(),
            url: url.into(),
            supporting_snippet: String::new(),
            rank_decision: None,
            provider_labels: Default::default(),
            source_subtypes: Default::default(),
        };
        let text = render_sources(&TurnProvenance {
            turn_id: "t".into(),
            entries: vec![
                entry("u", "one"),
                entry("u", "duplicate"),
                entry("v", "two"),
            ],
        });
        assert_eq!(text, "1. one — u\n2. two — v\n");
    }

    #[test]
    fn help_and_status_do_not_expose_secrets() {
        let help = render_help();
        for command in ["/help", "/status", "/quit", "/clear", "/compact"] {
            assert!(help.contains(command), "missing {command}");
        }
        assert!(!help.contains("API_KEY"));

        let config = AppConfig {
            models: crate::config::ModelsConfig {
                chat_id: "chat-model".into(),
                rank_id: "rank-model".into(),
                summarize_id: "summary-model".into(),
                chat_context_tokens: 1,
            },
            search: crate::config::SearchConfig {
                stage_budget_secs: 20,
                rank_timeout_secs: 5,
                per_backend_hit_cap: 3,
                model_output_bytes: 100,
            },
            providers: Default::default(),
            session: crate::config::SessionConfig {
                log_directory: "logs".into(),
                redact_credentials: true,
                redact_auth_headers: true,
            },
            fetch: crate::config::FetchConfig {
                timeout_secs: 4,
                max_bytes: 10,
                max_chars: 11,
                allowed_schemes: vec![],
                allowed_media_types: vec![],
                redirect_limit: 2,
            },
        };
        let status = render_status(&config, &BackendRegistry::new(), Path::new("config.toml"));
        assert!(status.contains("chat-model") && status.contains("stage budget: 20s"));
        assert!(!status.contains("OPENROUTER_API_KEY") && !status.contains("secret"));
    }

    #[test]
    fn every_memory_event_kind_is_clear() {
        let cases = [
            (MemoryEventKind::Started, "Compaction started"),
            (MemoryEventKind::Retry, "Compaction retry"),
            (MemoryEventKind::Completed, "Compaction completed"),
            (MemoryEventKind::Fallback, "Compaction fallback"),
            (MemoryEventKind::Cleared, "Conversation memory cleared"),
        ];
        for (kind, label) in cases {
            let text = render_memory_event(&MemoryEvent {
                kind,
                conversation_id: "conversation-1".into(),
                detail: "test detail".into(),
            });
            assert!(text.contains(label));
            assert!(text.contains("conversation-1") && text.contains("test detail"));
        }
    }
}
