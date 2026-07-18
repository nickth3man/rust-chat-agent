//! Provider-coverage assessment harness for `openrouter-chat-rust`.
//!
//! Exercises every configured search backend with a representative query,
//! drives the meta-search → dedup → rank pipeline multiple times to compute
//! cross-query aggregates, and runs a small chat-stream completion check.
//! Produces a terminal summary report and a JSON journal artifact under
//! `sessions/` for traceability.
//!
//! This binary is strictly additive: it does not modify production code
//! behavior.  It reuses the public API of the library crate.
//!
//! Usage:
//!
//! ```text
//! cargo run --bin coverage_assessment
//! ```
//!
//! Optional environment variables:
//!
//! - `COVERAGE_QUERY` — overrides the default representative query
//!   (`"rust programming language"`).
//! - `COVERAGE_RUNS` — overrides the default number of meta-search runs
//!   (default `3`, min `1`, max `10`).
//! - `COVERAGE_OUTPUT` — overrides the sessions/ output file path
//!   (default: derived from the session id).
//!
//! Blocking conditions are detected up front and reported with the input that
//! would resolve them.

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use openrouter_chat_rust::{
    config::{self, SearchConfig},
    contracts::{
        SearchBackend, SearchHit, ToolNetError,
        session::{RankingDecision, SessionMetadata},
    },
    rank::{Ranker, RigRanker},
    search::BackendRegistry,
    tools::meta_search::{
        MetaSearch, MetaSearchArgs, MetaSearchOutput, MetaSearchState, SearchActivity,
    },
};
use rig_core::{
    agent::MultiTurnStreamItem,
    client::CompletionClient,
    providers::openrouter,
    streaming::{StreamedAssistantContent, StreamingPrompt},
    tool::Tool,
};
use serde::{Deserialize, Serialize};

const KNOWN_PROVIDERS: &[&str] = &[
    "duckduckgo",
    "hn",
    "stract",
    "wiby",
    "mdn",
    "reddit",
    "lobsters",
    "searchmysite",
    "marginalia",
    "mwmbl",
    "wikipedia",
    "wikidata",
    "openlibrary",
    "free_dictionary",
    "arxiv",
    "crossref",
    "semantic_scholar",
    "pubmed",
    "github",
    "stackexchange",
    "npm",
    "crates_io",
    "gdelt",
    "firecrawl",
    "brave",
    "mojeek",
    "searxng",
];

const DEFAULT_QUERY: &str = "rust programming language";
const DEFAULT_META_SEARCH_RUNS: usize = 3;
const CHAT_STREAM_RUNS: usize = 3;
const CHAT_STREAM_TIMEOUT: Duration = Duration::from_secs(45);
const PER_BACKEND_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerBackendReport {
    name: String,
    status: String,
    hits: usize,
    elapsed_ms: u64,
    error: Option<String>,
    sample_titles: Vec<String>,
    sample_urls: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetaSearchRunReport {
    run_index: usize,
    query: String,
    raw_hits: usize,
    unique_urls: usize,
    selected: usize,
    ranking_elapsed_ms: u64,
    ranking_calls_per_second: f64,
    total_elapsed_ms: u64,
    warnings: Vec<String>,
    top_titles: Vec<String>,
    top_urls: Vec<String>,
    top_scores: Vec<Option<f64>>,
    top_decisions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatStreamReport {
    model: String,
    runs: usize,
    completed: usize,
    completion_rate: f64,
    first_response_ms: Option<u64>,
    failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualitativeObservation {
    category: String,
    detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoverageJournal {
    metadata: SessionMetadata,
    config_snapshot: ConfigSnapshot,
    representative_query: String,
    started_at: String,
    finished_at: String,
    per_backend: Vec<PerBackendReport>,
    meta_search_runs: Vec<MetaSearchRunReport>,
    meta_search_aggregate: MetaSearchAggregate,
    chat_stream: ChatStreamReport,
    qualitative: Vec<QualitativeObservation>,
    blocking: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConfigSnapshot {
    chat_id: String,
    rank_id: String,
    rank_timeout_secs: u64,
    summarize_id: String,
    enabled: Vec<String>,
    disabled: Vec<String>,
    per_backend_hit_cap: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetaSearchAggregate {
    runs: usize,
    total_raw_hits: usize,
    total_unique_urls: usize,
    total_selected: usize,
    mean_ranking_latency_ms: f64,
    ranking_calls_per_second: f64,
    mean_raw_to_unique_dedup_ratio: f64,
    total_deduped: usize,
}

fn timestamp_string() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => format!("{}.{:09}Z", d.as_secs(), d.subsec_nanos()),
        Err(_) => "0Z".into(),
    }
}

fn session_id_string() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    format!("coverage-{pid}-{nanos}")
}

fn safe_error_string(error: &ToolNetError) -> String {
    match error {
        ToolNetError::Timeout => "timeout".into(),
        ToolNetError::Network(message) => format!("network: {message}"),
        ToolNetError::HttpStatus { status, .. } => format!("http {status}"),
        ToolNetError::Parse(message) => format!("parse: {message}"),
        ToolNetError::BodyTooLarge { limit } => format!("body too large (limit {limit})"),
        ToolNetError::Content(message) => format!("content: {message}"),
    }
}

fn print_line(line: &str) {
    let _ = writeln!(io::stdout(), "{line}");
    let _ = io::stdout().flush();
}

fn section(title: &str) {
    print_line("");
    print_line(&format!("=== {title} ==="));
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(default)
}

async fn exercise_backend(backend: &Arc<dyn SearchBackend>, query: &str) -> PerBackendReport {
    let name = backend.name().to_owned();
    let started = Instant::now();
    let outcome: Result<Result<Vec<SearchHit>, ToolNetError>, tokio::time::error::Elapsed> =
        tokio::time::timeout(PER_BACKEND_TIMEOUT, backend.search(query)).await;
    let elapsed_ms = started.elapsed().as_millis() as u64;
    match outcome {
        Ok(Ok(hits)) => {
            let sample_titles = hits
                .iter()
                .take(3)
                .map(|hit| hit.title.chars().take(120).collect::<String>())
                .collect::<Vec<_>>();
            let sample_urls = hits
                .iter()
                .take(3)
                .map(|hit| hit.url.chars().take(160).collect::<String>())
                .collect::<Vec<_>>();
            PerBackendReport {
                name,
                status: "succeeded".into(),
                hits: hits.len(),
                elapsed_ms,
                error: None,
                sample_titles,
                sample_urls,
            }
        }
        Ok(Err(error)) => PerBackendReport {
            name,
            status: "failed".into(),
            hits: 0,
            elapsed_ms,
            error: Some(safe_error_string(&error)),
            sample_titles: Vec::new(),
            sample_urls: Vec::new(),
        },
        Err(_) => PerBackendReport {
            name,
            status: "failed".into(),
            hits: 0,
            elapsed_ms,
            error: Some("per-backend timeout".into()),
            sample_titles: Vec::new(),
            sample_urls: Vec::new(),
        },
    }
}

fn classify_meta_search_run(
    activities: &[SearchActivity],
    output: &MetaSearchOutput,
    warnings: &[String],
    query: &str,
    run_index: usize,
    total_elapsed: Duration,
) -> MetaSearchRunReport {
    let mut raw_hits: usize = 0;
    let mut unique_urls: usize = 0;
    let mut selected: usize = 0;
    let mut ranking_elapsed_ms: u64 = 0;

    for activity in activities {
        match activity {
            SearchActivity::ProviderResult { hits, .. } => {
                raw_hits = raw_hits.saturating_add(*hits)
            }
            SearchActivity::RankingStarted { candidates } => unique_urls = *candidates,
            SearchActivity::RankingCompleted {
                elapsed_ms,
                selected: selected_count,
                ..
            } => {
                ranking_elapsed_ms = *elapsed_ms;
                selected = *selected_count;
            }
            _ => {}
        }
    }

    let ranking_calls_per_second = if ranking_elapsed_ms > 0 {
        1000.0 / ranking_elapsed_ms as f64
    } else {
        0.0
    };

    // Pull top decisions from the last RankingCompleted event.
    let last_completed = activities.iter().rev().find_map(|a| match a {
        SearchActivity::RankingCompleted { decisions, .. } => Some(decisions),
        _ => None,
    });

    let top_decisions = match last_completed {
        Some(decisions) => {
            let selected_only: Vec<&RankingDecision> =
                decisions.iter().filter(|d| d.selected).collect();
            selected_only
                .iter()
                .take(5)
                .map(|d| d.decision.clone())
                .collect()
        }
        None => Vec::new(),
    };

    // The tool may retain authoritative sources after model ranking; journal
    // the actual returned selection rather than the pre-retention activity.
    let top_titles = output
        .selected
        .iter()
        .take(5)
        .map(|hit| hit.title.clone())
        .collect();
    let top_urls = output
        .selected
        .iter()
        .take(5)
        .map(|hit| hit.url.clone())
        .collect();
    let top_scores = output
        .selected
        .iter()
        .take(5)
        .map(|hit| hit.score)
        .collect();

    MetaSearchRunReport {
        run_index,
        query: query.into(),
        raw_hits,
        unique_urls,
        selected,
        ranking_elapsed_ms,
        ranking_calls_per_second,
        total_elapsed_ms: total_elapsed.as_millis() as u64,
        warnings: warnings.to_vec(),
        top_titles,
        top_urls,
        top_scores,
        top_decisions,
    }
}

async fn exercise_meta_search(
    registry: &BackendRegistry,
    ranker: Arc<dyn Ranker>,
    search_config: &SearchConfig,
    query: &str,
    runs: usize,
) -> (Vec<MetaSearchRunReport>, MetaSearchAggregate) {
    let mut reports = Vec::with_capacity(runs);
    let mut total_raw_hits: usize = 0;
    let mut total_unique_urls: usize = 0;
    let mut total_selected: usize = 0;
    let mut total_ranking_latency_ms: u64 = 0;
    let mut total_deduped: usize = 0;

    for run_index in 1..=runs {
        let state = MetaSearchState::new();
        let search = MetaSearch::with_state(
            registry.clone(),
            ranker.clone(),
            search_config.clone(),
            state.clone(),
        );

        let started = Instant::now();
        let outcome = search
            .call(MetaSearchArgs {
                query: query.into(),
            })
            .await;
        let total_elapsed = started.elapsed();

        let activities = state.activities().await;
        match outcome {
            Ok(output) => {
                let report = classify_meta_search_run(
                    &activities,
                    &output,
                    &output.warnings,
                    query,
                    run_index,
                    total_elapsed,
                );
                // Only successful ranked runs contribute to aggregate search
                // quality, just as ranking latency does below.
                total_raw_hits = total_raw_hits.saturating_add(report.raw_hits);
                total_unique_urls = total_unique_urls.saturating_add(report.unique_urls);
                total_selected = total_selected.saturating_add(report.selected);
                total_ranking_latency_ms =
                    total_ranking_latency_ms.saturating_add(report.ranking_elapsed_ms);
                total_deduped = total_deduped
                    .saturating_add(report.raw_hits.saturating_sub(report.unique_urls));
                reports.push(report);
            }
            Err(error) => {
                let warning = format!("meta_search run {run_index} failed: {error}");
                print_line(&warning);
                reports.push(MetaSearchRunReport {
                    run_index,
                    query: query.into(),
                    raw_hits: 0,
                    unique_urls: 0,
                    selected: 0,
                    ranking_elapsed_ms: 0,
                    ranking_calls_per_second: 0.0,
                    total_elapsed_ms: total_elapsed.as_millis() as u64,
                    warnings: vec![warning],
                    top_titles: Vec::new(),
                    top_urls: Vec::new(),
                    top_scores: Vec::new(),
                    top_decisions: Vec::new(),
                });
            }
        }
    }

    let successful = reports
        .iter()
        .filter(|r| r.ranking_elapsed_ms > 0)
        .count()
        .max(1);
    let mean_ranking_latency_ms = total_ranking_latency_ms as f64 / successful as f64;
    let ranking_calls_per_second = if mean_ranking_latency_ms > 0.0 {
        1000.0 / mean_ranking_latency_ms
    } else {
        0.0
    };
    let mean_raw_to_unique_dedup_ratio = if total_raw_hits > 0 {
        total_unique_urls as f64 / total_raw_hits as f64
    } else {
        0.0
    };

    let aggregate = MetaSearchAggregate {
        runs,
        total_raw_hits,
        total_unique_urls,
        total_selected,
        mean_ranking_latency_ms,
        ranking_calls_per_second,
        mean_raw_to_unique_dedup_ratio,
        total_deduped,
    };

    (reports, aggregate)
}

async fn exercise_chat_stream(
    client: &openrouter::Client<reqwest::Client>,
    chat_model: &str,
) -> ChatStreamReport {
    let mut completed = 0usize;
    let mut failures = Vec::new();
    let mut first_response_ms: Option<u64> = None;

    for attempt in 1..=CHAT_STREAM_RUNS {
        let agent = client
            .agent(chat_model)
            .preamble(
                "You are a small language model returning a single short sentence. \
                 Do not call any tools.",
            )
            .build();

        let started = Instant::now();
        let stream_build = agent
            .stream_prompt("Reply with one sentence starting with ORBITAL-COBALT-1.")
            .max_turns(1)
            .into_future();
        let stream_result = tokio::time::timeout(CHAT_STREAM_TIMEOUT, stream_build).await;

        match stream_result {
            Ok(mut stream) => {
                let mut stream_started = false;
                let mut emitted_final = false;
                let mut text_chars: usize = 0;
                let mut stream_error: Option<String> = None;
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                            emitted_final = true;
                        }
                        Ok(MultiTurnStreamItem::StreamAssistantItem(
                            StreamedAssistantContent::Text(text),
                        )) => {
                            text_chars = text_chars.saturating_add(text.text.len());
                            if !stream_started {
                                stream_started = true;
                                if first_response_ms.is_none() {
                                    first_response_ms = Some(started.elapsed().as_millis() as u64);
                                }
                            }
                        }
                        Ok(_) => {}
                        Err(error) => {
                            stream_error = Some(error.to_string());
                            break;
                        }
                    }
                }
                let elapsed_ms = started.elapsed().as_millis() as u64;
                if let Some(error) = stream_error {
                    failures.push(format!("attempt {attempt}: stream error: {error}"));
                } else if emitted_final && text_chars > 0 {
                    completed += 1;
                } else {
                    failures.push(format!(
                        "attempt {attempt}: stream ended without FinalResponse or text (text_chars={text_chars}, elapsed_ms={elapsed_ms})"
                    ));
                }
            }
            Err(_) => {
                let elapsed_ms = started.elapsed().as_millis() as u64;
                failures.push(format!(
                    "attempt {attempt}: chat stream timed out after {elapsed_ms}ms"
                ));
            }
        }
    }

    ChatStreamReport {
        model: chat_model.into(),
        runs: CHAT_STREAM_RUNS,
        completed,
        completion_rate: completed as f64 / CHAT_STREAM_RUNS as f64,
        first_response_ms,
        failures,
    }
}

fn observations_from_meta_search(
    reports: &[MetaSearchRunReport],
    expected_sources: &[&str],
) -> Vec<QualitativeObservation> {
    let mut observations = Vec::new();
    if reports.is_empty() {
        observations.push(QualitativeObservation {
            category: "ranking".into(),
            detail: "No meta_search runs completed; cannot evaluate ranking quality.".into(),
        });
        return observations;
    }

    // 0. If every run failed before producing selections, surface that prominently.
    let successful_runs = reports
        .iter()
        .filter(|r| r.selected > 0 && r.ranking_elapsed_ms > 0)
        .count();
    if successful_runs == 0 {
        let common_warning = reports
            .iter()
            .flat_map(|r| r.warnings.iter())
            .find(|w| !w.is_empty())
            .cloned()
            .unwrap_or_else(|| "no warnings recorded".to_string());
        observations.push(QualitativeObservation {
            category: "blocking".into(),
            detail: format!(
                "ranker produced zero selections across {} run(s); first warning: {common_warning}",
                reports.len()
            ),
        });
        observations.push(QualitativeObservation {
            category: "coverage".into(),
            detail: "expected-source check is inconclusive: ranking did not finish".into(),
        });
        return observations;
    }

    let last = reports.last().expect("non-empty by guard");
    let aggregate_decisions: Vec<&str> = last.top_decisions.iter().map(String::as_str).collect();

    // 1. Look for irrelevant / off-topic top results.
    let mut irrelevant_count = 0usize;
    for (idx, decision) in aggregate_decisions.iter().enumerate() {
        let lower = decision.to_ascii_lowercase();
        let maybe_off_topic = lower.contains("not relevant")
            || lower.contains("irrelevant")
            || lower.contains("off-topic")
            || lower.contains("not directly relevant");
        if maybe_off_topic {
            irrelevant_count += 1;
            observations.push(QualitativeObservation {
                category: "ranking".into(),
                detail: format!(
                    "selected rank #{idx} carries a low-relevance decision text: \"{decision}\""
                ),
            });
        }
    }

    // 2. Look for missing expected sources.
    let joined_urls = last.top_urls.join(" ");
    for expected in expected_sources {
        if !joined_urls
            .to_ascii_lowercase()
            .contains(&expected.to_ascii_lowercase())
        {
            observations.push(QualitativeObservation {
                category: "coverage".into(),
                detail: format!(
                    "expected authoritative source \"{expected}\" was absent from selected top results"
                ),
            });
        }
    }

    // 3. Spurious decisions: same source_id ranked multiple times or empty decisions.
    for decision in &last.top_decisions {
        if decision.trim().is_empty() {
            observations.push(QualitativeObservation {
                category: "ranking".into(),
                detail: "ranking model returned an empty decision string for a selected hit".into(),
            });
        }
    }

    // 4. Cross-run stability: did the same number of candidates surface each run?
    let first = &reports[0];
    if reports.len() > 1 {
        let stable_unique = reports.iter().all(|r| r.unique_urls == first.unique_urls);
        if !stable_unique {
            observations.push(QualitativeObservation {
                category: "stability".into(),
                detail: format!(
                    "unique URL count varied across runs ({} -> [{}])",
                    first.unique_urls,
                    reports
                        .iter()
                        .map(|r| r.unique_urls.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            });
        }
    }

    // 5. Score sanity: do selected hits all carry a score?
    if last.top_scores.iter().any(|score| score.is_none()) {
        observations.push(QualitativeObservation {
            category: "ranking".into(),
            detail: "one or more selected hits arrived without a model-assigned score".into(),
        });
    }

    if irrelevant_count == 0
        && reports
            .iter()
            .all(|r| r.top_decisions.iter().all(|d| !d.trim().is_empty()))
    {
        observations.push(QualitativeObservation {
            category: "ranking".into(),
            detail:
                "all selected hits carried positive decisions and scored URLs in the top results"
                    .into(),
        });
    }

    observations
}

async fn write_journal(journal: &CoverageJournal, path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let serialized = serde_json::to_string_pretty(journal)
        .map_err(|error| io::Error::other(format!("serialize: {error}")))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serialized.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn derive_session_path(root: &Path, session_id: &str) -> PathBuf {
    let log_dir = root.join("sessions");
    log_dir.join(format!("{session_id}.json"))
}

fn render_terminal_summary(
    reports: &Vec<PerBackendReport>,
    meta_runs: &[MetaSearchRunReport],
    meta_aggregate: &MetaSearchAggregate,
    chat: &ChatStreamReport,
    observations: &[QualitativeObservation],
    query: &str,
) {
    section("Provider coverage assessment summary");
    print_line(&format!("Representative query: \"{query}\""));
    print_line(&format!("Meta-search runs:    {}", meta_runs.len()));
    print_line(&format!("Chat-stream runs:    {}", chat.runs));

    section("Per-backend coverage");
    let succeeded = reports.iter().filter(|r| r.status == "succeeded").count();
    let failed = reports.iter().filter(|r| r.status == "failed").count();
    let skipped = reports.iter().filter(|r| r.status == "skipped").count();
    print_line(&format!(
        "summary: succeeded={succeeded} failed={failed} skipped={skipped} total={}",
        reports.len()
    ));
    for report in reports {
        let status_icon = match report.status.as_str() {
            "succeeded" => "ok ",
            "failed" => "ERR",
            "skipped" => "skp",
            _ => "???",
        };
        let mut line = format!(
            "  [{status_icon}] {:<14} hits={:<3} elapsed_ms={:<6}",
            report.name, report.hits, report.elapsed_ms
        );
        if let Some(error) = &report.error {
            line.push_str(&format!(" err={error}"));
        }
        print_line(&line);
    }

    section("Cross-query aggregates");
    print_line(&format!(
        "Total raw hits:            {}",
        meta_aggregate.total_raw_hits
    ));
    print_line(&format!(
        "Total unique URLs:         {}",
        meta_aggregate.total_unique_urls
    ));
    print_line(&format!(
        "Total deduplicated hits:   {}",
        meta_aggregate.total_deduped
    ));
    print_line(&format!(
        "Total selected by ranker:  {}",
        meta_aggregate.total_selected
    ));
    print_line(&format!(
        "Mean ranking latency (ms): {:.1}",
        meta_aggregate.mean_ranking_latency_ms
    ));
    print_line(&format!(
        "Ranking calls / second:    {:.3}",
        meta_aggregate.ranking_calls_per_second
    ));
    print_line(&format!(
        "Mean raw→unique dedup ratio:{:.2}",
        meta_aggregate.mean_raw_to_unique_dedup_ratio
    ));
    for run in meta_runs {
        print_line(&format!(
            "  run #{}: raw={} unique={} selected={} ranking_ms={} warnings={}",
            run.run_index,
            run.raw_hits,
            run.unique_urls,
            run.selected,
            run.ranking_elapsed_ms,
            run.warnings.len()
        ));
    }

    section("Chat-stream completion");
    print_line(&format!(
        "Model: {} runs={} completed={} completion_rate={:.2}",
        chat.model, chat.runs, chat.completed, chat.completion_rate
    ));
    if let Some(first_ms) = chat.first_response_ms {
        print_line(&format!("First successful stream: {first_ms}ms"));
    }
    for failure in &chat.failures {
        print_line(&format!("  - {failure}"));
    }

    section("Qualitative observations");
    if observations.is_empty() {
        print_line("  (no specific observations)");
    } else {
        for observation in observations {
            print_line(&format!(
                "  [{}] {}",
                observation.category, observation.detail
            ));
        }
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let resolved = match config::load() {
        Ok(value) => value,
        Err(error) => {
            eprintln!(
                "coverage_assessment: blocking configuration error: {error}\n\
                 Resolve by copying config.example.toml to config.toml and setting OPENROUTER_API_KEY in .env."
            );
            std::process::exit(2);
        }
    };

    if resolved.openrouter_api_key.trim().is_empty() {
        eprintln!(
            "coverage_assessment: blocking — OPENROUTER_API_KEY is empty.\n\
             Set OPENROUTER_API_KEY in .env (process environment takes precedence) before running."
        );
        std::process::exit(2);
    }

    let query = std::env::var("COVERAGE_QUERY")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_QUERY.to_string());

    let requested_runs = env_or("COVERAGE_RUNS", DEFAULT_META_SEARCH_RUNS);
    let runs = requested_runs.clamp(1, 10);

    let started_at = timestamp_string();
    let session_id = session_id_string();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));

    section("Configuration");
    print_line(&format!("chat_id:    {}", resolved.public.models.chat_id));
    print_line(&format!("rank_id:    {}", resolved.public.models.rank_id));
    print_line(&format!(
        "summarize:  {}",
        resolved.public.models.summarize_id
    ));
    print_line(&format!(
        "hit cap:    {}",
        resolved.public.search.per_backend_hit_cap
    ));
    print_line(&format!("meta-search runs: {runs}"));
    print_line(&format!("representative query: \"{query}\""));

    section("Phase 1 — per-backend coverage");
    let registry = match BackendRegistry::from_config(&resolved) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("coverage_assessment: registry build failed: {error}");
            std::process::exit(2);
        }
    };

    let enabled_names: Vec<&'static str> = registry.enabled_names();
    let enabled_set: BTreeSet<&'static str> = enabled_names.iter().copied().collect();
    let mut per_backend: Vec<PerBackendReport> = Vec::new();

    // Enabled backends — exercised live.
    let mut handles = Vec::new();
    for name in &enabled_names {
        let backend = registry.find(name).expect("registry lists it").clone();
        let query = query.clone();
        handles.push(tokio::spawn(async move {
            exercise_backend(&backend, &query).await
        }));
    }
    for handle in handles {
        match handle.await {
            Ok(report) => per_backend.push(report),
            Err(error) => print_line(&format!("  task join error: {error}")),
        }
    }

    // Disabled backends — recorded as skipped.
    let mut disabled: Vec<String> = Vec::new();
    for &name in KNOWN_PROVIDERS {
        if enabled_set.contains(name) {
            continue;
        }
        disabled.push(name.to_string());
        per_backend.push(PerBackendReport {
            name: name.to_string(),
            status: "skipped".into(),
            hits: 0,
            elapsed_ms: 0,
            error: Some("disabled in config.toml".into()),
            sample_titles: Vec::new(),
            sample_urls: Vec::new(),
        });
    }
    disabled.sort();

    per_backend.sort_by(|a, b| a.name.cmp(&b.name));

    let succeeded = per_backend
        .iter()
        .filter(|r| r.status == "succeeded")
        .count();
    let failed = per_backend.iter().filter(|r| r.status == "failed").count();
    print_line(&format!(
        "  enabled exercised: {} (succeeded={succeeded}, failed={failed})",
        enabled_names.len()
    ));
    print_line(&format!("  disabled (skipped): {}", disabled.len()));

    if succeeded == 0 {
        let blocking =
            "all enabled backends failed with unrecoverable errors; resolve by verifying \
             network access, OPENROUTER_API_KEY, and provider credentials in .env"
                .to_string();
        eprintln!("coverage_assessment: {blocking}");
        let finished_at = timestamp_string();
        let snapshot = ConfigSnapshot {
            chat_id: resolved.public.models.chat_id.clone(),
            rank_id: resolved.public.models.rank_id.clone(),
            rank_timeout_secs: resolved.public.search.rank_timeout_secs,
            summarize_id: resolved.public.models.summarize_id.clone(),
            enabled: enabled_names.iter().map(|s| s.to_string()).collect(),
            disabled: disabled.clone(),
            per_backend_hit_cap: resolved.public.search.per_backend_hit_cap,
        };
        let journal = CoverageJournal {
            metadata: SessionMetadata {
                format: "openrouter-chat-coverage".into(),
                version: 1,
                session_id: session_id.clone(),
                created_at: started_at.clone(),
                updated_at: finished_at,
            },
            config_snapshot: snapshot,
            representative_query: query.clone(),
            started_at,
            finished_at: timestamp_string(),
            per_backend,
            meta_search_runs: Vec::new(),
            meta_search_aggregate: MetaSearchAggregate {
                runs: 0,
                total_raw_hits: 0,
                total_unique_urls: 0,
                total_selected: 0,
                mean_ranking_latency_ms: 0.0,
                ranking_calls_per_second: 0.0,
                mean_raw_to_unique_dedup_ratio: 0.0,
                total_deduped: 0,
            },
            chat_stream: ChatStreamReport {
                model: resolved.public.models.chat_id.clone(),
                runs: 0,
                completed: 0,
                completion_rate: 0.0,
                first_response_ms: None,
                failures: vec![blocking.clone()],
            },
            qualitative: vec![QualitativeObservation {
                category: "blocking".into(),
                detail: blocking.clone(),
            }],
            blocking: Some(blocking),
        };

        let output_path = std::env::var("COVERAGE_OUTPUT")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| derive_session_path(root, &session_id));
        let _ = write_journal(&journal, &output_path).await;
        std::process::exit(3);
    }

    section("Phase 2 — meta-search cross-query aggregates");
    let rank_id = resolved.public.models.rank_id.clone();
    let rank_timeout_secs = resolved.public.search.rank_timeout_secs.max(1);
    print_line(&format!("ranker: {rank_id} (timeout={rank_timeout_secs}s)"));

    let inner: Arc<dyn Ranker> = Arc::new(RigRanker::from_key(
        &resolved.openrouter_api_key,
        rank_id.clone(),
        Duration::from_secs(rank_timeout_secs),
    )?);

    // Wrap the configured ranker in ChunkedListwiseRanker so a single large
    // request (often 70+ candidates after dedup) is split into multiple
    // smaller listwise calls. Smaller batches fit the LLM's per-call token
    // cap and finish well under the outer timeout. The merge re-orders the
    // combined result globally so downstream selection is unaffected.
    let chunked_batch_size = std::env::var("COVERAGE_CHUNK_BATCH_SIZE")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(openrouter_chat_rust::rank::chunked::DEFAULT_BATCH_SIZE);
    let chunked_timeout_secs = std::env::var("COVERAGE_CHUNK_PER_BATCH_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(openrouter_chat_rust::rank::chunked::DEFAULT_PER_BATCH_TIMEOUT.as_secs());
    let chunked =
        openrouter_chat_rust::rank::chunked::ChunkedListwiseRanker::new(inner, chunked_batch_size)
            .with_per_batch_timeout(Duration::from_secs(chunked_timeout_secs));
    let ranker: Arc<dyn Ranker> = Arc::new(chunked);
    print_line(&format!(
        "ranker wrapper: ChunkedListwiseRanker(batch_size={chunked_batch_size}, per_batch_timeout={chunked_timeout_secs}s)"
    ));

    let (meta_runs, meta_aggregate) =
        exercise_meta_search(&registry, ranker, &resolved.public.search, &query, runs).await;

    section("Phase 3 — chat-stream completion");
    let client = openrouter::Client::new(&resolved.openrouter_api_key)?;
    let chat = exercise_chat_stream(&client, &resolved.public.models.chat_id).await;

    let observations =
        observations_from_meta_search(&meta_runs, &["rust-lang.org", "wikipedia.org"]);

    let finished_at = timestamp_string();

    let snapshot = ConfigSnapshot {
        chat_id: resolved.public.models.chat_id.clone(),
        rank_id: rank_id.clone(),
        rank_timeout_secs,
        summarize_id: resolved.public.models.summarize_id.clone(),
        enabled: enabled_names.iter().map(|s| s.to_string()).collect(),
        disabled: disabled.clone(),
        per_backend_hit_cap: resolved.public.search.per_backend_hit_cap,
    };

    let successful_meta_runs = meta_runs
        .iter()
        .filter(|run| run.ranking_elapsed_ms > 0)
        .count();
    let blocking = (successful_meta_runs == 0)
        .then(|| format!("all {runs} meta-search runs failed before ranking completed"));
    let journal = CoverageJournal {
        metadata: SessionMetadata {
            format: "openrouter-chat-coverage".into(),
            version: 1,
            session_id: session_id.clone(),
            created_at: started_at.clone(),
            updated_at: finished_at.clone(),
        },
        config_snapshot: snapshot,
        representative_query: query.clone(),
        started_at,
        finished_at,
        per_backend,
        meta_search_runs: meta_runs,
        meta_search_aggregate: meta_aggregate,
        chat_stream: chat,
        qualitative: observations,
        blocking,
    };

    let output_path = std::env::var("COVERAGE_OUTPUT")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| derive_session_path(root, &session_id));
    write_journal(&journal, &output_path).await?;
    print_line(&format!("\nJournal artifact: {}", output_path.display()));

    render_terminal_summary(
        &journal.per_backend,
        &journal.meta_search_runs,
        &journal.meta_search_aggregate,
        &journal.chat_stream,
        &journal.qualitative,
        &query,
    );

    Ok(())
}
