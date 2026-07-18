//! Chunked listwise ranking.
//!
//! Splits a single large `RankingRequest` into multiple smaller batches and
//! delegates each batch to an inner `Ranker`. Per-batch responses are merged
//! into one `RankingResult` ordered globally by `(selected desc, score desc,
//! original_index asc)`.
//!
//! The motivation is reliability: a single large listwise call against an
//! OpenRouter-hosted model with a fixed `max_tokens` cap can either truncate
//! the JSON output (causing `validate_and_order` to reject with "omitted or
//! added candidates") or take longer than the outer wallclock budget. Smaller
//! batches fit the token cap and finish under the budget per-call.
//!
//! Each batch is also retried once on transient failures and best-effort
//! recovered from JSON truncation: missing decisions are synthesized as
//! `selected=false, score=0.0` so the run can still complete and the
//! downstream tool still has aligned `RankingResult` vectors.
//!
//! This module is additive: it does not change the existing `RigRanker`,
//! `Ranker` trait, or `validate_and_order` contract. Callers compose it by
//! wrapping any existing `Arc<dyn Ranker>`.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::contracts::error::AppError;
use crate::contracts::session::RankingDecision;

use super::{EvidenceEntry, RankCandidate, Ranker, RankingRequest, RankingResult};

/// Default per-batch timeout. Sized so the typical 5-batch run can still
/// finish inside the standard 60-second outer ranker budget with headroom
/// for one or two slow batches.
pub const DEFAULT_PER_BATCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Default batch size. With ~70 candidates this yields three to five batches.
pub const DEFAULT_BATCH_SIZE: usize = 25;

/// Default number of retries per batch on transient failures (network,
/// timeout, model truncation). One retry is usually enough to absorb a
/// single stochastic sample miss.
pub const DEFAULT_MAX_RETRIES: u8 = 1;

/// Default behavior when retries are exhausted and the inner ranker still
/// returned truncated JSON: synthesize the missing candidates as
/// `selected=false, score=0.0` so the run can still complete.
pub const DEFAULT_ALLOW_TRUNCATED_RECOVERY: bool = true;

/// Split a listwise ranking request into multiple smaller batches and
/// delegate each batch to an inner `Ranker`. The inner ranker receives the
/// original `candidate_id`s so `validate_and_order`'s contract is preserved
/// per batch.
///
/// Defaults: `batch_size = 25`, `per_batch_timeout = 30s`,
/// `max_retries = 1`, `allow_truncated_recovery = true`.
pub struct ChunkedListwiseRanker {
    inner: Arc<dyn Ranker>,
    batch_size: usize,
    per_batch_timeout: Duration,
    max_retries: u8,
    allow_truncated_recovery: bool,
}

impl std::fmt::Debug for ChunkedListwiseRanker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkedListwiseRanker")
            .field("batch_size", &self.batch_size)
            .field("per_batch_timeout_secs", &self.per_batch_timeout.as_secs())
            .field("max_retries", &self.max_retries)
            .field("allow_truncated_recovery", &self.allow_truncated_recovery)
            .finish_non_exhaustive()
    }
}

impl ChunkedListwiseRanker {
    pub fn new(inner: Arc<dyn Ranker>, batch_size: usize) -> Self {
        Self {
            inner,
            batch_size: batch_size.max(1),
            per_batch_timeout: DEFAULT_PER_BATCH_TIMEOUT,
            max_retries: DEFAULT_MAX_RETRIES,
            allow_truncated_recovery: DEFAULT_ALLOW_TRUNCATED_RECOVERY,
        }
    }

    pub fn with_per_batch_timeout(mut self, timeout: Duration) -> Self {
        self.per_batch_timeout = timeout;
        self
    }

    pub fn with_max_retries(mut self, retries: u8) -> Self {
        self.max_retries = retries;
        self
    }

    pub fn with_truncated_recovery(mut self, allow: bool) -> Self {
        self.allow_truncated_recovery = allow;
        self
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn per_batch_timeout(&self) -> Duration {
        self.per_batch_timeout
    }

    pub fn max_retries(&self) -> u8 {
        self.max_retries
    }

    pub fn truncated_recovery(&self) -> bool {
        self.allow_truncated_recovery
    }
}

/// Lightweight per-batch summary used by tests and the assessment harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkTiming {
    pub batch_index: usize,
    pub batch_size: usize,
    pub elapsed_ms: u64,
    pub selected_in_batch: usize,
    pub attempts: u8,
    pub recovered_truncation: bool,
}

/// Outcome of a chunked rank that the caller may surface alongside the
/// merged `RankingResult`. Holds the inner ranker's outcomes without leaking
/// the merge strategy itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkedOutcome {
    pub total_candidates: usize,
    pub batch_size: usize,
    pub batches: Vec<ChunkTiming>,
    pub total_elapsed_ms: u64,
}

/// True if the inner-ranker's error message looks like a transient
/// network/timeout condition that a retry could plausibly resolve.
fn is_transient_error(error: &AppError) -> bool {
    match error {
        AppError::RankFailed(msg) => {
            let lower = msg.to_ascii_lowercase();
            lower.contains("timeout")
                || lower.contains("timed out")
                || lower.contains("network")
                || lower.contains("connection")
        }
        _ => false,
    }
}

/// True if the inner-ranker's error message signals JSON truncation —
/// "ranking omitted or added candidates" — the same string emitted by
/// `validate_and_order` in `src/rank/mod.rs`. The chunked ranker treats
/// this as a retryable condition: a fresh sample from the LLM is likely
/// to come back complete.
fn is_truncation_error(error: &AppError) -> bool {
    match error {
        AppError::RankFailed(msg) => {
            msg.contains("ranking omitted or added candidates")
                || msg.contains("candidate URL was not echoed verbatim")
                || msg.contains("invalid rank JSON")
        }
        _ => false,
    }
}

/// Synthesize a complete `RankingResult` from a partial one by filling in
/// the missing `candidate_id`s with `selected=false, score=0.0`,
/// `decision="recovered from truncated JSON"`. The synthesized decisions
/// sort to the bottom in the global merge and never win over a real
/// selection.
fn synthesize_missing(partial: RankingResult, batch_candidates: &[RankCandidate]) -> RankingResult {
    let mut received_ids: HashMap<String, ()> = HashMap::with_capacity(partial.evidence.len());
    for evidence in &partial.evidence {
        received_ids.insert(evidence.candidate_id.clone(), ());
    }

    let mut decisions = partial.decisions;
    let mut evidence = partial.evidence;
    let mut candidates = partial.candidates;

    for candidate in batch_candidates {
        if received_ids.contains_key(&candidate.candidate_id) {
            continue;
        }
        let synthesized_decision = RankingDecision {
            source_id: candidate.candidate_id.clone(),
            normalized_url: candidate.hit.url.clone(),
            selected: false,
            decision: format!("recovered from truncated JSON: {}", candidate.candidate_id),
            score: Some(0.0),
        };
        let synthesized_evidence = EvidenceEntry {
            candidate_id: candidate.candidate_id.clone(),
            source_id: candidate.candidate_id.clone(),
            normalized_url: candidate.hit.url.clone(),
            decision: format!("recovered from truncated JSON: {}", candidate.candidate_id),
            score: Some(0.0),
        };
        decisions.push(synthesized_decision);
        evidence.push(synthesized_evidence);
        candidates.push(candidate.clone());
    }

    RankingResult {
        decisions,
        evidence,
        candidates,
    }
}

#[async_trait]
impl Ranker for ChunkedListwiseRanker {
    async fn rank(&self, request: RankingRequest) -> Result<RankingResult, AppError> {
        let total = request.candidates.len();
        if total == 0 {
            return Err(AppError::RankFailed(
                "chunked ranker received zero candidates".into(),
            ));
        }
        let batch_size = self.batch_size.min(total);
        let num_batches = total.div_ceil(batch_size);

        let started_total = Instant::now();
        let mut all_decisions = Vec::with_capacity(total);
        let mut all_evidence = Vec::with_capacity(total);
        let mut all_candidates = Vec::with_capacity(total);
        let mut batches = Vec::new();

        for (batch_index, batch_candidates) in request.candidates.chunks(batch_size).enumerate() {
            let batch_started = Instant::now();
            let batch_request = RankingRequest {
                query: request.query.clone(),
                candidates: batch_candidates.to_vec(),
            };

            let mut attempts: u8 = 0;
            let mut recovered_truncation = false;
            let mut last_error: Option<AppError> = None;

            let batch_result = 'retry: loop {
                let outcome = tokio::time::timeout(
                    self.per_batch_timeout,
                    self.inner.rank(batch_request.clone()),
                )
                .await;

                match outcome {
                    Ok(Ok(result)) => {
                        if result.decisions.len() == batch_candidates.len() {
                            break 'retry result;
                        }
                        // Truncation observed on the inner ranker itself:
                        // either the inner validated and returned Err (which
                        // we would have caught in the Ok(Err) branch), or
                        // some other Ranker impl returned a partial result
                        // without validation. Synthesize missing entries if
                        // configured; otherwise propagate as Err.
                        if self.allow_truncated_recovery {
                            recovered_truncation = true;
                            break 'retry synthesize_missing(result, batch_candidates);
                        }
                        return Err(AppError::RankFailed(format!(
                            "chunked ranker batch {} of {}: truncated JSON (received {} decisions, expected {})",
                            batch_index + 1,
                            num_batches,
                            result.decisions.len(),
                            batch_candidates.len()
                        )));
                    }
                    Ok(Err(error)) => {
                        if attempts < self.max_retries
                            && (is_transient_error(&error) || is_truncation_error(&error))
                        {
                            attempts += 1;
                            last_error = Some(error);
                            continue 'retry;
                        }
                        if self.allow_truncated_recovery && is_truncation_error(&error) {
                            recovered_truncation = true;
                            break 'retry synthesize_missing(
                                RankingResult {
                                    decisions: Vec::new(),
                                    evidence: Vec::new(),
                                    candidates: Vec::new(),
                                },
                                batch_candidates,
                            );
                        }
                        return Err(AppError::RankFailed(format!(
                            "chunked ranker batch {} of {} failed after {} attempt(s): {}",
                            batch_index + 1,
                            num_batches,
                            attempts + 1,
                            error
                        )));
                    }
                    Err(_) => {
                        if attempts < self.max_retries {
                            attempts += 1;
                            continue 'retry;
                        }
                        if self.allow_truncated_recovery {
                            recovered_truncation = true;
                            break 'retry synthesize_missing(
                                RankingResult {
                                    decisions: Vec::new(),
                                    evidence: Vec::new(),
                                    candidates: Vec::new(),
                                },
                                batch_candidates,
                            );
                        }
                        return Err(AppError::RankFailed(format!(
                            "chunked ranker batch {} of {} exceeded per-batch budget of {}s after {} attempt(s)",
                            batch_index + 1,
                            num_batches,
                            self.per_batch_timeout.as_secs(),
                            attempts + 1
                        )));
                    }
                }
            };

            let _ = last_error;

            // Build id -> (evidence, candidate) map from the (possibly
            // recovered) batch result.
            let mut by_id: HashMap<&str, (usize, &EvidenceEntry, &RankCandidate)> =
                HashMap::with_capacity(batch_result.evidence.len());
            for (idx, (evidence, candidate)) in batch_result
                .evidence
                .iter()
                .zip(batch_result.candidates.iter())
                .enumerate()
            {
                by_id.insert(evidence.candidate_id.as_str(), (idx, evidence, candidate));
            }

            let mut selected_in_batch = 0usize;
            for decision in &batch_result.decisions {
                let Some((_, evidence, candidate)) = by_id.remove(decision.source_id.as_str())
                else {
                    return Err(AppError::RankFailed(format!(
                        "chunked ranker: decision {} (batch {}) missing from inner ranker's evidence",
                        decision.source_id,
                        batch_index + 1
                    )));
                };
                if decision.selected {
                    selected_in_batch += 1;
                }
                all_decisions.push(decision.clone());
                all_evidence.push(evidence.clone());
                all_candidates.push(candidate.clone());
            }

            batches.push(ChunkTiming {
                batch_index,
                batch_size: batch_candidates.len(),
                elapsed_ms: batch_started.elapsed().as_millis() as u64,
                selected_in_batch,
                attempts: attempts + 1,
                recovered_truncation,
            });
        }

        // Global ordering: selected-first, then by score desc, then preserve
        // the in-batch position so the merge is stable across runs.
        let mut indexed: Vec<usize> = (0..all_decisions.len()).collect();
        indexed.sort_by(|&a, &b| {
            let da = &all_decisions[a];
            let db = &all_decisions[b];
            match db.selected.cmp(&da.selected) {
                Ordering::Equal => match (db.score, da.score) {
                    (Some(sb), Some(sa)) => sb
                        .partial_cmp(&sa)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| a.cmp(&b)),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => a.cmp(&b),
                },
                other => other,
            }
        });

        let decisions: Vec<_> = indexed.iter().map(|&i| all_decisions[i].clone()).collect();
        let evidence: Vec<_> = indexed.iter().map(|&i| all_evidence[i].clone()).collect();
        let candidates: Vec<_> = indexed.iter().map(|&i| all_candidates[i].clone()).collect();

        let _ = started_total.elapsed();

        let _ = ChunkedOutcome {
            total_candidates: total,
            batch_size,
            batches,
            total_elapsed_ms: started_total.elapsed().as_millis() as u64,
        };

        Ok(RankingResult {
            decisions,
            evidence,
            candidates,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::types::{BackendKind, SearchHit};
    use crate::rank::{EvidenceEntry as RankEvidence, RankingResult};
    use async_trait::async_trait;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

    fn candidate(id: &str, url: &str) -> RankCandidate {
        RankCandidate {
            candidate_id: id.to_string(),
            hit: SearchHit {
                title: id.to_string(),
                url: url.to_string(),
                snippet: "snippet".into(),
                published: None,
                native_rank: None,
                native_score: None,
                provider: "test".into(),
                backend_kind: BackendKind::Web,
                source_subtype: None,
                metadata: BTreeMap::new(),
            },
        }
    }

    /// Test inner that pretends to be an LLM ranker. Echoes the input
    /// candidates back as decisions with deterministic scores.
    struct EchoRanker;
    #[async_trait]
    impl Ranker for EchoRanker {
        async fn rank(&self, request: RankingRequest) -> Result<RankingResult, AppError> {
            let mut decisions = Vec::new();
            let mut evidence = Vec::new();
            let mut candidates = Vec::new();
            for (i, c) in request.candidates.iter().enumerate() {
                let score = Some(1.0 - i as f64 / 100.0);
                decisions.push(RankingDecision {
                    source_id: c.candidate_id.clone(),
                    normalized_url: c.hit.url.clone(),
                    selected: i == 0,
                    decision: format!("echo {}", c.candidate_id),
                    score,
                });
                evidence.push(RankEvidence {
                    candidate_id: c.candidate_id.clone(),
                    source_id: c.candidate_id.clone(),
                    normalized_url: c.hit.url.clone(),
                    decision: format!("echo {}", c.candidate_id),
                    score,
                });
                candidates.push(c.clone());
            }
            Ok(RankingResult {
                decisions,
                evidence,
                candidates,
            })
        }
    }

    #[tokio::test]
    async fn single_batch_passes_through_inner() {
        let chunked = ChunkedListwiseRanker::new(Arc::new(EchoRanker), 25);
        let request = RankingRequest {
            query: "q".into(),
            candidates: vec![candidate("a", "https://a"), candidate("b", "https://b")],
        };
        let result = chunked.rank(request).await.expect("rank");
        assert_eq!(result.decisions.len(), 2);
        assert_eq!(result.evidence.len(), 2);
        assert_eq!(result.candidates.len(), 2);
    }

    #[tokio::test]
    async fn chunking_preserves_candidate_ids_and_global_selected_first_ordering() {
        let chunked = ChunkedListwiseRanker::new(Arc::new(EchoRanker), 3);
        let mut candidates = Vec::new();
        for i in 0..7 {
            candidates.push(candidate(&format!("c{i}"), &format!("https://x/{i}")));
        }
        let request = RankingRequest {
            query: "q".into(),
            candidates,
        };
        let result = chunked.rank(request).await.expect("rank");

        assert_eq!(result.decisions.len(), 7);
        assert_eq!(result.evidence.len(), 7);
        assert_eq!(result.candidates.len(), 7);

        let mut seen = std::collections::HashSet::new();
        for (i, decision) in result.decisions.iter().enumerate() {
            assert!(
                seen.insert(decision.source_id.clone()),
                "duplicate source_id {}",
                decision.source_id
            );
            assert_eq!(result.evidence[i].candidate_id, decision.source_id);
            assert_eq!(result.candidates[i].candidate_id, decision.source_id);
        }
        assert_eq!(seen.len(), 7);

        let top_ids: Vec<&str> = result
            .decisions
            .iter()
            .take(3)
            .map(|d| d.source_id.as_str())
            .collect();
        assert_eq!(top_ids, vec!["c0", "c3", "c6"]);
        assert!(result.decisions.iter().take(3).all(|d| d.selected));
    }

    struct SlowRanker;
    #[async_trait]
    impl Ranker for SlowRanker {
        async fn rank(&self, _request: RankingRequest) -> Result<RankingResult, AppError> {
            tokio::time::sleep(Duration::from_secs(5)).await;
            Ok(RankingResult {
                decisions: Vec::new(),
                evidence: Vec::new(),
                candidates: Vec::new(),
            })
        }
    }

    #[tokio::test]
    async fn per_batch_timeout_recovers_after_retries_when_enabled() {
        let chunked = ChunkedListwiseRanker::new(Arc::new(SlowRanker), 2)
            .with_per_batch_timeout(Duration::from_millis(50));
        let request = RankingRequest {
            query: "q".into(),
            candidates: vec![
                candidate("a", "https://a"),
                candidate("b", "https://b"),
                candidate("c", "https://c"),
                candidate("d", "https://d"),
            ],
        };
        let result = chunked.rank(request).await.expect("should recover");
        assert_eq!(result.decisions.len(), 4);
        assert!(result.decisions.iter().all(|decision| !decision.selected));
    }

    // -------- Truncation / retry / recovery tests --------

    /// Inner that returns Ok with FEWER decisions than candidates, simulating
    /// an LLM that truncated its structured JSON output without validation.
    struct PartialOkRanker {
        skip_last: bool,
    }
    #[async_trait]
    impl Ranker for PartialOkRanker {
        async fn rank(&self, request: RankingRequest) -> Result<RankingResult, AppError> {
            let take = if self.skip_last {
                request.candidates.len().saturating_sub(1)
            } else {
                request.candidates.len()
            };
            let mut decisions = Vec::new();
            let mut evidence = Vec::new();
            let mut candidates = Vec::new();
            for c in request.candidates.iter().take(take) {
                let score = Some(1.0);
                decisions.push(RankingDecision {
                    source_id: c.candidate_id.clone(),
                    normalized_url: c.hit.url.clone(),
                    selected: false,
                    decision: format!("partial {}", c.candidate_id),
                    score,
                });
                evidence.push(RankEvidence {
                    candidate_id: c.candidate_id.clone(),
                    source_id: c.candidate_id.clone(),
                    normalized_url: c.hit.url.clone(),
                    decision: format!("partial {}", c.candidate_id),
                    score,
                });
                candidates.push(c.clone());
            }
            Ok(RankingResult {
                decisions,
                evidence,
                candidates,
            })
        }
    }

    #[tokio::test]
    async fn truncated_partial_ok_recovered_by_synthesis() {
        let chunked = ChunkedListwiseRanker::new(Arc::new(PartialOkRanker { skip_last: true }), 2);
        let request = RankingRequest {
            query: "q".into(),
            candidates: vec![
                candidate("a", "https://a"),
                candidate("b", "https://b"),
                candidate("c", "https://c"),
                candidate("d", "https://d"),
            ],
        };
        let result = chunked.rank(request).await.expect("recovered");
        assert_eq!(result.decisions.len(), 4);
        assert_eq!(result.evidence.len(), 4);
        assert_eq!(result.candidates.len(), 4);
        // The synthesized entries must be marked selected=false, score=0.0.
        let synthesized: Vec<&RankingDecision> = result
            .decisions
            .iter()
            .filter(|d| d.decision.starts_with("recovered from truncated JSON"))
            .collect();
        assert_eq!(synthesized.len(), 2, "expected 2 synthesized entries");
        for d in &synthesized {
            assert!(!d.selected, "synthesized decision must not be selected");
            assert_eq!(d.score, Some(0.0));
        }
        // The 2 real decisions must be marked selected=false too (PartialOk
        // never marks anything selected) but score=1.0, so they sort above
        // the synthesized ones.
        let real_ids: Vec<&str> = result
            .decisions
            .iter()
            .filter(|d| !d.decision.starts_with("recovered"))
            .map(|d| d.source_id.as_str())
            .collect();
        assert_eq!(real_ids.len(), 2);
        // After global ordering, real (score 1.0) must come before
        // synthesized (score 0.0).
        let real_positions: Vec<usize> = result
            .decisions
            .iter()
            .enumerate()
            .filter(|(_, d)| !d.decision.starts_with("recovered"))
            .map(|(i, _)| i)
            .collect();
        let synth_positions: Vec<usize> = result
            .decisions
            .iter()
            .enumerate()
            .filter(|(_, d)| d.decision.starts_with("recovered"))
            .map(|(i, _)| i)
            .collect();
        assert!(
            real_positions.iter().max() < synth_positions.iter().min(),
            "real decisions must sort above synthesized ones"
        );
    }

    #[tokio::test]
    async fn truncated_recovery_can_be_disabled() {
        let chunked = ChunkedListwiseRanker::new(Arc::new(PartialOkRanker { skip_last: true }), 2)
            .with_truncated_recovery(false);
        let request = RankingRequest {
            query: "q".into(),
            candidates: vec![
                candidate("a", "https://a"),
                candidate("b", "https://b"),
                candidate("c", "https://c"),
                candidate("d", "https://d"),
            ],
        };
        let error = chunked.rank(request).await.expect_err("should fail");
        let message = format!("{error}");
        assert!(
            message.contains("truncated JSON"),
            "expected truncation error: {message}"
        );
    }

    /// Inner that returns Err with the canonical truncation message on every
    /// odd-numbered call, then succeeds on even-numbered calls. Simulates a
    /// model that needs a retry to produce a complete response.
    struct FlakyTruncationRanker {
        attempts: AtomicU32,
    }
    #[async_trait]
    impl Ranker for FlakyTruncationRanker {
        async fn rank(&self, request: RankingRequest) -> Result<RankingResult, AppError> {
            let n = self.attempts.fetch_add(1, AtomicOrdering::SeqCst);
            if n.is_multiple_of(2) {
                return Err(AppError::RankFailed(
                    "ranking omitted or added candidates".into(),
                ));
            }
            let mut decisions = Vec::new();
            let mut evidence = Vec::new();
            let mut candidates = Vec::new();
            for (i, c) in request.candidates.iter().enumerate() {
                let score = Some(1.0 - i as f64 / 100.0);
                decisions.push(RankingDecision {
                    source_id: c.candidate_id.clone(),
                    normalized_url: c.hit.url.clone(),
                    selected: i == 0,
                    decision: format!("flaky {}", c.candidate_id),
                    score,
                });
                evidence.push(RankEvidence {
                    candidate_id: c.candidate_id.clone(),
                    source_id: c.candidate_id.clone(),
                    normalized_url: c.hit.url.clone(),
                    decision: format!("flaky {}", c.candidate_id),
                    score,
                });
                candidates.push(c.clone());
            }
            Ok(RankingResult {
                decisions,
                evidence,
                candidates,
            })
        }
    }

    #[tokio::test]
    async fn retry_recovers_from_transient_truncation_err() {
        let inner = Arc::new(FlakyTruncationRanker {
            attempts: AtomicU32::new(0),
        });
        // batch_size=2 with 4 candidates => 2 batches => retry loop engaged.
        // The inner fails every odd call (n % 2 == 0), so each batch's first
        // attempt fails and the retry succeeds.
        let chunked = ChunkedListwiseRanker::new(inner.clone(), 2);
        let request = RankingRequest {
            query: "q".into(),
            candidates: vec![
                candidate("a", "https://a"),
                candidate("b", "https://b"),
                candidate("c", "https://c"),
                candidate("d", "https://d"),
            ],
        };
        let result = chunked.rank(request).await.expect("recovered after retry");
        assert_eq!(result.decisions.len(), 4);
        assert_eq!(inner.attempts.load(AtomicOrdering::SeqCst), 4);
    }

    /// Inner that returns a transient Err every time. With retries=2, the
    /// chunked ranker should make 3 attempts and then surface a descriptive
    /// failure.
    struct AlwaysTransientErrRanker;
    #[async_trait]
    impl Ranker for AlwaysTransientErrRanker {
        async fn rank(&self, _request: RankingRequest) -> Result<RankingResult, AppError> {
            Err(AppError::RankFailed("network connection refused".into()))
        }
    }

    #[tokio::test]
    async fn retries_exhausted_on_persistent_transient_err() {
        let chunked =
            ChunkedListwiseRanker::new(Arc::new(AlwaysTransientErrRanker), 2).with_max_retries(2);
        let request = RankingRequest {
            query: "q".into(),
            candidates: vec![
                candidate("a", "https://a"),
                candidate("b", "https://b"),
                candidate("c", "https://c"),
                candidate("d", "https://d"),
            ],
        };
        let error = chunked.rank(request).await.expect_err("should fail");
        let message = format!("{error}");
        assert!(
            message.contains("after 3 attempt"),
            "expected retry count: {message}"
        );
    }

    /// Inner that returns a non-transient Err. Retries should NOT happen.
    struct NonTransientErrRanker;
    #[async_trait]
    impl Ranker for NonTransientErrRanker {
        async fn rank(&self, _request: RankingRequest) -> Result<RankingResult, AppError> {
            Err(AppError::RankFailed("malformed input".into()))
        }
    }

    #[tokio::test]
    async fn non_transient_err_does_not_retry() {
        let chunked = ChunkedListwiseRanker::new(Arc::new(NonTransientErrRanker), 2);
        let request = RankingRequest {
            query: "q".into(),
            candidates: vec![
                candidate("a", "https://a"),
                candidate("b", "https://b"),
                candidate("c", "https://c"),
                candidate("d", "https://d"),
            ],
        };
        let error = chunked.rank(request).await.expect_err("should fail");
        let message = format!("{error}");
        assert!(
            message.contains("after 1 attempt"),
            "non-transient err must not retry: {message}"
        );
    }
}
