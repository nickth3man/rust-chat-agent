//! The independent AI ranking lane.
//!
//! This module deliberately owns the boundary between search results and the
//! model.  In particular, model output is never allowed to invent, omit, or
//! reorder an identifier without being checked against the request.

use async_trait::async_trait;
use rig_core::{
    client::CompletionClient,
    completion::{AssistantContent, CompletionRequestBuilder},
    providers::openrouter,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, time::Duration};

use crate::contracts::{error::AppError, session::RankingDecision, types::SearchHit};

/// A candidate identifier is assigned by the caller (normally the
/// meta-search deduplication layer), rather than by the ranker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankCandidate {
    pub candidate_id: String,
    pub hit: SearchHit,
}

/// Evidence is kept separate from the frozen session type so this lane can be
/// used before session persistence is wired in.  Its identifiers intentionally
/// match `RankingDecision`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceEntry {
    pub candidate_id: String,
    pub source_id: String,
    pub normalized_url: String,
    pub decision: String,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingRequest {
    pub query: String,
    pub candidates: Vec<RankCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingResult {
    pub decisions: Vec<RankingDecision>,
    pub evidence: Vec<EvidenceEntry>,
    pub candidates: Vec<RankCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ModelDecision {
    candidate_id: String,
    selected: bool,
    decision: String,
    score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ModelOutput {
    decisions: Vec<ModelDecision>,
}

#[async_trait]
pub trait Ranker: Send + Sync {
    async fn rank(&self, request: RankingRequest) -> Result<RankingResult, AppError>;
}

/// OpenRouter-backed ranker.  `CompletionModel` is the concrete Rig 0.40
/// model handle; callers can construct it with `Client::new(key)` and
/// `client.completion_model(model_id)`.
#[derive(Clone)]
pub struct RigRanker {
    model: openrouter::CompletionModel<reqwest::Client>,
    timeout: Duration,
}

impl RigRanker {
    pub fn new(
        client: openrouter::Client<reqwest::Client>,
        model_id: impl Into<String>,
        timeout: Duration,
    ) -> Self {
        Self {
            model: client.completion_model(model_id),
            timeout,
        }
    }

    pub fn from_key(
        api_key: &str,
        model_id: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self, AppError> {
        let client = openrouter::Client::new(api_key)
            .map_err(|error| AppError::RankFailed(format!("OpenRouter client: {error}")))?;
        Ok(Self::new(client, model_id, timeout))
    }
}

#[async_trait]
impl Ranker for RigRanker {
    async fn rank(&self, request: RankingRequest) -> Result<RankingResult, AppError> {
        let prompt = build_prompt(&request)?;
        let schema = schemars::schema_for!(ModelOutput);
        let completion = CompletionRequestBuilder::new(self.model.clone(), prompt)
            .preamble("Return only JSON matching the supplied schema. Rank only the supplied candidate IDs. Keep decision text concise.".to_string())
            .temperature(0.0)
            .max_tokens(2048)
            .output_schema(schema)
            .send();
        let response = tokio::time::timeout(self.timeout, completion)
            .await
            .map_err(|_| AppError::RankFailed("ranker timed out".into()))?
            .map_err(|error| AppError::RankFailed(format!("completion failed: {error:?}")))?;
        let text = response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        let output: ModelOutput = serde_json::from_str(&text)
            .map_err(|error| AppError::RankFailed(format!("invalid rank JSON: {error}")))?;
        validate_and_order(request, output)
    }
}

fn build_prompt(request: &RankingRequest) -> Result<String, AppError> {
    if request.candidates.is_empty() {
        return Err(AppError::RankFailed("no ranking candidates".into()));
    }
    let compact = request
        .candidates
        .iter()
        .map(|candidate| {
            serde_json::json!({
                "candidate_id": candidate.candidate_id,
                "title": candidate.hit.title,
                "url": candidate.hit.url,
                "snippet": candidate.hit.snippet,
                "published": candidate.hit.published,
                "provider": candidate.hit.provider,
                "backend_kind": candidate.hit.backend_kind,
                "source_subtype": candidate.hit.source_subtype,
                "native_rank": candidate.hit.native_rank,
                "native_score": candidate.hit.native_score,
                "metadata": candidate.hit.metadata,
            })
        })
        .collect::<Vec<_>>();
    Ok(format!(
        "Query: {}\nCandidates (JSON): {}\nReturn every candidate exactly once in ranked order. Set selected, score (0..1), and a concise decision.",
        request.query,
        serde_json::to_string(&compact).map_err(|e| AppError::RankFailed(e.to_string()))?
    ))
}

fn validate_and_order(
    request: RankingRequest,
    output: ModelOutput,
) -> Result<RankingResult, AppError> {
    let expected: HashSet<&str> = request
        .candidates
        .iter()
        .map(|candidate| candidate.candidate_id.as_str())
        .collect();
    let mut seen = HashSet::new();
    if output.decisions.len() != expected.len() {
        return Err(AppError::RankFailed(
            "ranking omitted or added candidates".into(),
        ));
    }
    let mut candidates_by_id = request
        .candidates
        .iter()
        .map(|candidate| (candidate.candidate_id.as_str(), candidate))
        .collect::<std::collections::HashMap<_, _>>();
    let mut candidates = Vec::with_capacity(output.decisions.len());
    let mut decisions = Vec::with_capacity(output.decisions.len());
    let mut evidence = Vec::with_capacity(output.decisions.len());
    for decision in output.decisions {
        if !expected.contains(decision.candidate_id.as_str()) {
            return Err(AppError::RankFailed(format!(
                "unknown candidate ID: {}",
                decision.candidate_id
            )));
        }
        if !seen.insert(decision.candidate_id.clone()) {
            return Err(AppError::RankFailed(format!(
                "duplicate candidate ID: {}",
                decision.candidate_id
            )));
        }
        let candidate = candidates_by_id
            .remove(decision.candidate_id.as_str())
            .expect("validated candidate ID");
        let normalized_url = candidate.hit.url.clone();
        decisions.push(RankingDecision {
            source_id: decision.candidate_id.clone(),
            normalized_url: normalized_url.clone(),
            selected: decision.selected,
            decision: decision.decision.clone(),
            score: decision.score,
        });
        evidence.push(EvidenceEntry {
            candidate_id: decision.candidate_id,
            source_id: decisions.last().unwrap().source_id.clone(),
            normalized_url,
            decision: decision.decision,
            score: decision.score,
        });
        candidates.push(candidate.clone());
    }
    Ok(RankingResult {
        decisions,
        evidence,
        candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::types::BackendKind;
    use std::collections::BTreeMap;

    fn request() -> RankingRequest {
        RankingRequest {
            query: "rust".into(),
            candidates: ["a", "b"]
                .into_iter()
                .map(|id| RankCandidate {
                    candidate_id: id.into(),
                    hit: SearchHit {
                        title: id.into(),
                        url: format!("https://{id}.example"),
                        snippet: "snippet".into(),
                        published: None,
                        native_rank: None,
                        native_score: None,
                        provider: "test".into(),
                        backend_kind: BackendKind::Web,
                        source_subtype: None,
                        metadata: BTreeMap::new(),
                    },
                })
                .collect(),
        }
    }
    fn output(ids: &[&str]) -> ModelOutput {
        ModelOutput {
            decisions: ids
                .iter()
                .map(|id| ModelDecision {
                    candidate_id: (*id).into(),
                    selected: true,
                    decision: "good".into(),
                    score: Some(0.9),
                })
                .collect(),
        }
    }

    #[test]
    fn validates_and_returns_model_order() {
        let result = validate_and_order(request(), output(&["b", "a"])).unwrap();
        assert_eq!(result.candidates[0].candidate_id, "b");
        assert_eq!(result.decisions[1].source_id, "a");
    }
    #[test]
    fn rejects_unknown_duplicate_and_missing_ids() {
        assert!(validate_and_order(request(), output(&["a", "x"])).is_err());
        assert!(validate_and_order(request(), output(&["a", "a"])).is_err());
        assert!(validate_and_order(request(), output(&["a"])).is_err());
    }
    #[test]
    fn rejects_malformed_json() {
        assert!(serde_json::from_str::<ModelOutput>("not json").is_err());
    }
}
