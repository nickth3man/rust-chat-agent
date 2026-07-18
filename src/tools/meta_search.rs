//! Concurrent federated search and model-assisted result selection.

use crate::config::SearchConfig;
use crate::contracts::{
    error::{AppError, ToolNetError},
    provenance::{EvidenceEntry, TurnProvenance},
    types::SearchHit,
};
use crate::rank::{RankCandidate, Ranker, RankingRequest, RankingResult};
use crate::search::BackendRegistry;
use futures::future::join_all;
use reqwest::Url;
use rig_core::tool::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, broadcast};

const MAX_FIELD_CHARS: usize = 512;
const ACTIVITY_CAP: usize = 256;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct MetaSearchArgs {
    pub query: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SelectedHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub providers: Vec<String>,
    pub source_subtypes: Vec<String>,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankedEvidence {
    pub normalized_url: String,
    pub decision: String,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetaSearchOutput {
    pub query: String,
    pub backends_queried: Vec<String>,
    pub selected: Vec<SelectedHit>,
    pub warnings: Vec<String>,
    pub ranked_evidence: Vec<RankedEvidence>,
}

#[derive(Debug, Clone, Serialize)]
pub enum SearchActivity {
    QueryStarted {
        query: String,
    },
    ProviderStarted {
        provider: String,
    },
    ProviderResult {
        provider: String,
        elapsed_ms: u64,
        hits: usize,
        retry_count: Option<u32>,
        normalized_hits: Vec<SearchHit>,
    },
    ProviderError {
        provider: String,
        elapsed_ms: u64,
        error: String,
        retry_count: Option<u32>,
    },
    RankingStarted {
        candidates: usize,
    },
    RankingCompleted {
        elapsed_ms: u64,
        selected: usize,
        decisions: Vec<crate::contracts::session::RankingDecision>,
    },
    RankingFailed {
        elapsed_ms: u64,
        error: String,
    },
}

#[derive(Clone)]
pub struct MetaSearchState {
    activity_tx: broadcast::Sender<SearchActivity>,
    activity_log: Arc<Mutex<Vec<SearchActivity>>>,
    provenance: Arc<Mutex<BTreeMap<String, TurnProvenance>>>,
    active_turn: Arc<Mutex<Option<String>>>,
}

impl Default for MetaSearchState {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaSearchState {
    pub fn new() -> Self {
        let (activity_tx, _) = broadcast::channel(128);
        Self {
            activity_tx,
            activity_log: Arc::new(Mutex::new(Vec::new())),
            provenance: Arc::new(Mutex::new(BTreeMap::new())),
            active_turn: Arc::new(Mutex::new(None)),
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<SearchActivity> {
        self.activity_tx.subscribe()
    }

    pub async fn activities(&self) -> Vec<SearchActivity> {
        self.activity_log.lock().await.clone()
    }

    pub async fn begin_turn(&self, turn_id: impl Into<String>) {
        let turn_id = turn_id.into();
        self.activity_log.lock().await.clear();
        let mut map = self.provenance.lock().await;
        map.insert(
            turn_id.clone(),
            TurnProvenance {
                turn_id: turn_id.clone(),
                entries: Vec::new(),
            },
        );
        *self.active_turn.lock().await = Some(turn_id);
    }

    pub async fn take_activities(&self) -> Vec<SearchActivity> {
        std::mem::take(&mut *self.activity_log.lock().await)
    }

    pub async fn snapshot(&self, turn_id: &str) -> Option<TurnProvenance> {
        self.provenance.lock().await.get(turn_id).cloned()
    }

    pub async fn take(&self, turn_id: &str) -> Option<TurnProvenance> {
        let result = self.provenance.lock().await.remove(turn_id);
        let mut active = self.active_turn.lock().await;
        if active.as_deref() == Some(turn_id) {
            *active = None;
        }
        result
    }

    async fn publish(&self, event: SearchActivity) {
        let mut log = self.activity_log.lock().await;
        log.push(event.clone());
        if log.len() > ACTIVITY_CAP {
            log.remove(0);
        }
        let _ = self.activity_tx.send(event);
    }

    async fn add_provenance(&self, turn_id: &str, entries: Vec<EvidenceEntry>) {
        let mut map = self.provenance.lock().await;
        let turn = map
            .entry(turn_id.to_owned())
            .or_insert_with(|| TurnProvenance {
                turn_id: turn_id.to_owned(),
                entries: Vec::new(),
            });
        turn.merge(entries);
    }

    async fn selected_turn(&self, fallback: &str) -> String {
        self.active_turn
            .lock()
            .await
            .clone()
            .unwrap_or_else(|| fallback.to_owned())
    }
}

#[derive(Debug)]
pub enum MetaSearchError {
    InvalidQuery,
    Search(AppError),
    Ranking(AppError),
    NoBackendsSucceeded,
    NoSelectedCandidates,
    Output(String),
}

impl fmt::Display for MetaSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidQuery => f.write_str("search query must not be empty"),
            Self::Search(e) => write!(f, "search failed: {e}"),
            Self::Ranking(e) => write!(f, "ranking failed: {e}"),
            Self::NoBackendsSucceeded => f.write_str("all search providers failed"),
            Self::NoSelectedCandidates => f.write_str("ranker selected zero candidates"),
            Self::Output(e) => write!(f, "search output failed: {e}"),
        }
    }
}
impl std::error::Error for MetaSearchError {}

pub struct MetaSearch {
    registry: BackendRegistry,
    ranker: Arc<dyn Ranker>,
    config: SearchConfig,
    state: MetaSearchState,
}

impl MetaSearch {
    pub fn new(registry: BackendRegistry, ranker: Arc<dyn Ranker>, config: SearchConfig) -> Self {
        Self::with_state(registry, ranker, config, MetaSearchState::new())
    }

    pub fn with_state(
        registry: BackendRegistry,
        ranker: Arc<dyn Ranker>,
        config: SearchConfig,
        state: MetaSearchState,
    ) -> Self {
        Self {
            registry,
            ranker,
            config,
            state,
        }
    }

    pub fn state(&self) -> MetaSearchState {
        self.state.clone()
    }

    async fn execute(&self, args: MetaSearchArgs) -> Result<MetaSearchOutput, MetaSearchError> {
        let query = args.query.trim().to_owned();
        if query.is_empty() {
            return Err(MetaSearchError::InvalidQuery);
        }
        self.state
            .publish(SearchActivity::QueryStarted {
                query: query.clone(),
            })
            .await;
        let names = self
            .registry
            .enabled_names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let budget = Duration::from_secs(self.config.stage_budget_secs.max(1));
        let futures = self.registry.iter().map(|backend| {
            let backend = Arc::clone(backend);
            let state = self.state.clone();
            let query = query.clone();
            async move {
                let name = backend.name().to_owned();
                state
                    .publish(SearchActivity::ProviderStarted {
                        provider: name.clone(),
                    })
                    .await;
                let started = Instant::now();
                let result = tokio::time::timeout(budget, backend.search(&query)).await;
                let elapsed_ms = started.elapsed().as_millis() as u64;
                match result {
                    Ok(Ok(hits)) => {
                        let normalized_hits = hits
                            .iter()
                            .cloned()
                            .map(normalize_provider_hit)
                            .collect::<Vec<_>>();
                        state
                            .publish(SearchActivity::ProviderResult {
                                provider: name.clone(),
                                elapsed_ms,
                                hits: hits.len(),
                                retry_count: None,
                                normalized_hits,
                            })
                            .await;
                        (name, Ok(hits))
                    }
                    Ok(Err(error)) => {
                        let message = safe_error(&error);
                        state
                            .publish(SearchActivity::ProviderError {
                                provider: name.clone(),
                                elapsed_ms,
                                error: message.clone(),
                                retry_count: None,
                            })
                            .await;
                        (name, Err(message))
                    }
                    Err(_) => {
                        let message = "provider stage timed out".to_owned();
                        state
                            .publish(SearchActivity::ProviderError {
                                provider: name.clone(),
                                elapsed_ms,
                                error: message.clone(),
                                retry_count: None,
                            })
                            .await;
                        (name, Err(message))
                    }
                }
            }
        });
        let outcomes = join_all(futures).await;
        let mut all_hits = Vec::new();
        let mut warnings = Vec::new();
        let mut success_count = 0;
        for (name, result) in outcomes {
            match result {
                Ok(hits) => {
                    success_count += 1;
                    all_hits.extend(hits.into_iter().map(|hit| (name.clone(), hit)));
                }
                Err(error) => warnings.push(format!("{name}: {error}")),
            }
        }
        if success_count == 0 {
            return Err(MetaSearchError::NoBackendsSucceeded);
        }
        let deduped = deduplicate(all_hits);
        if deduped.is_empty() {
            return Err(MetaSearchError::NoSelectedCandidates);
        }
        let mut candidate_lookup = BTreeMap::new();
        let candidates = deduped
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let candidate_id = format!("c{index}");
                candidate_lookup.insert(candidate_id.clone(), item.clone());
                RankCandidate {
                    candidate_id,
                    hit: item.hit.clone(),
                }
            })
            .collect::<Vec<_>>();
        self.state
            .publish(SearchActivity::RankingStarted {
                candidates: candidates.len(),
            })
            .await;
        let started = Instant::now();
        let ranked = tokio::time::timeout(
            Duration::from_secs(self.config.rank_timeout_secs.max(1)),
            self.ranker.rank(RankingRequest {
                query: query.clone(),
                candidates,
            }),
        )
        .await
        .map_err(|_| AppError::RankFailed("ranker timed out".into()))
        .and_then(|result| result);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let ranked = match ranked {
            Ok(value) => {
                self.state
                    .publish(SearchActivity::RankingCompleted {
                        elapsed_ms,
                        selected: value.decisions.iter().filter(|d| d.selected).count(),
                        decisions: value.decisions.clone(),
                    })
                    .await;
                value
            }
            Err(error) => {
                self.state
                    .publish(SearchActivity::RankingFailed {
                        elapsed_ms,
                        error: safe_app_error(&error),
                    })
                    .await;
                return Err(MetaSearchError::Ranking(error));
            }
        };
        if !ranked.decisions.iter().any(|decision| decision.selected) {
            return Err(MetaSearchError::NoSelectedCandidates);
        }
        self.record_provenance(&query, &ranked, &candidate_lookup)
            .await;
        let mut selected = Vec::new();
        let mut ranked_evidence = Vec::new();
        for decision in ranked.decisions.iter().filter(|d| d.selected) {
            if let Some(item) = candidate_lookup.get(decision.source_id.as_str()) {
                selected.push(SelectedHit {
                    title: bound(&item.hit.title),
                    url: bound(&item.hit.url),
                    snippet: bound(&item.hit.snippet),
                    providers: labels(item, "provider_labels"),
                    source_subtypes: labels(item, "source_subtypes"),
                    score: decision.score,
                });
                ranked_evidence.push(RankedEvidence {
                    normalized_url: bound(&item.key),
                    decision: bound(&decision.decision),
                    score: decision.score,
                });
            }
        }
        cap_output(
            MetaSearchOutput {
                query,
                backends_queried: names,
                selected,
                warnings,
                ranked_evidence,
            },
            self.config.model_output_bytes.max(1),
        )
    }

    async fn record_provenance(
        &self,
        query: &str,
        ranked: &RankingResult,
        candidate_lookup: &BTreeMap<String, NormalizedHit>,
    ) {
        // A stable query key permits runtime wiring to call begin_turn separately,
        // while still making direct tool use observable.
        let turn_id = self.state.selected_turn(query).await;
        let entries = ranked
            .decisions
            .iter()
            .filter(|d| d.selected)
            .filter_map(|decision| {
                candidate_lookup
                    .get(decision.source_id.as_str())
                    .map(|item| EvidenceEntry {
                        source_id: decision.source_id.clone(),
                        normalized_url: item.key.clone(),
                        title: item.hit.title.clone(),
                        url: item.hit.url.clone(),
                        supporting_snippet: item.hit.snippet.clone(),
                        rank_decision: Some(decision.decision.clone()),
                        provider_labels: item.providers.clone(),
                        source_subtypes: item.subtypes.clone(),
                    })
            })
            .collect();
        self.state.add_provenance(&turn_id, entries).await;
    }
}

impl Tool for MetaSearch {
    const NAME: &'static str = "meta_search";
    type Error = MetaSearchError;
    type Args = MetaSearchArgs;
    type Output = MetaSearchOutput;

    fn description(&self) -> String {
        "Search all configured providers concurrently, deduplicate results, and return model-ranked evidence.".into()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(MetaSearchArgs)).expect("schema is serializable")
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.execute(args).await
    }
}

#[derive(Clone)]
struct NormalizedHit {
    key: String,
    hit: SearchHit,
    providers: BTreeSet<String>,
    subtypes: BTreeSet<String>,
}

fn deduplicate(items: Vec<(String, SearchHit)>) -> Vec<NormalizedHit> {
    let mut map = BTreeMap::<String, NormalizedHit>::new();
    for (provider, mut hit) in items {
        let key = normalize_url(&hit.url);
        let subtype = hit
            .source_subtype
            .clone()
            .unwrap_or_else(|| format!("{:?}", hit.backend_kind));
        let entry = map.entry(key.clone()).or_insert_with(|| NormalizedHit {
            key: key.clone(),
            hit: hit.clone(),
            providers: BTreeSet::new(),
            subtypes: BTreeSet::new(),
        });
        entry.providers.insert(provider.clone());
        entry.subtypes.insert(subtype);
        if better_text(&hit.title, &entry.hit.title) {
            entry.hit.title = hit.title.clone();
        }
        if better_text(&hit.snippet, &entry.hit.snippet) {
            entry.hit.snippet = hit.snippet.clone();
        }
        if entry.hit.published.is_none() {
            entry.hit.published = hit.published.take();
        }
        entry.hit.provider = entry
            .providers
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(",");
        entry.hit.source_subtype =
            Some(entry.subtypes.iter().cloned().collect::<Vec<_>>().join(","));
        entry
            .hit
            .metadata
            .insert("provider_labels".into(), entry.hit.provider.clone());
        entry.hit.metadata.insert(
            "source_subtypes".into(),
            entry.hit.source_subtype.clone().unwrap_or_default(),
        );
        entry.hit.metadata.insert(
            "cross_engine_agreement".into(),
            entry.providers.len().to_string(),
        );
    }
    map.into_values().collect()
}

fn normalize_provider_hit(mut hit: SearchHit) -> SearchHit {
    hit.url = normalize_url(&hit.url);
    hit
}

fn better_text(new: &str, old: &str) -> bool {
    (old.trim().is_empty() && !new.trim().is_empty())
        || (new.trim().len() > old.trim().len() && !new.trim().is_empty())
}

pub fn normalize_url(value: &str) -> String {
    let Ok(url) = Url::parse(value) else {
        return value.trim().trim_end_matches('/').to_ascii_lowercase();
    };
    let host = url
        .host_str()
        .unwrap_or_default()
        .trim_start_matches("www.")
        .to_ascii_lowercase();
    let port = url
        .port()
        .filter(|port| {
            !((*port == 80 && url.scheme() == "http") || (*port == 443 && url.scheme() == "https"))
        })
        .map(|p| format!(":{p}"))
        .unwrap_or_default();
    let path = url.path().trim_end_matches('/');
    let mut pairs = url
        .query_pairs()
        .filter(|(key, _)| {
            let key = key.to_ascii_lowercase();
            !(key.starts_with("utm_") || matches!(key.as_str(), "fbclid" | "gclid"))
        })
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    pairs.sort();
    let query = if pairs.is_empty() {
        String::new()
    } else {
        format!(
            "?{}",
            url::form_urlencoded::Serializer::new(String::new())
                .extend_pairs(pairs)
                .finish()
        )
    };
    format!("{host}{port}{path}{query}")
}

fn labels(item: &NormalizedHit, key: &str) -> Vec<String> {
    match key {
        "provider_labels" => item.providers.iter().cloned().collect(),
        _ => item.subtypes.iter().cloned().collect(),
    }
}
fn bound(value: &str) -> String {
    value.chars().take(MAX_FIELD_CHARS).collect()
}
fn safe_error(error: &ToolNetError) -> String {
    let message = match error {
        ToolNetError::Timeout => "provider timed out",
        ToolNetError::Network(_) => "provider network failure",
        ToolNetError::HttpStatus { status, .. } => {
            return format!("provider returned HTTP {status}");
        }
        ToolNetError::Parse(_) => "provider response parse failure",
        ToolNetError::BodyTooLarge { .. } => "provider response exceeded limit",
        ToolNetError::Content(_) => "provider returned unsupported content",
    };
    message.into()
}
fn safe_app_error(error: &AppError) -> String {
    match error {
        AppError::RankFailed(_) => "ranker failed".into(),
        _ => error.to_string().chars().take(128).collect(),
    }
}

fn cap_output(
    mut output: MetaSearchOutput,
    cap: usize,
) -> Result<MetaSearchOutput, MetaSearchError> {
    while serde_json::to_vec(&output)
        .map_err(|e| MetaSearchError::Output(e.to_string()))?
        .len()
        > cap
        && output.selected.len() > 1
    {
        output.selected.pop();
        output.ranked_evidence.pop();
    }
    truncate_output(&mut output, cap)?;
    if serde_json::to_vec(&output)
        .map_err(|e| MetaSearchError::Output(e.to_string()))?
        .len()
        > cap
    {
        output.selected.clear();
        output.ranked_evidence.clear();
        output.warnings.clear();
        output.query = "search output truncated".into();
        truncate_output(&mut output, cap)?;
    }
    Ok(output)
}

fn truncate_output(output: &mut MetaSearchOutput, cap: usize) -> Result<(), MetaSearchError> {
    loop {
        let size = serde_json::to_vec(output)
            .map_err(|e| MetaSearchError::Output(e.to_string()))?
            .len();
        if size <= cap {
            return Ok(());
        }
        let mut changed = false;
        for value in [&mut output.query]
            .into_iter()
            .chain(output.warnings.iter_mut())
        {
            if value.chars().count() > 8 {
                let n = value.chars().count() - 1;
                *value = value.chars().take(n).collect();
                changed = true;
                break;
            }
        }
        if !changed {
            for hit in &mut output.selected {
                for value in [&mut hit.title, &mut hit.url, &mut hit.snippet]
                    .into_iter()
                    .chain(hit.providers.iter_mut())
                    .chain(hit.source_subtypes.iter_mut())
                {
                    if value.chars().count() > 8 {
                        *value = value.chars().take(value.chars().count() - 1).collect();
                        changed = true;
                        break;
                    }
                }
                if changed {
                    break;
                }
            }
        }
        if !changed {
            for evidence in &mut output.ranked_evidence {
                for value in [&mut evidence.normalized_url, &mut evidence.decision] {
                    if value.chars().count() > 8 {
                        *value = value.chars().take(value.chars().count() - 1).collect();
                        changed = true;
                        break;
                    }
                }
                if changed {
                    break;
                }
            }
        }
        if !changed {
            return Err(MetaSearchError::Output(format!(
                "configured output cap {cap} is too small"
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::types::{BackendKind, SearchBackend};
    use crate::rank::{EvidenceEntry as RankEvidence, RankingResult};
    use async_trait::async_trait;
    use std::collections::BTreeMap;

    fn hit(url: &str, title: &str) -> SearchHit {
        SearchHit {
            title: title.into(),
            url: url.into(),
            snippet: "useful snippet".into(),
            published: None,
            native_rank: None,
            native_score: None,
            provider: "fixture".into(),
            backend_kind: BackendKind::Web,
            source_subtype: Some("web".into()),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn normalizes_tracking_parameters_and_deduplicates_labels() {
        assert_eq!(
            normalize_url("HTTPS://www.Example.test:443/a/?utm_source=x&b=2#frag&a=1"),
            "example.test/a?b=2"
        );
        let merged = deduplicate(vec![
            ("one".into(), hit("https://example.test/a?gclid=x", "short")),
            (
                "two".into(),
                hit("http://www.example.test/a/", "a better title"),
            ),
        ]);
        assert_eq!(merged.len(), 1);
        assert_eq!(
            merged[0].providers.iter().cloned().collect::<Vec<_>>(),
            ["one", "two"]
        );
        assert_eq!(merged[0].hit.title, "a better title");
    }

    #[test]
    fn capped_output_is_valid_and_drops_lowest_ranked_items() {
        let output = MetaSearchOutput {
            query: "q".into(),
            backends_queried: vec!["a".into()],
            selected: (0..5)
                .map(|n| SelectedHit {
                    title: format!("title {n}"),
                    url: format!("https://example.test/{n}"),
                    snippet: "x".repeat(300),
                    providers: vec!["provider".into()],
                    source_subtypes: vec!["web".into()],
                    score: Some(n as f64),
                })
                .collect(),
            warnings: vec![],
            ranked_evidence: (0..5)
                .map(|n| RankedEvidence {
                    normalized_url: format!("u{n}"),
                    decision: "d".into(),
                    score: Some(n as f64),
                })
                .collect(),
        };
        let result = cap_output(output, 1_000).unwrap();
        let json = serde_json::to_vec(&result).unwrap();
        assert!(json.len() <= 1_000);
        assert!(serde_json::from_slice::<serde_json::Value>(&json).is_ok());
        assert!(result.selected.len() < 5);
        assert_eq!(result.selected.first().unwrap().title, "title 0");
        assert!(!result.selected.iter().any(|hit| hit.title == "title 4"));
    }

    #[tokio::test]
    async fn activity_and_provenance_store_do_not_require_a_receiver() {
        let state = MetaSearchState::new();
        state.begin_turn("turn-1").await;
        state
            .publish(SearchActivity::QueryStarted { query: "q".into() })
            .await;
        state
            .publish(SearchActivity::ProviderResult {
                provider: "fixture".into(),
                elapsed_ms: 1,
                hits: 1,
                retry_count: None,
                normalized_hits: vec![normalize_provider_hit(hit(
                    "https://example.test/?utm_source=test",
                    "title",
                ))],
            })
            .await;
        assert_eq!(state.snapshot("turn-1").await.unwrap().turn_id, "turn-1");
        assert_eq!(state.activities().await.len(), 2);
        let activities = state.take_activities().await;
        assert!(matches!(
            &activities[1],
            SearchActivity::ProviderResult { normalized_hits, .. }
                if normalized_hits[0].title == "title"
        ));
        assert!(state.activities().await.is_empty());
        state.begin_turn("turn-2").await;
        assert!(state.activities().await.is_empty());
    }

    struct FakeBackend {
        name: &'static str,
        result: Result<Vec<SearchHit>, ToolNetError>,
    }

    #[async_trait]
    impl SearchBackend for FakeBackend {
        async fn search(&self, _query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
            self.result.clone()
        }
        fn name(&self) -> &'static str {
            self.name
        }
        fn category(&self) -> crate::contracts::types::Category {
            crate::contracts::types::Category::General
        }
    }

    struct FakeRanker;
    #[async_trait]
    impl Ranker for FakeRanker {
        async fn rank(&self, request: RankingRequest) -> Result<RankingResult, AppError> {
            let mut candidates = request.candidates;
            candidates.reverse();
            let decisions = candidates
                .iter()
                .enumerate()
                .map(|(n, c)| crate::contracts::session::RankingDecision {
                    source_id: c.candidate_id.clone(),
                    normalized_url: c.hit.url.clone(),
                    selected: n == 0,
                    decision: "relevant".into(),
                    score: Some(1.0 - n as f64 / 10.0),
                })
                .collect();
            Ok(RankingResult {
                decisions,
                evidence: Vec::<RankEvidence>::new(),
                candidates,
            })
        }
    }

    #[tokio::test]
    async fn reversed_ranker_order_preserves_candidate_mapping_and_provenance() {
        let mut registry = BackendRegistry::new();
        registry.add(Arc::new(FakeBackend {
            name: "alpha",
            result: Ok(vec![hit("https://example.test/alpha", "alpha title")]),
        }));
        registry.add(Arc::new(FakeBackend {
            name: "beta",
            result: Ok(vec![hit("https://example.test/beta", "beta title")]),
        }));
        let state = MetaSearchState::new();
        state.begin_turn("mapping-turn").await;
        let tool = MetaSearch::with_state(
            registry,
            Arc::new(FakeRanker),
            search_config(),
            state.clone(),
        );
        let output = tool
            .call(MetaSearchArgs {
                query: "rust".into(),
            })
            .await
            .unwrap();
        assert_eq!(output.selected.len(), 1);
        assert_eq!(output.selected[0].title, "beta title");
        assert_eq!(output.selected[0].url, "https://example.test/beta");
        assert_eq!(output.selected[0].providers, vec!["beta"]);
        assert_eq!(
            output.ranked_evidence[0].normalized_url,
            "example.test/beta"
        );
        let provenance = state.snapshot("mapping-turn").await.unwrap();
        assert_eq!(provenance.entries[0].title, "beta title");
        assert_eq!(provenance.entries[0].normalized_url, "example.test/beta");
        assert_eq!(
            provenance.entries[0].provider_labels,
            ["beta".to_owned()].into_iter().collect()
        );
        let ranking_activity = state
            .activities()
            .await
            .into_iter()
            .find_map(|activity| match activity {
                SearchActivity::RankingCompleted {
                    selected,
                    decisions,
                    ..
                } => Some((selected, decisions)),
                _ => None,
            })
            .unwrap();
        assert_eq!(ranking_activity.0, 1);
        assert_eq!(ranking_activity.1.len(), 2);
        assert_eq!(ranking_activity.1[0].source_id, "c1");
        assert_eq!(ranking_activity.1[1].source_id, "c0");
    }

    struct FailingRanker;
    #[async_trait]
    impl Ranker for FailingRanker {
        async fn rank(&self, _request: RankingRequest) -> Result<RankingResult, AppError> {
            Err(AppError::RankFailed("fixture rank failure".into()))
        }
    }

    fn search_config() -> SearchConfig {
        SearchConfig {
            stage_budget_secs: 1,
            rank_timeout_secs: 1,
            per_backend_hit_cap: 5,
            model_output_bytes: 6144,
        }
    }

    #[tokio::test]
    async fn partial_backend_failure_still_returns_and_records_warning() {
        let mut registry = BackendRegistry::new();
        registry.add(Arc::new(FakeBackend {
            name: "ok",
            result: Ok(vec![hit("https://example.test", "title")]),
        }));
        registry.add(Arc::new(FakeBackend {
            name: "bad",
            result: Err(ToolNetError::Network("offline".into())),
        }));
        let state = MetaSearchState::new();
        state.begin_turn("turn").await;
        let tool = MetaSearch::with_state(
            registry,
            Arc::new(FakeRanker),
            search_config(),
            state.clone(),
        );
        let output = tool
            .call(MetaSearchArgs {
                query: "rust".into(),
            })
            .await
            .unwrap();
        assert_eq!(output.selected.len(), 1);
        assert!(
            output
                .warnings
                .iter()
                .any(|warning| warning.contains("bad"))
        );
        assert_eq!(state.snapshot("turn").await.unwrap().entries.len(), 1);
    }

    #[tokio::test]
    async fn all_backend_failure_is_a_tool_error() {
        let mut registry = BackendRegistry::new();
        registry.add(Arc::new(FakeBackend {
            name: "bad",
            result: Err(ToolNetError::Timeout),
        }));
        let tool = MetaSearch::new(registry, Arc::new(FakeRanker), search_config());
        assert!(matches!(
            tool.call(MetaSearchArgs {
                query: "rust".into()
            })
            .await,
            Err(MetaSearchError::NoBackendsSucceeded)
        ));
    }

    #[tokio::test]
    async fn ranker_failure_is_not_replaced_with_search_order() {
        let mut registry = BackendRegistry::new();
        registry.add(Arc::new(FakeBackend {
            name: "ok",
            result: Ok(vec![hit("https://example.test", "title")]),
        }));
        let tool = MetaSearch::new(registry, Arc::new(FailingRanker), search_config());
        assert!(matches!(
            tool.call(MetaSearchArgs {
                query: "rust".into()
            })
            .await,
            Err(MetaSearchError::Ranking(AppError::RankFailed(_)))
        ));
    }
}
