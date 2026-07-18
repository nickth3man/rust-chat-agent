use super::provenance::TurnProvenance;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub format: String,
    pub version: u32,
    pub session_id: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub role: TranscriptRole,
    pub content: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultSummary {
    pub title: String,
    pub snippet: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultSummary {
    pub hit_count: usize,
    #[serde(default)]
    pub results: Vec<ResultSummary>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderState {
    Started,
    Succeeded,
    Partial,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingDecision {
    pub source_id: String,
    pub normalized_url: String,
    pub selected: bool,
    pub decision: String,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum LogicalEvent {
    User {
        timestamp: String,
        content: String,
    },
    Assistant {
        timestamp: String,
        content: String,
    },
    Tool {
        timestamp: String,
        name: String,
        arguments: Value,
        elapsed_ms: u64,
        retry_count: u32,
        error: Option<String>,
        result: Option<ToolResultSummary>,
    },
    Provider {
        timestamp: String,
        provider: String,
        state: ProviderState,
        elapsed_ms: u64,
        retry_count: u32,
        error: Option<String>,
        hit_count: usize,
    },
    Ranking {
        timestamp: String,
        query: String,
        elapsed_ms: u64,
        decisions: Vec<RankingDecision>,
        error: Option<String>,
    },
    Compaction {
        timestamp: String,
        reason: String,
        removed_entries: usize,
        summary: Option<String>,
        error: Option<String>,
    },
    StreamFailure {
        timestamp: String,
        elapsed_ms: u64,
        retry_count: u32,
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDocument {
    pub metadata: SessionMetadata,
    #[serde(default)]
    pub transcript: Vec<TranscriptEntry>,
    #[serde(default)]
    pub events: Vec<LogicalEvent>,
    #[serde(default)]
    pub provenance: Vec<TurnProvenance>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_event_serializes_structured_activity() {
        let event = LogicalEvent::Tool {
            timestamp: "2026-01-01T00:00:00Z".into(),
            name: "meta_search".into(),
            arguments: serde_json::json!({"query": "rust"}),
            elapsed_ms: 42,
            retry_count: 1,
            error: None,
            result: Some(ToolResultSummary {
                hit_count: 1,
                results: vec![ResultSummary {
                    title: "Rust".into(),
                    snippet: "A language".into(),
                    url: "https://www.rust-lang.org/".into(),
                }],
            }),
        };
        let json = serde_json::to_string(&event).unwrap();
        for field in [
            "timestamp",
            "arguments",
            "elapsed_ms",
            "retry_count",
            "hit_count",
            "url",
        ] {
            assert!(json.contains(field), "missing structured field {field}");
        }
    }
}
