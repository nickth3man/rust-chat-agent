use super::error::ToolNetError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    General,
    Web,
    News,
    Academic,
    Code,
    Reference,
    Library,
    Dictionary,
    Social,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Api,
    Json,
    Html,
    Xml,
    Web,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub published: Option<String>,
    pub native_rank: Option<u32>,
    pub native_score: Option<f64>,
    pub provider: String,
    pub backend_kind: BackendKind,
    pub source_subtype: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[async_trait]
pub trait SearchBackend: Send + Sync {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError>;
    fn name(&self) -> &'static str;
    fn category(&self) -> Category;
}
