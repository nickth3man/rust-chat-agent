use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct HnBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize)]
struct HnResponse {
    hits: Vec<HnHit>,
}
#[derive(Deserialize)]
struct HnHit {
    title: Option<String>,
    story_title: Option<String>,
    url: Option<String>,
    story_url: Option<String>,
    author: Option<String>,
    points: Option<i64>,
    created_at: Option<String>,
    #[serde(rename = "objectID")]
    object_id: Option<String>,
}
impl HnBackend {
    pub fn build(http: BackendHttp, cap: usize) -> Self {
        Self { http, cap }
    }
    pub fn from_config(
        cfg: &ProviderConfig,
        http: BackendHttp,
        secret: Option<&ProviderSecret>,
        cap: usize,
    ) -> Result<Option<Arc<dyn SearchBackend>>, AppError> {
        if !cfg.enable {
            return Ok(None);
        }
        let _ = secret;
        Ok(Some(Arc::new(Self::build(http, cap))))
    }
}
#[async_trait]
impl SearchBackend for HnBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let cap = self.cap.to_string();
        let response: HnResponse = self
            .http
            .get_json(
                "https://hn.algolia.com/api/v1/search",
                &[("query", query), ("hitsPerPage", &cap)],
            )
            .await?;
        Ok(parse_hits(response, self.cap))
    }
    fn name(&self) -> &'static str {
        "hn"
    }
    fn category(&self) -> Category {
        Category::News
    }
}

fn parse_hits(response: HnResponse, cap: usize) -> Vec<SearchHit> {
    response
        .hits
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, hit)| {
            let url = hit.url.or(hit.story_url)?;
            let title = hit.title.or(hit.story_title).unwrap_or_else(|| url.clone());
            let mut metadata = BTreeMap::new();
            if let Some(object_id) = hit.object_id {
                metadata.insert("object_id".into(), object_id.clone());
                metadata.insert(
                    "discussion_url".into(),
                    format!("https://news.ycombinator.com/item?id={object_id}"),
                );
            }
            Some(SearchHit {
                title,
                url,
                snippet: hit
                    .author
                    .map(|a| format!("Hacker News by {a}"))
                    .unwrap_or_default(),
                published: hit.created_at,
                native_rank: Some((i + 1) as u32),
                native_score: hit.points.map(|n| n as f64),
                provider: "hn".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("algolia".into()),
                metadata,
            })
        })
        .collect()
}

pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let response: HnResponse =
        serde_json::from_str(input).map_err(|error| ToolNetError::Parse(error.to_string()))?;
    Ok(parse_hits(response, cap))
}
