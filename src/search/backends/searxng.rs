use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct SearxngBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    results: Vec<Item>,
}
#[derive(Deserialize)]
struct Item {
    url: Option<String>,
    title: Option<String>,
    content: Option<String>,
    engines: Option<Vec<String>>,
    category: Option<String>,
    score: Option<f64>,
    #[serde(rename = "publishedDate")]
    published_date: Option<String>,
}
impl SearxngBackend {
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
        if secret
            .and_then(|s| s.base_url.as_ref())
            .or(cfg.base_url.as_ref())
            .is_none()
        {
            return Err(AppError::Config(
                "searxng requires a resolved base URL".into(),
            ));
        }
        let _ = secret;
        Ok(Some(Arc::new(Self::build(http, cap))))
    }
}
#[async_trait]
impl SearchBackend for SearxngBackend {
    async fn search(&self, q: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let r: Response = self
            .http
            .get_json(
                &self.http.url("search"),
                &[
                    ("q", q),
                    ("format", "json"),
                    ("pageno", "1"),
                    ("categories", "general"),
                ],
            )
            .await?;
        Ok(parse_response(r, self.cap))
    }
    fn name(&self) -> &'static str {
        "searxng"
    }
    fn category(&self) -> Category {
        Category::Web
    }
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let r: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_response(r, cap))
}
fn parse_response(r: Response, cap: usize) -> Vec<SearchHit> {
    r.results
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, x)| {
            let url = x.url?;
            let mut metadata = std::collections::BTreeMap::new();
            if let Some(v) = x.engines {
                metadata.insert("engines".into(), v.join(","));
            }
            if let Some(v) = x.category {
                metadata.insert("category".into(), v);
            }
            Some(SearchHit {
                title: x.title.unwrap_or_else(|| url.clone()),
                url,
                snippet: x.content.unwrap_or_default(),
                published: x.published_date,
                native_rank: Some((i + 1) as u32),
                native_score: x.score,
                provider: "searxng".into(),
                backend_kind: BackendKind::Api,
                source_subtype: Some("general".into()),
                metadata,
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_labels_and_cap() {
        let h=parse_json(r#"{"results":[{"url":"https://e","title":"T","content":"C","engines":["google","bing"],"category":"general","score":1.2,"publishedDate":"2026"},{"url":"https://2","title":"2"}]}"#,1).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].metadata["engines"], "google,bing");
    }
}
