use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct BraveBackend {
    http: BackendHttp,
    cap: usize,
    api_key: String,
}
#[derive(Deserialize)]
struct Response {
    web: Option<Web>,
}
#[derive(Deserialize)]
struct Web {
    #[serde(default)]
    results: Vec<ResultItem>,
}
#[derive(Deserialize)]
struct ResultItem {
    title: Option<String>,
    url: Option<String>,
    description: Option<String>,
    page_age: Option<String>,
}
impl BraveBackend {
    pub fn build(http: BackendHttp, api_key: String, cap: usize) -> Self {
        Self { http, cap, api_key }
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
        let key = secret
            .and_then(|s| s.api_key.clone())
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| AppError::Config("brave requires a resolved API key".into()))?;
        Ok(Some(Arc::new(Self::build(http, key, cap))))
    }
}
#[async_trait]
impl SearchBackend for BraveBackend {
    async fn search(&self, q: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let headers = [
            ("x-subscription-token", &self.api_key[..]),
            ("accept", "application/json"),
        ];
        let r: Response = self
            .http
            .get_json_with_headers(
                "https://api.search.brave.com/res/v1/web/search",
                &[
                    ("q", q),
                    ("count", &self.cap.to_string()),
                    ("country", "US"),
                    ("search_lang", "en"),
                    ("result_filter", "web"),
                ],
                &headers,
            )
            .await?;
        Ok(parse_response(r, self.cap))
    }
    fn name(&self) -> &'static str {
        "brave"
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
    r.web
        .map(|w| w.results)
        .unwrap_or_default()
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, x)| {
            Some(SearchHit {
                title: x.title?,
                url: x.url?,
                snippet: x.description.unwrap_or_default(),
                published: x.page_age,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "brave".into(),
                backend_kind: BackendKind::Api,
                source_subtype: Some("web".into()),
                metadata: Default::default(),
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_age_and_cap() {
        let h=parse_json(r#"{"web":{"results":[{"title":"T","url":"https://e","description":"D","page_age":"today"},{"title":"2","url":"https://2"}]}}"#,1).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].published.as_deref(), Some("today"));
    }
}
