use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct MarginaliaBackend {
    http: BackendHttp,
    cap: usize,
}

#[derive(Deserialize)]
struct MarginaliaResponse {
    #[serde(default)]
    results: Vec<MarginaliaResult>,
}

#[derive(Deserialize)]
struct MarginaliaResult {
    url: Option<String>,
    title: Option<String>,
    description: Option<String>,
}

impl MarginaliaBackend {
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
impl SearchBackend for MarginaliaBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let count = self.cap.to_string();
        let response: MarginaliaResponse = self
            .http
            .get_json_with_headers(
                "https://api2.marginalia-search.com/search",
                &[("query", query), ("count", &count)],
                &[("API-Key", "public")],
            )
            .await?;
        Ok(parse_response(response, self.cap))
    }

    fn name(&self) -> &'static str {
        "marginalia"
    }
    fn category(&self) -> Category {
        Category::Web
    }
}

fn parse_response(response: MarginaliaResponse, cap: usize) -> Vec<SearchHit> {
    response
        .results
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, result)| {
            let url = result.url?;
            Some(SearchHit {
                title: result.title.unwrap_or_else(|| url.clone()),
                url,
                snippet: result.description.unwrap_or_default(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "marginalia".into(),
                backend_kind: BackendKind::Api,
                source_subtype: Some("search".into()),
                metadata: Default::default(),
            })
        })
        .collect()
}

pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let response: MarginaliaResponse =
        serde_json::from_str(input).map_err(|error| ToolNetError::Parse(error.to_string()))?;
    Ok(parse_response(response, cap))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_caps_fixture() {
        let input = r#"{"results":[{"url":"https://one.test","title":"One","description":"First"},{"url":"https://two.test","title":"Two","description":"Second"}]}"#;
        let hits = parse_json(input, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "One");
        assert_eq!(hits[0].native_rank, Some(1));
        assert_eq!(hits[0].backend_kind, BackendKind::Api);
    }
}
