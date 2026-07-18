use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone)]
pub struct FirecrawlBackend {
    http: BackendHttp,
    cap: usize,
    api_key: String,
    endpoint: String,
}
#[derive(Serialize)]
struct Request<'a> {
    query: &'a str,
    limit: usize,
    sources: [&'static str; 1],
}
#[derive(Deserialize)]
struct Response {
    data: Option<Data>,
    warning: Option<String>,
    id: Option<String>,
    #[serde(rename = "creditsUsed")]
    credits_used: Option<serde_json::Value>,
}
#[derive(Deserialize)]
struct Data {
    #[serde(default)]
    web: Vec<Web>,
}
#[derive(Deserialize)]
struct Web {
    url: Option<String>,
    title: Option<String>,
    description: Option<String>,
}
impl FirecrawlBackend {
    pub fn build(http: BackendHttp, api_key: String, cap: usize) -> Self {
        let endpoint = normalize_endpoint(&http.url(""));
        Self {
            http,
            cap,
            api_key,
            endpoint,
        }
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
            .ok_or_else(|| AppError::Config("firecrawl requires a resolved API key".into()))?;
        if secret
            .and_then(|s| s.base_url.as_ref())
            .or(cfg.base_url.as_ref())
            .is_none()
        {
            return Err(AppError::Config(
                "firecrawl requires a resolved base URL".into(),
            ));
        }
        Ok(Some(Arc::new(Self::build(http, key, cap))))
    }
}
#[async_trait]
impl SearchBackend for FirecrawlBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let body = Request {
            query,
            limit: 5,
            sources: ["web"],
        };
        let headers = [
            ("authorization", &format!("Bearer {}", self.api_key)[..]),
            ("accept", "application/json"),
        ];
        let response: Response = self
            .http
            .post_json_with_headers(&self.endpoint, &body, &headers)
            .await?;
        Ok(parse_response(response, self.cap))
    }
    fn name(&self) -> &'static str {
        "firecrawl"
    }
    fn category(&self) -> Category {
        Category::Web
    }
}
fn normalize_endpoint(endpoint: &str) -> String {
    endpoint.trim_end_matches('/').to_owned()
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let r: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_response(r, cap))
}
fn parse_response(r: Response, cap: usize) -> Vec<SearchHit> {
    let mut common = std::collections::BTreeMap::new();
    if let Some(v) = r.warning {
        common.insert("warning".into(), v);
    }
    if let Some(v) = r.id {
        common.insert("id".into(), v);
    }
    if let Some(v) = r.credits_used {
        common.insert("creditsUsed".into(), v.to_string());
    }
    r.data
        .map(|d| d.web)
        .unwrap_or_default()
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, w)| {
            Some(SearchHit {
                title: w.title?,
                url: w.url?,
                snippet: w.description.unwrap_or_default(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "firecrawl".into(),
                backend_kind: BackendKind::Api,
                source_subtype: Some("web".into()),
                metadata: common.clone(),
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_optional_metadata_and_cap() {
        let h=parse_json(r#"{"data":{"web":[{"url":"https://e","title":"T","description":"D"},{"url":"https://2","title":"2"}]},"warning":"w","id":"i","creditsUsed":2}"#,1).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].metadata["warning"], "w");
    }
    #[test]
    fn endpoint_normalization_is_used_by_request_builder() {
        for configured in [
            "https://api.firecrawl.dev/v2/search",
            "https://api.firecrawl.dev/v2/search/",
        ] {
            let cfg = ProviderConfig {
                enable: true,
                api_key_env: None,
                optional_api_key_env: None,
                concurrency: 1,
                min_interval_ms: 0,
                timeout_secs: 1,
                user_agent: "test".into(),
                base_url: Some(configured.into()),
                base_url_env: None,
            };
            let http = BackendHttp::new("firecrawl", &cfg, None).unwrap();
            let backend = FirecrawlBackend::build(http, "secret".into(), 5);
            assert_eq!(backend.endpoint, "https://api.firecrawl.dev/v2/search");
        }
    }
}
