use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct GithubBackend {
    http: BackendHttp,
    cap: usize,
    token: Option<String>,
}
pub type GitHubBackend = GithubBackend;

#[derive(Deserialize)]
struct Response {
    items: Vec<Repository>,
}
#[derive(Deserialize)]
struct Repository {
    full_name: String,
    html_url: String,
    description: Option<String>,
    stargazers_count: u64,
    updated_at: Option<String>,
}

impl GithubBackend {
    pub fn build(http: BackendHttp, cap: usize, token: Option<String>) -> Self {
        Self { http, cap, token }
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
        Ok(Some(Arc::new(Self::build(
            http,
            cap,
            secret.and_then(|s| s.api_key.clone()),
        ))))
    }
}

/// GitHub requires its vendor media type even when no token is supplied.
pub fn request_headers(token: Option<&str>) -> Vec<(String, String)> {
    let mut headers = vec![("accept".into(), "application/vnd.github+json".into())];
    if let Some(token) = token.filter(|v| !v.trim().is_empty()) {
        headers.push(("authorization".into(), format!("Bearer {token}")));
    }
    headers
}

#[async_trait]
impl SearchBackend for GithubBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let per_page = self.cap.to_string();
        let headers = request_headers(self.token.as_deref());
        let refs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let response: Response = self
            .http
            .get_json_with_headers(
                "https://api.github.com/search/repositories",
                &[("q", query), ("per_page", &per_page)],
                &refs,
            )
            .await?;
        Ok(parse_json_response(response, self.cap))
    }
    fn name(&self) -> &'static str {
        "github"
    }
    fn category(&self) -> Category {
        Category::Code
    }
}

pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let response: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_json_response(response, cap))
}

fn parse_json_response(response: Response, cap: usize) -> Vec<SearchHit> {
    response
        .items
        .into_iter()
        .take(cap)
        .enumerate()
        .map(|(i, item)| {
            let mut metadata = BTreeMap::new();
            metadata.insert("full_name".into(), item.full_name.clone());
            metadata.insert("stars".into(), item.stargazers_count.to_string());
            SearchHit {
                title: item.full_name,
                url: item.html_url,
                snippet: item.description.unwrap_or_default(),
                published: item.updated_at,
                native_rank: Some((i + 1) as u32),
                native_score: Some(item.stargazers_count as f64),
                provider: "github".into(),
                backend_kind: BackendKind::Api,
                source_subtype: Some("repositories".into()),
                metadata,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_repository_and_cap() {
        let json = r#"{"items":[{"full_name":"acme/tool","html_url":"https://github.com/acme/tool","description":"A tool","stargazers_count":7,"updated_at":"2024-01-01"}]}"#;
        let hits = parse_json(json, 1).unwrap();
        assert_eq!(hits[0].title, "acme/tool");
        assert_eq!(hits[0].native_score, Some(7.0));
    }
    #[test]
    fn auth_is_explicit() {
        assert_eq!(
            request_headers(Some("secret"))[1],
            ("authorization".into(), "Bearer secret".into())
        );
        assert_eq!(request_headers(None).len(), 1);
    }
}
