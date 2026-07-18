use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct CratesIoBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize)]
struct Response {
    crates: Vec<CrateItem>,
}
#[derive(Deserialize)]
struct CrateItem {
    name: String,
    description: Option<String>,
    repository: Option<String>,
    id: String,
    downloads: u64,
}

impl CratesIoBackend {
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
impl SearchBackend for CratesIoBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let per_page = self.cap.to_string();
        let response: Response = self
            .http
            .get_json(
                "https://crates.io/api/v1/crates",
                &[("q", query), ("per_page", &per_page)],
            )
            .await?;
        Ok(parse_response(response, self.cap))
    }
    fn name(&self) -> &'static str {
        "crates_io"
    }
    fn category(&self) -> Category {
        Category::Library
    }
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let response: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_response(response, cap))
}
fn parse_response(response: Response, cap: usize) -> Vec<SearchHit> {
    response
        .crates
        .into_iter()
        .take(cap)
        .enumerate()
        .map(|(i, item)| {
            let url = item
                .repository
                .clone()
                .unwrap_or_else(|| format!("https://crates.io/crates/{}", item.name));
            let mut metadata = BTreeMap::new();
            metadata.insert("id".into(), item.id);
            metadata.insert("downloads".into(), item.downloads.to_string());
            if let Some(repository) = item.repository {
                metadata.insert("repository".into(), repository);
            }
            SearchHit {
                title: item.name,
                url,
                snippet: item.description.unwrap_or_default(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: Some(item.downloads as f64),
                provider: "crates_io".into(),
                backend_kind: BackendKind::Api,
                source_subtype: Some("crates".into()),
                metadata,
            }
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_crate_fields() {
        let h = parse_json(r#"{"crates":[{"name":"serde","description":"data","repository":"https://github.com/serde-rs/serde","id":"serde","downloads":99}]}"#, 1).unwrap();
        assert_eq!(h[0].metadata["downloads"], "99");
        assert_eq!(h[0].url, "https://github.com/serde-rs/serde");
    }
}
