use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct NpmBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize)]
struct Response {
    objects: Vec<Object>,
}
#[derive(Deserialize)]
struct Object {
    package: Package,
}
#[derive(Deserialize)]
struct Package {
    name: String,
    description: Option<String>,
    links: Option<Links>,
    version: Option<String>,
}
#[derive(Deserialize)]
struct Links {
    npm: Option<String>,
}

impl NpmBackend {
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
impl SearchBackend for NpmBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let size = self.cap.to_string();
        let response: Response = self
            .http
            .get_json(
                "https://registry.npmjs.org/-/v1/search",
                &[("text", query), ("size", &size)],
            )
            .await?;
        Ok(parse_response(response, self.cap))
    }
    fn name(&self) -> &'static str {
        "npm"
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
        .objects
        .into_iter()
        .take(cap)
        .enumerate()
        .map(|(i, object)| {
            let p = object.package;
            let url = p
                .links
                .as_ref()
                .and_then(|l| l.npm.clone())
                .unwrap_or_else(|| format!("https://www.npmjs.com/package/{}", p.name));
            let mut metadata = BTreeMap::new();
            if let Some(version) = p.version {
                metadata.insert("version".into(), version);
            }
            SearchHit {
                title: p.name,
                url,
                snippet: p.description.unwrap_or_default(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "npm".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("registry".into()),
                metadata,
            }
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_package_shape() {
        let h = parse_json(r#"{"objects":[{"package":{"name":"foo","description":"bar","links":{"npm":"https://npmjs.com/package/foo"},"version":"1.2.3"}}]}"#, 1).unwrap();
        assert_eq!(h[0].title, "foo");
        assert_eq!(h[0].metadata["version"], "1.2.3");
    }
}
