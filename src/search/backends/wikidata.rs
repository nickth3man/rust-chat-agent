use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct WikidataBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize, Default)]
struct Response {
    #[serde(default)]
    search: Vec<Item>,
}
#[derive(Deserialize)]
struct Item {
    id: String,
    label: Option<String>,
    description: Option<String>,
    concepturi: Option<String>,
}

impl WikidataBackend {
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
impl SearchBackend for WikidataBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let cap = self.cap.to_string();
        let response: Response = self
            .http
            .get_json(
                "https://www.wikidata.org/w/api.php",
                &[
                    ("action", "wbsearchentities"),
                    ("search", query),
                    ("language", "en"),
                    ("uselang", "en"),
                    ("format", "json"),
                    ("limit", &cap),
                ],
            )
            .await?;
        Ok(parse_response(response, self.cap))
    }
    fn name(&self) -> &'static str {
        "wikidata"
    }
    fn category(&self) -> Category {
        Category::Reference
    }
}
fn parse_response(response: Response, cap: usize) -> Vec<SearchHit> {
    response
        .search
        .into_iter()
        .take(cap)
        .enumerate()
        .map(|(i, item)| {
            let url = item
                .concepturi
                .clone()
                .unwrap_or_else(|| format!("https://www.wikidata.org/entity/{}", item.id));
            let mut metadata = std::collections::BTreeMap::new();
            metadata.insert("entity_id".into(), item.id);
            SearchHit {
                title: item.label.unwrap_or_else(|| url.clone()),
                url,
                snippet: item.description.unwrap_or_default(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "wikidata".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("entity-search".into()),
                metadata,
            }
        })
        .collect()
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let response: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_response(response, cap))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_entity_fixture() {
        let h = parse_json(r#"{"search":[{"id":"Q42","label":"Douglas Adams","description":"English writer","concepturi":"http://www.wikidata.org/entity/Q42"}]}"#, 1).unwrap();
        assert_eq!(h[0].snippet, "English writer");
        assert_eq!(h[0].metadata["entity_id"], "Q42");
    }
}
