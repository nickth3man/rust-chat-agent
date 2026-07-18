use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct OpenLibraryBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize, Default)]
struct Response {
    #[serde(default)]
    docs: Vec<Doc>,
}
#[derive(Deserialize)]
struct Doc {
    title: Option<String>,
    key: Option<String>,
    author_name: Option<Vec<String>>,
    first_publish_year: Option<i32>,
    edition_key: Option<Vec<String>>,
}
impl OpenLibraryBackend {
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
impl SearchBackend for OpenLibraryBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let limit = self.cap.to_string();
        let response: Response = self
            .http
            .get_json(
                "https://openlibrary.org/search.json",
                &[("q", query), ("limit", &limit), ("mode", "everything")],
            )
            .await?;
        Ok(parse_response(response, self.cap))
    }
    fn name(&self) -> &'static str {
        "openlibrary"
    }
    fn category(&self) -> Category {
        Category::Library
    }
}
fn parse_response(response: Response, cap: usize) -> Vec<SearchHit> {
    response
        .docs
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, d)| {
            let key = d
                .key
                .or_else(|| d.edition_key.and_then(|v| v.into_iter().next()))?;
            let url = if key.starts_with("http") {
                key.clone()
            } else {
                format!("https://openlibrary.org/{}", key.trim_start_matches('/'))
            };
            let title = d.title.unwrap_or(key);
            let authors = d.author_name.unwrap_or_default();
            let mut metadata = std::collections::BTreeMap::new();
            if !authors.is_empty() {
                metadata.insert("authors".into(), authors.join(", "));
            }
            if let Some(year) = d.first_publish_year {
                metadata.insert("first_publish_year".into(), year.to_string());
            }
            Some(SearchHit {
                title,
                url,
                snippet: metadata.get("authors").cloned().unwrap_or_default(),
                published: d.first_publish_year.map(|y| y.to_string()),
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "openlibrary".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("search".into()),
                metadata,
            })
        })
        .collect()
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let r: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_response(r, cap))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_book_fixture() {
        let h = parse_json(r#"{"docs":[{"title":"The Hitchhiker's Guide","key":"/works/OL1W","author_name":["Douglas Adams"],"first_publish_year":1979}]}"#, 1).unwrap();
        assert_eq!(h[0].url, "https://openlibrary.org/works/OL1W");
        assert_eq!(h[0].published.as_deref(), Some("1979"));
    }
}
