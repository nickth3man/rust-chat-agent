use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct SearchMySiteBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    results: Vec<ResultItem>,
}
#[derive(Deserialize)]
struct ResultItem {
    url: Option<String>,
    title: Option<String>,
    highlight: Option<String>,
}
impl SearchMySiteBackend {
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
impl SearchBackend for SearchMySiteBackend {
    async fn search(&self, q: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let r: Response = self
            .http
            .get_json("https://searchmysite.net/api/v1/search/", &[("q", q)])
            .await?;
        Ok(parse_results(r.results, self.cap))
    }
    fn name(&self) -> &'static str {
        "searchmysite"
    }
    fn category(&self) -> Category {
        Category::Web
    }
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let r: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_results(r.results, cap))
}
fn parse_results(items: Vec<ResultItem>, cap: usize) -> Vec<SearchHit> {
    items
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, r)| {
            let url = r.url?;
            let title = r.title.unwrap_or_else(|| url.clone());
            Some(SearchHit {
                title,
                url,
                snippet: r.highlight.unwrap_or_default(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "searchmysite".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("indie-web".into()),
                metadata: Default::default(),
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_highlight_fallback_and_cap() {
        let h=parse_json(r#"{"results":[{"url":"https://one","title":"One","highlight":"<b>one</b>"},{"url":"https://two"}]}"#,1).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].title, "One");
        assert_eq!(h[0].snippet, "<b>one</b>");
    }
}
