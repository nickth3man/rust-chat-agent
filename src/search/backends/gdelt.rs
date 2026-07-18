use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct GdeltBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    articles: Vec<Article>,
}
#[derive(Deserialize)]
struct Article {
    url: Option<String>,
    title: Option<String>,
    seendate: Option<String>,
    domain: Option<String>,
    language: Option<String>,
    sourcecountry: Option<String>,
}
impl GdeltBackend {
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
    async fn search_impl(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let response: Response = self
            .http
            .get_json(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                &[
                    ("query", query),
                    ("mode", "artlist"),
                    ("format", "json"),
                    ("maxrecords", &self.cap.to_string()),
                ],
            )
            .await?;
        Ok(parse_articles(response.articles, self.cap))
    }
}
#[async_trait]
impl SearchBackend for GdeltBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        self.search_impl(query).await
    }
    fn name(&self) -> &'static str {
        "gdelt"
    }
    fn category(&self) -> Category {
        Category::News
    }
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let response: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_articles(response.articles, cap))
}
fn parse_articles(articles: Vec<Article>, cap: usize) -> Vec<SearchHit> {
    articles
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, a)| {
            let url = a.url?;
            let title = a.title.unwrap_or_else(|| url.clone());
            let mut metadata = std::collections::BTreeMap::new();
            if let Some(v) = a.domain {
                metadata.insert("domain".into(), v);
            }
            if let Some(v) = a.language {
                metadata.insert("language".into(), v);
            }
            if let Some(v) = a.sourcecountry {
                metadata.insert("sourcecountry".into(), v);
            }
            Some(SearchHit {
                title,
                url,
                snippet: String::new(),
                published: a.seendate,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "gdelt".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("article".into()),
                metadata,
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_cap_and_metadata() {
        let h = parse_json(r#"{"articles":[{"url":"https://e","title":"T","seendate":"20260717","domain":"e","language":"English","sourcecountry":"US"},{"url":"https://two"}]}"#, 1).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].metadata["domain"], "e");
    }
}
