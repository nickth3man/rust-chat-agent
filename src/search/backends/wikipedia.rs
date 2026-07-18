use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;
use url::Url;

#[derive(Clone)]
pub struct WikipediaBackend {
    http: BackendHttp,
    cap: usize,
}

#[derive(Deserialize)]
struct Response {
    query: Option<Query>,
}
#[derive(Deserialize)]
struct Query {
    #[serde(default)]
    search: Vec<Item>,
}
#[derive(Deserialize)]
struct Item {
    title: String,
    snippet: Option<String>,
    #[serde(rename = "pageid")]
    page_id: Option<u64>,
}

impl WikipediaBackend {
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
    async fn request(&self, query: &str) -> Result<Response, ToolNetError> {
        let cap = self.cap.to_string();
        self.http
            .get_json(
                "https://en.wikipedia.org/w/api.php",
                &[
                    ("action", "query"),
                    ("list", "search"),
                    ("format", "json"),
                    ("srsearch", query),
                    ("srlimit", &cap),
                    ("utf8", "1"),
                ],
            )
            .await
    }
}

#[async_trait]
impl SearchBackend for WikipediaBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        Ok(parse_json_response(self.request(query).await?, self.cap))
    }
    fn name(&self) -> &'static str {
        "wikipedia"
    }
    fn category(&self) -> Category {
        Category::Reference
    }
}

fn canonical_url(title: &str) -> String {
    let mut url = Url::parse("https://en.wikipedia.org/wiki").expect("static URL");
    url.path_segments_mut()
        .expect("URL is hierarchical")
        .push(&title.replace(' ', "_"));
    url.to_string()
}

fn strip_markup(value: &str) -> String {
    scraper::Html::parse_fragment(value)
        .root_element()
        .text()
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_json_response(response: Response, cap: usize) -> Vec<SearchHit> {
    response
        .query
        .map(|q| q.search)
        .unwrap_or_default()
        .into_iter()
        .take(cap)
        .enumerate()
        .map(|(i, item)| SearchHit {
            title: item.title.clone(),
            url: canonical_url(&item.title),
            snippet: strip_markup(item.snippet.as_deref().unwrap_or_default()),
            published: None,
            native_rank: Some((i + 1) as u32),
            native_score: None,
            provider: "wikipedia".into(),
            backend_kind: BackendKind::Json,
            source_subtype: Some("mediawiki".into()),
            metadata: item
                .page_id
                .map(|id| [("page_id".into(), id.to_string())].into_iter().collect())
                .unwrap_or_default(),
        })
        .collect()
}

pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let response: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_json_response(response, cap))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_fixture_and_strips_markup() {
        let hits = parse_json(r#"{"query":{"search":[{"title":"Rust (programming language)","snippet":"A <span class=\"searchmatch\">systems</span> language","pageid":10}]}}"#, 5).unwrap();
        assert_eq!(
            hits[0].url,
            "https://en.wikipedia.org/wiki/Rust_(programming_language)"
        );
        assert_eq!(hits[0].snippet, "A systems language");
    }

    #[test]
    fn collapses_unicode_whitespace_without_losing_word_boundaries() {
        assert_eq!(
            strip_markup(" A\u{2003}\n systems\t language "),
            "A systems language"
        );
    }
}
