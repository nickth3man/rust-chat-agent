use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct StackexchangeBackend {
    http: BackendHttp,
    cap: usize,
    key: Option<String>,
}
pub type StackExchangeBackend = StackexchangeBackend;
#[derive(Deserialize)]
struct Response {
    items: Vec<Item>,
}
#[derive(Deserialize)]
struct Item {
    title: String,
    link: String,
    score: i64,
    is_answered: bool,
}

impl StackexchangeBackend {
    pub fn build(http: BackendHttp, cap: usize, key: Option<String>) -> Self {
        Self { http, cap, key }
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
    fn query<'a>(&'a self, query: &'a str, page_size: &'a str) -> Vec<(&'a str, &'a str)> {
        let mut q = vec![
            ("q", query),
            ("site", "stackoverflow"),
            ("sort", "relevance"),
            ("pagesize", page_size),
        ];
        if let Some(key) = self.key.as_deref() {
            q.push(("key", key));
        }
        q
    }
}

#[async_trait]
impl SearchBackend for StackexchangeBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let cap = self.cap.to_string();
        let response: Response = self
            .http
            .get_json(
                "https://api.stackexchange.com/2.3/search/advanced",
                &self.query(query, &cap),
            )
            .await?;
        Ok(parse_response(response, self.cap))
    }
    fn name(&self) -> &'static str {
        "stackexchange"
    }
    fn category(&self) -> Category {
        Category::Reference
    }
}

pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let response: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_response(response, cap))
}
fn parse_response(response: Response, cap: usize) -> Vec<SearchHit> {
    response
        .items
        .into_iter()
        .take(cap)
        .enumerate()
        .map(|(i, item)| {
            let mut metadata = BTreeMap::new();
            metadata.insert("is_answered".into(), item.is_answered.to_string());
            SearchHit {
                title: clean_text(&item.title),
                url: item.link,
                snippet: String::new(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: Some(item.score as f64),
                provider: "stackexchange".into(),
                backend_kind: BackendKind::Api,
                source_subtype: Some("stackoverflow".into()),
                metadata,
            }
        })
        .collect()
}

/// Conservative decoding without adding an HTML entity dependency: remove tags and
/// decode the entities commonly emitted by Stack Exchange titles.
pub fn clean_text(input: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in input.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_answer_and_decodes_title() {
        let h = parse_json(r#"{"items":[{"title":"A &amp; B","link":"https://stackoverflow.com/q/1","score":3,"is_answered":true}]}"#, 2).unwrap();
        assert_eq!(h[0].title, "A & B");
        assert_eq!(h[0].metadata["is_answered"], "true");
    }
    #[test]
    fn strips_markup() {
        assert_eq!(clean_text("<b>A</b> &lt;x&gt;"), "A <x>");
    }
}
