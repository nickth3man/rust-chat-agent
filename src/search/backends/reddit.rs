use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct RedditBackend {
    http: BackendHttp,
    cap: usize,
}
impl RedditBackend {
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
impl SearchBackend for RedditBackend {
    async fn search(&self, q: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let v: Value = self
            .http
            .get_json(
                "https://www.reddit.com/search.json",
                &[
                    ("q", q),
                    ("limit", &self.cap.to_string()),
                    ("sort", "relevance"),
                ],
            )
            .await?;
        parse_value(&v, self.cap)
    }
    fn name(&self) -> &'static str {
        "reddit"
    }
    fn category(&self) -> Category {
        Category::Social
    }
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let v: Value = serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    parse_value(&v, cap)
}
fn parse_value(v: &Value, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    Ok(v.pointer("/data/children")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(cap)
        .enumerate()
        .filter_map(|(i, c)| {
            let d = c.get("data")?;
            let permalink = d.get("permalink").and_then(Value::as_str);
            let url = d.get("url").and_then(Value::as_str).or(permalink)?;
            let url = if url.starts_with('/') {
                format!("https://www.reddit.com{url}")
            } else {
                url.to_owned()
            };
            let title = d
                .get("title")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .unwrap_or(&url)
                .to_owned();
            let mut metadata = BTreeMap::new();
            if let Some(p) = permalink {
                metadata.insert("permalink".into(), p.into());
            }
            if let Some(s) = d.get("subreddit").and_then(Value::as_str) {
                metadata.insert("subreddit".into(), s.into());
            }
            Some(SearchHit {
                title,
                url,
                snippet: d
                    .get("selftext")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .or_else(|| {
                        d.get("subreddit_name_prefixed")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .unwrap_or_default(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: d.get("score").and_then(Value::as_f64),
                provider: "reddit".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("search".into()),
                metadata,
            })
        })
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maps_metadata_fallback_and_cap() {
        let h=parse_json(r#"{"data":{"children":[{"data":{"title":"post","permalink":"/r/rust/comments/1/x","subreddit":"rust","score":7}},{"data":{"title":"two","url":"https://x"}}]}}"#,1).unwrap();
        assert_eq!(h.len(), 1);
        assert!(h[0].url.starts_with("https://www.reddit.com/"));
        assert_eq!(h[0].native_score, Some(7.0));
        assert_eq!(h[0].metadata["subreddit"], "rust");
    }
}
