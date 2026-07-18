use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct SemanticScholarBackend {
    http: BackendHttp,
    cap: usize,
}
impl SemanticScholarBackend {
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
impl SearchBackend for SemanticScholarBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let cap = self.cap.to_string();
        let v: Value = self
            .http
            .get_json(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                &[
                    ("query", query),
                    ("limit", &cap),
                    ("fields", "title,abstract,url,year,citationCount"),
                ],
            )
            .await?;
        parse_json_value(&v, self.cap)
    }
    fn name(&self) -> &'static str {
        "semantic_scholar"
    }
    fn category(&self) -> Category {
        Category::Academic
    }
}
fn s(v: Option<&Value>) -> String {
    v.and_then(Value::as_str).unwrap_or_default().to_owned()
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let v: Value = serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    parse_json_value(&v, cap)
}
fn parse_json_value(v: &Value, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    Ok(v.get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, x)| {
            let url = s(x.get("url"));
            if url.is_empty() {
                return None;
            }
            let title = {
                let t = s(x.get("title"));
                if t.is_empty() { url.clone() } else { t }
            };
            let mut metadata = BTreeMap::new();
            if let Some(y) = x.get("year").and_then(Value::as_i64) {
                metadata.insert("year".into(), y.to_string());
            }
            Some(SearchHit {
                title,
                url,
                snippet: s(x.get("abstract")),
                published: x.get("year").and_then(Value::as_i64).map(|y| y.to_string()),
                native_rank: Some(i as u32 + 1),
                native_score: x.get("citationCount").and_then(Value::as_f64),
                provider: "semantic_scholar".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("paper_search".into()),
                metadata,
            })
        })
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture() {
        let h=parse_json(r#"{"data":[{"title":"T","abstract":"A","url":"https://x","year":2023,"citationCount":7}]}"#,1).unwrap();
        assert_eq!(h[0].native_score, Some(7.0));
    }
}
