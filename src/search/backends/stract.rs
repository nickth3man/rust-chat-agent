use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Clone)]
pub struct StractBackend {
    http: BackendHttp,
    cap: usize,
}

impl StractBackend {
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
impl SearchBackend for StractBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let response: Value = self
            .http
            .post_json(
                "https://stract.com/beta/api/search",
                &json!({"query": query, "numResults": self.cap}),
            )
            .await?;
        parse_json_value(&response, self.cap)
    }
    fn name(&self) -> &'static str {
        "stract"
    }
    fn category(&self) -> Category {
        Category::Web
    }
}

fn text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| text(Some(v)))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        Some(Value::Object(map)) => map
            .get("text")
            .or_else(|| map.get("fragments"))
            .map(|v| text(Some(v)))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let value: Value =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    parse_json_value(&value, cap)
}

fn parse_json_value(value: &Value, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    Ok(value
        .get("webpages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(cap)
        .enumerate()
        .filter_map(|(i, item)| {
            let url = item.get("url")?.as_str()?.to_owned();
            let title = item
                .get("title")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .unwrap_or(&url)
                .to_owned();
            Some(SearchHit {
                title,
                url,
                snippet: text(item.get("snippet")),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "stract".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("beta".into()),
                metadata: Default::default(),
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_and_caps() {
        let s = r#"{"webpages":[{"title":"Rust","url":"https://rust-lang.org","snippet":[{"text":"fast"},{"text":"safe"}]},{"url":"https://example.com"}]}"#;
        let h = parse_json(s, 1).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].title, "Rust");
        assert_eq!(h[0].snippet, "fast safe");
    }
}
