use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone)]
pub struct MwmblBackend {
    http: BackendHttp,
    cap: usize,
}

impl MwmblBackend {
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
        // api.mwmbl.org currently redirects to the canonical API endpoint.
        // Follow only this provider's documented redirect; other adapters
        // retain BackendHttp's fail-closed redirect policy.
        Ok(Some(Arc::new(Self::build(
            http.with_redirect_limit(5)?,
            cap,
        ))))
    }
}

#[async_trait]
impl SearchBackend for MwmblBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let response: Value = self
            .http
            .get_json("https://api.mwmbl.org/search", &[("s", query)])
            .await?;
        parse_value(response, self.cap)
    }

    fn name(&self) -> &'static str {
        "mwmbl"
    }
    fn category(&self) -> Category {
        Category::Web
    }
}

fn text(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    match value {
        Value::String(value) => value.clone(),
        Value::Array(values) => values.iter().map(|value| text(Some(value))).collect(),
        Value::Object(object) => ["text", "value", "content", "fragment"]
            .iter()
            .find_map(|key| object.get(*key))
            .map_or_else(String::new, |value| text(Some(value))),
        _ => String::new(),
    }
}

fn parse_value(value: Value, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let results = value
        .as_array()
        .ok_or_else(|| ToolNetError::Parse("Mwmbl response must be an array".into()))?;
    Ok(results
        .iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, result)| {
            let object = result.as_object()?;
            let url = object.get("url").and_then(Value::as_str)?.to_owned();
            let title = text(object.get("title")).trim().to_owned();
            Some(SearchHit {
                title: if title.is_empty() { url.clone() } else { title },
                url,
                snippet: text(object.get("extract")).trim().to_owned(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "mwmbl".into(),
                backend_kind: BackendKind::Api,
                source_subtype: Some("search".into()),
                metadata: Default::default(),
            })
        })
        .collect())
}

pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let value: Value =
        serde_json::from_str(input).map_err(|error| ToolNetError::Parse(error.to_string()))?;
    parse_value(value, cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fragment_forms_and_caps_fixture() {
        let input = r#"[{"title":[{"text":"Mwmbl"}," result"],"url":"https://one.test","extract":["A ",{"text":"snippet"}]},{"title":"Two","url":"https://two.test","extract":"Second"}]"#;
        let hits = parse_json(input, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Mwmbl result");
        assert_eq!(hits[0].snippet, "A snippet");
        assert_eq!(hits[0].native_rank, Some(1));
    }
}
