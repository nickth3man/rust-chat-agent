use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct CrossrefBackend {
    http: BackendHttp,
    cap: usize,
}
impl CrossrefBackend {
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
impl SearchBackend for CrossrefBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let cap = self.cap.to_string();
        let v: Value = self
            .http
            .get_json(
                "https://api.crossref.org/works",
                &[("query", query), ("rows", &cap)],
            )
            .await?;
        parse_json_value(&v, self.cap)
    }
    fn name(&self) -> &'static str {
        "crossref"
    }
    fn category(&self) -> Category {
        Category::Academic
    }
}
fn text(v: Option<&Value>) -> Option<String> {
    v.and_then(|x| x.as_str().map(str::to_owned))
        .filter(|x| !x.trim().is_empty())
}
fn date(v: Option<&Value>) -> Option<String> {
    let a = v?.get("date-parts")?.get(0)?.as_array()?;
    Some(
        a.iter()
            .filter_map(|x| x.as_i64())
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("-"),
    )
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let v: Value = serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    parse_json_value(&v, cap)
}
fn parse_json_value(v: &Value, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let items = v
        .pointer("/message/items")
        .and_then(Value::as_array)
        .or_else(|| v.get("items").and_then(Value::as_array))
        .cloned()
        .unwrap_or_default();
    Ok(items
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, x)| {
            let doi = text(x.get("DOI"));
            let url = doi
                .as_ref()
                .map(|d| format!("https://doi.org/{d}"))
                .or_else(|| text(x.get("URL")))?;
            let title = x
                .get("title")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(|x| text(Some(x)))
                .unwrap_or_else(|| url.clone());
            let mut metadata = BTreeMap::new();
            if let Some(d) = &doi {
                metadata.insert("doi".into(), d.clone());
            }
            Some(SearchHit {
                title,
                url,
                snippet: text(x.get("abstract")).unwrap_or_default(),
                published: date(x.get("published").or_else(|| x.get("issued"))),
                native_rank: Some(i as u32 + 1),
                native_score: None,
                provider: "crossref".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("works".into()),
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
        let h=parse_json(r#"{"message":{"items":[{"DOI":"10/x","title":["T"],"abstract":"A","published":{"date-parts":[[2020,2,3]]}}]}}"#,1).unwrap();
        assert_eq!(h[0].url, "https://doi.org/10/x");
        assert_eq!(h[0].published.as_deref(), Some("2020-2-3"));
    }
}
