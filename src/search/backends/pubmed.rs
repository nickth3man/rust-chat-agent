use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct PubmedBackend {
    http: BackendHttp,
    cap: usize,
}
impl PubmedBackend {
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
impl SearchBackend for PubmedBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let cap = self.cap.to_string();
        let ids: Value = self
            .http
            .get_json(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                &[
                    ("db", "pubmed"),
                    ("term", query),
                    ("retmode", "json"),
                    ("retmax", &cap),
                ],
            )
            .await?;
        let id_list = parse_ids_value(&ids);
        if id_list.is_empty() {
            return Ok(Vec::new());
        }
        let joined = id_list.join(",");
        let summary: Value = self
            .http
            .get_json(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                &[("db", "pubmed"), ("id", &joined), ("retmode", "json")],
            )
            .await?;
        parse_summary_value(&summary, self.cap)
    }
    fn name(&self) -> &'static str {
        "pubmed"
    }
    fn category(&self) -> Category {
        Category::Academic
    }
}
pub fn parse_ids(input: &str) -> Result<Vec<String>, ToolNetError> {
    let v: Value = serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_ids_value(&v))
}
fn parse_ids_value(v: &Value) -> Vec<String> {
    v.pointer("/esearchresult/idlist")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}
pub fn parse_summary(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let v: Value = serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    parse_summary_value(&v, cap)
}
fn parse_summary_value(v: &Value, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let map = v
        .get("result")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let order = map
        .get("uids")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok(order
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, id)| {
            let id = id.as_str()?;
            let x = map.get(id)?;
            let title = x
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or(id)
                .to_owned();
            let url = format!("https://pubmed.ncbi.nlm.nih.gov/{id}/");
            let mut metadata = BTreeMap::new();
            metadata.insert("pmid".into(), id.into());
            Some(SearchHit {
                title,
                url,
                snippet: x
                    .get("sortfirstauthor")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                published: x.get("pubdate").and_then(Value::as_str).map(str::to_owned),
                native_rank: Some(i as u32 + 1),
                native_score: None,
                provider: "pubmed".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("esummary".into()),
                metadata,
            })
        })
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_shapes() {
        assert_eq!(
            parse_ids(r#"{"esearchresult":{"idlist":["1","2"]}}"#)
                .unwrap()
                .len(),
            2
        );
        let h = parse_summary(
            r#"{"result":{"uids":["1"],"1":{"title":"Paper","pubdate":"2020"}}}"#,
            1,
        )
        .unwrap();
        assert_eq!(h[0].url, "https://pubmed.ncbi.nlm.nih.gov/1/");
    }
}
