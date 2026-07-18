use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct MdnBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize)]
struct MdnResponse {
    #[serde(default)]
    documents: Vec<MdnDocument>,
}
#[derive(Deserialize)]
struct MdnDocument {
    title: Option<String>,
    mdn_url: Option<String>,
    summary: Option<String>,
}
impl MdnBackend {
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
impl SearchBackend for MdnBackend {
    async fn search(&self, q: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let r: MdnResponse = self
            .http
            .get_json("https://developer.mozilla.org/api/v1/search", &[("q", q)])
            .await?;
        Ok(parse_results(r.documents, self.cap))
    }
    fn name(&self) -> &'static str {
        "mdn"
    }
    fn category(&self) -> Category {
        Category::Code
    }
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let r: MdnResponse =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_results(r.documents, cap))
}
fn parse_results(docs: Vec<MdnDocument>, cap: usize) -> Vec<SearchHit> {
    docs.into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, d)| {
            let raw = d.mdn_url?;
            let url = if raw.starts_with("http://") || raw.starts_with("https://") {
                raw
            } else {
                format!(
                    "https://developer.mozilla.org{}",
                    if raw.starts_with('/') {
                        raw
                    } else {
                        format!("/{raw}")
                    }
                )
            };
            let title = d.title.unwrap_or_else(|| url.clone());
            Some(SearchHit {
                title,
                url,
                snippet: d.summary.unwrap_or_default(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "mdn".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("mdn-api".into()),
                metadata: Default::default(),
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absolute_url_and_cap() {
        let h=parse_json(r#"{"documents":[{"title":"Fetch","mdn_url":"/en-US/docs/Web/API/Fetch_API","summary":"network"},{"mdn_url":"/two"}]}"#,1).unwrap();
        assert_eq!(
            h[0].url,
            "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API"
        );
        assert_eq!(h[0].snippet, "network");
    }
}
