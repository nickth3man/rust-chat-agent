use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct WibyBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize)]
struct WibyHit {
    #[serde(rename = "Title")]
    title: Option<String>,
    #[serde(rename = "URL")]
    url: Option<String>,
    #[serde(rename = "Snippet")]
    snippet: Option<String>,
}
impl WibyBackend {
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
impl SearchBackend for WibyBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let hits: Vec<WibyHit> = self
            .http
            .get_json("https://wiby.me/json/", &[("q", query)])
            .await?;
        Ok(parse_results(hits, self.cap))
    }
    fn name(&self) -> &'static str {
        "wiby"
    }
    fn category(&self) -> Category {
        Category::Web
    }
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let hits: Vec<WibyHit> =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_results(hits, cap))
}
fn parse_results(hits: Vec<WibyHit>, cap: usize) -> Vec<SearchHit> {
    hits.into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, h)| {
            let url = h.url?;
            let title = h.title.unwrap_or_else(|| url.clone());
            Some(SearchHit {
                title,
                url,
                snippet: h.snippet.unwrap_or_default(),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "wiby".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("classic-web".into()),
                metadata: Default::default(),
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_fallback_and_cap() {
        let h=parse_json(r#"[{"Title":"","URL":"https://example.com","Snippet":"old web"},{"Title":"two","URL":"https://two"}]"#,1).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].url, "https://example.com");
        assert_eq!(h[0].snippet, "old web");
    }
}
