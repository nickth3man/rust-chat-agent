use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;
#[derive(Clone)]
pub struct MojeekBackend {
    http: BackendHttp,
    cap: usize,
    api_key: String,
}
#[derive(Deserialize)]
struct Response {
    status: Option<String>,
    #[serde(default)]
    results: Vec<Item>,
}
#[derive(Deserialize)]
struct Item {
    url: Option<String>,
    title: Option<String>,
    desc: Option<String>,
    score: Option<f64>,
    date: Option<String>,
}
impl MojeekBackend {
    pub fn build(http: BackendHttp, api_key: String, cap: usize) -> Self {
        Self { http, cap, api_key }
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
        let key = secret
            .and_then(|s| s.api_key.clone())
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| AppError::Config("mojeek requires a resolved API key".into()))?;
        Ok(Some(Arc::new(Self::build(http, key, cap))))
    }
}
#[async_trait]
impl SearchBackend for MojeekBackend {
    async fn search(&self, q: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let r: Response = self
            .http
            .get_json(
                "https://api.mojeek.com/search",
                &[
                    ("q", q),
                    ("api_key", &self.api_key),
                    ("fmt", "json"),
                    ("t", &self.cap.to_string()),
                ],
            )
            .await?;
        if r.status
            .as_deref()
            .is_some_and(|s| !s.eq_ignore_ascii_case("ok") && !s.eq_ignore_ascii_case("success"))
        {
            return Err(ToolNetError::Content(
                "mojeek reported an unsuccessful response".into(),
            ));
        }
        Ok(parse_response(r, self.cap))
    }
    fn name(&self) -> &'static str {
        "mojeek"
    }
    fn category(&self) -> Category {
        Category::Web
    }
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let r: Response =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    if r.status
        .as_deref()
        .is_some_and(|s| !s.eq_ignore_ascii_case("ok") && !s.eq_ignore_ascii_case("success"))
    {
        return Err(ToolNetError::Content(
            "mojeek reported an unsuccessful response".into(),
        ));
    }
    Ok(parse_response(r, cap))
}
fn parse_response(r: Response, cap: usize) -> Vec<SearchHit> {
    r.results
        .into_iter()
        .take(cap)
        .enumerate()
        .filter_map(|(i, x)| {
            Some(SearchHit {
                title: x.title?,
                url: x.url?,
                snippet: x.desc.unwrap_or_default(),
                published: x.date,
                native_rank: Some((i + 1) as u32),
                native_score: x.score,
                provider: "mojeek".into(),
                backend_kind: BackendKind::Api,
                source_subtype: Some("web".into()),
                metadata: Default::default(),
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_score_and_error() {
        let h=parse_json(r#"{"status":"OK","results":[{"url":"https://e","title":"T","desc":"D","score":0.8,"date":"2020"},{"url":"https://2","title":"2"}]}"#,1).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].native_score, Some(0.8));
        assert!(parse_json(r#"{"status":"error","results":[]}"#, 5).is_err());
    }
}
