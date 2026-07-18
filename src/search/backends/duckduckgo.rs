use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use scraper::{Html, Selector};
use std::sync::Arc;
use url::Url;

#[derive(Clone)]
pub struct DuckDuckGoBackend {
    http: BackendHttp,
    cap: usize,
}
impl DuckDuckGoBackend {
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
impl SearchBackend for DuckDuckGoBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let html = self
            .http
            .post_form_text("https://lite.duckduckgo.com/lite/", &[("q", query)])
            .await?;
        Ok(parse_results(&html, self.cap))
    }
    fn name(&self) -> &'static str {
        "duckduckgo"
    }
    fn category(&self) -> Category {
        Category::Web
    }
}

pub fn parse_results(html: &str, cap: usize) -> Vec<SearchHit> {
    let doc = Html::parse_document(html);
    let result = Selector::parse("a.result-link").expect("static selector");
    let snippet = Selector::parse(".result-snippet").expect("static selector");
    doc.select(&result)
        .take(cap)
        .enumerate()
        .filter_map(|(i, node)| {
            let title = node.text().collect::<Vec<_>>().join(" ").trim().to_owned();
            let raw = node.value().attr("href")?;
            let url = unwrap_redirect(raw);
            let mut ancestor = node.parent();
            let mut text = None;
            for _ in 0..4 {
                let Some(parent) = ancestor.and_then(scraper::ElementRef::wrap) else {
                    break;
                };
                if let Some(found) = parent.select(&snippet).next() {
                    text = Some(found.text().collect::<Vec<_>>().join(" "));
                    break;
                }
                ancestor = parent.parent();
            }
            let text = text.unwrap_or_default();
            Some(SearchHit {
                title,
                url,
                snippet: text,
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "duckduckgo".into(),
                backend_kind: BackendKind::Html,
                source_subtype: Some("lite".into()),
                metadata: Default::default(),
            })
        })
        .collect()
}
pub fn unwrap_redirect(raw: &str) -> String {
    let url = Url::parse(raw)
        .or_else(|_| Url::parse("https://duckduckgo.com").and_then(|base| base.join(raw)));
    let Ok(url) = url else {
        return raw.to_owned();
    };
    for key in ["uddg", "u", "url"] {
        if let Some(value) = url
            .query_pairs()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.into_owned())
        {
            return value;
        }
    }
    raw.to_owned()
}
