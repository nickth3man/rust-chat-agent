use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use scraper::{Html, Selector};
use std::sync::Arc;

#[derive(Clone)]
pub struct LobstersBackend {
    http: BackendHttp,
    cap: usize,
}
impl LobstersBackend {
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
impl SearchBackend for LobstersBackend {
    async fn search(&self, q: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let html = self
            .http
            .get_text(
                "https://lobste.rs/search",
                &[("q", q), ("what", "stories"), ("order", "relevance")],
            )
            .await?;
        Ok(parse_results(&html, self.cap))
    }
    fn name(&self) -> &'static str {
        "lobsters"
    }
    fn category(&self) -> Category {
        Category::Code
    }
}
pub fn parse_html(html: &str, cap: usize) -> Vec<SearchHit> {
    parse_results(html, cap)
}
fn parse_results(html: &str, cap: usize) -> Vec<SearchHit> {
    let doc = Html::parse_document(html);
    let stories = Selector::parse("li.story, article.story").expect("static selector");
    let links = Selector::parse("h2 a, a.u-url, a.story-link").expect("static selector");
    let snippets = Selector::parse(".story_text, .story-summary, .summary, .details")
        .expect("static selector");
    doc.select(&stories)
        .take(cap)
        .enumerate()
        .filter_map(|(i, story)| {
            let link = story.select(&links).next()?;
            let url = link.value().attr("href")?.to_owned();
            let title = link.text().collect::<Vec<_>>().join(" ").trim().to_owned();
            let snippet = story
                .select(&snippets)
                .next()
                .map(|n| n.text().collect::<Vec<_>>().join(" ").trim().to_owned())
                .unwrap_or_default();
            Some(SearchHit {
                title: if title.is_empty() { url.clone() } else { title },
                url,
                snippet,
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "lobsters".into(),
                backend_kind: BackendKind::Html,
                source_subtype: Some("search".into()),
                metadata: Default::default(),
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_story_snippet_and_cap() {
        let h = parse_html(
            r#"<ul><li class="story"><h2><a href="https://lobste.rs/s/1">Rust story</a></h2><div class="story_text">A useful summary</div></li><li class="story"><h2><a href="https://x">Two</a></h2></li></ul>"#,
            1,
        );
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].title, "Rust story");
        assert_eq!(h[0].snippet, "A useful summary");
    }
}
