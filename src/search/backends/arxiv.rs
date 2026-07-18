use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use quick_xml::Reader;
use quick_xml::events::Event;
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct ArxivBackend {
    http: BackendHttp,
    cap: usize,
}
impl ArxivBackend {
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
impl SearchBackend for ArxivBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let cap = self.cap.to_string();
        let xml = self
            .http
            .get_text(
                "https://export.arxiv.org/api/query",
                &[("search_query", query), ("max_results", &cap)],
            )
            .await?;
        parse_xml(&xml, self.cap)
    }
    fn name(&self) -> &'static str {
        "arxiv"
    }
    fn category(&self) -> Category {
        Category::Academic
    }
}

pub fn parse_xml(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let mut reader = Reader::from_str(input);
    reader.config_mut().trim_text(true);
    let mut hits = Vec::new();
    let mut current: Option<String> = None;
    let mut entry = BTreeMap::<String, String>::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let name = String::from_utf8_lossy(e.local_name().as_ref()).into_owned();
                if name == "entry" {
                    entry.clear();
                }
                current = Some(name);
            }
            Ok(Event::Text(t)) => {
                if let Some(key) = &current {
                    if ["id", "title", "summary", "published"].contains(&key.as_str()) {
                        entry.insert(
                            key.clone(),
                            t.decode()
                                .map_err(|e| ToolNetError::Parse(e.to_string()))?
                                .into_owned(),
                        );
                    }
                }
            }
            Ok(Event::End(e)) => {
                let name = String::from_utf8_lossy(e.local_name().as_ref()).into_owned();
                if name == "entry" && hits.len() < cap {
                    let url = entry.get("id").cloned().unwrap_or_default();
                    if !url.is_empty() {
                        let mut metadata = BTreeMap::new();
                        metadata.insert("id".into(), url.clone());
                        hits.push(SearchHit {
                            title: entry.get("title").cloned().unwrap_or_else(|| url.clone()),
                            url,
                            snippet: entry.get("summary").cloned().unwrap_or_default(),
                            published: entry.get("published").cloned(),
                            native_rank: Some(hits.len() as u32 + 1),
                            native_score: None,
                            provider: "arxiv".into(),
                            backend_kind: BackendKind::Xml,
                            source_subtype: Some("atom".into()),
                            metadata,
                        });
                    }
                }
                current = None;
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(ToolNetError::Parse(e.to_string())),
            _ => {}
        }
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture() {
        let x = r#"<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/1</id><title>A paper</title><summary>Abstract</summary><published>2024-01-01T00:00:00Z</published></entry></feed>"#;
        let h = parse_xml(x, 1).unwrap();
        assert_eq!(h[0].title, "A paper");
        assert_eq!(h[0].url, "https://arxiv.org/abs/1");
    }
}
