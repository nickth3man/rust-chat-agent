use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::{AppError, BackendKind, Category, SearchBackend, SearchHit, ToolNetError};
use crate::search::http::BackendHttp;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;
use url::Url;

#[derive(Clone)]
pub struct FreeDictionaryBackend {
    http: BackendHttp,
    cap: usize,
}
#[derive(Deserialize)]
struct Entry {
    word: Option<String>,
    phonetic: Option<String>,
    #[serde(default)]
    meanings: Vec<Meaning>,
    #[serde(rename = "sourceUrls")]
    source_urls: Option<Vec<String>>,
}
#[derive(Deserialize)]
struct Meaning {
    #[serde(rename = "partOfSpeech")]
    part_of_speech: Option<String>,
    definitions: Vec<Definition>,
}
#[derive(Deserialize)]
struct Definition {
    definition: Option<String>,
    example: Option<String>,
}
impl FreeDictionaryBackend {
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
impl SearchBackend for FreeDictionaryBackend {
    async fn search(&self, query: &str) -> Result<Vec<SearchHit>, ToolNetError> {
        let mut url =
            Url::parse("https://api.dictionaryapi.dev/api/v2/entries/en/").expect("static URL");
        url.path_segments_mut().unwrap().push(query);
        let response: Vec<Entry> = self.http.get_json(url.as_str(), &[]).await?;
        Ok(parse_response(response, self.cap))
    }
    fn name(&self) -> &'static str {
        "free_dictionary"
    }
    fn category(&self) -> Category {
        Category::Dictionary
    }
}
fn parse_response(entries: Vec<Entry>, cap: usize) -> Vec<SearchHit> {
    entries
        .into_iter()
        .take(cap)
        .enumerate()
        .map(|(i, e)| {
            let mut parts = Vec::new();
            let mut defs = Vec::new();
            let mut examples = Vec::new();
            for m in e.meanings {
                if let Some(p) = m.part_of_speech {
                    parts.push(p);
                }
                for d in m.definitions {
                    if let Some(definition) = d.definition {
                        defs.push(definition);
                    }
                    if let Some(example) = d.example.filter(|example| !example.trim().is_empty()) {
                        examples.push(example);
                    }
                }
            }
            let source = e
                .source_urls
                .as_ref()
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or_else(|| {
                    format!(
                        "https://api.dictionaryapi.dev/api/v2/entries/en/{}",
                        e.word.as_deref().unwrap_or_default()
                    )
                });
            let mut metadata = std::collections::BTreeMap::new();
            if !parts.is_empty() {
                metadata.insert("parts_of_speech".into(), parts.join(", "));
            }
            if let Some(p) = e.phonetic {
                metadata.insert("phonetic".into(), p);
            }
            if let Some(urls) = e.source_urls {
                metadata.insert("source_urls".into(), urls.join(", "));
            }
            if !examples.is_empty() {
                metadata.insert("examples".into(), examples.join(" | "));
            }
            SearchHit {
                title: e.word.unwrap_or_default(),
                url: source,
                snippet: defs.join(" "),
                published: None,
                native_rank: Some((i + 1) as u32),
                native_score: None,
                provider: "free_dictionary".into(),
                backend_kind: BackendKind::Json,
                source_subtype: Some("dictionaryapi".into()),
                metadata,
            }
        })
        .collect()
}
pub fn parse_json(input: &str, cap: usize) -> Result<Vec<SearchHit>, ToolNetError> {
    let r: Vec<Entry> =
        serde_json::from_str(input).map_err(|e| ToolNetError::Parse(e.to_string()))?;
    Ok(parse_response(r, cap))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_definition_fixture() {
        let h = parse_json(r#"[{"word":"bright","meanings":[{"partOfSpeech":"adjective","definitions":[{"definition":"Giving out light"}]}],"sourceUrls":["https://en.wiktionary.org/wiki/bright"]}]"#, 1).unwrap();
        assert_eq!(h[0].snippet, "Giving out light");
        assert_eq!(h[0].metadata["parts_of_speech"], "adjective");
    }
}
