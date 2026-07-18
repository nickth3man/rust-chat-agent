use crate::config::ResolvedConfig;
use crate::contracts::{AppError, SearchBackend};
use crate::search::backends;
use crate::search::http::{BackendHttp, LimiterPools};
use std::sync::Arc;

/// Collection of enabled adapters. B-merge owns provider dispatch. Every
/// adapter freezes this constructor convention:
/// `from_config(cfg: &ProviderConfig, http: BackendHttp,
/// secret: Option<&ProviderSecret>, cap: usize) ->
/// Result<Option<Arc<dyn SearchBackend>>, AppError>`.
#[derive(Default, Clone)]
pub struct BackendRegistry {
    backends: Vec<Arc<dyn SearchBackend>>,
}
impl BackendRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, backend: Arc<dyn SearchBackend>) {
        self.backends.push(backend);
    }
    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn SearchBackend>> {
        self.backends.iter()
    }
    pub fn find(&self, name: &str) -> Option<&Arc<dyn SearchBackend>> {
        self.backends.iter().find(|b| b.name() == name)
    }
    pub fn enabled_names(&self) -> Vec<&'static str> {
        self.backends.iter().map(|b| b.name()).collect()
    }

    /// Sole B-merge dispatch point. Adapter constructors all use the frozen
    /// `from_config(cfg, http, secret, cap)` convention; provider-specific
    /// policy remains in the adapter modules.
    pub fn from_config(config: &ResolvedConfig) -> Result<Self, AppError> {
        const NAMES: &[&str] = &[
            "duckduckgo",
            "hn",
            "stract",
            "wiby",
            "mdn",
            "reddit",
            "lobsters",
            "searchmysite",
            "marginalia",
            "mwmbl",
            "wikipedia",
            "wikidata",
            "openlibrary",
            "free_dictionary",
            "arxiv",
            "crossref",
            "semantic_scholar",
            "pubmed",
            "github",
            "stackexchange",
            "npm",
            "crates_io",
            "gdelt",
            "firecrawl",
            "brave",
            "mojeek",
            "searxng",
        ];
        for name in config.public.providers.keys() {
            if !NAMES.contains(&name.as_str()) {
                return Err(AppError::Config(format!("unknown search provider: {name}")));
            }
        }

        let mut pools = LimiterPools::new();
        let shared = config
            .public
            .providers
            .get("wikipedia")
            .or_else(|| config.public.providers.get("wikidata"))
            .or_else(|| config.public.providers.get("openlibrary"));
        if let Some(cfg) = shared {
            pools.insert_shared("wikimedia-reference", cfg.concurrency, cfg.min_interval_ms);
        }
        if let Some(cfg) = config.public.providers.get("arxiv") {
            pools.insert_shared("arxiv", cfg.concurrency, cfg.min_interval_ms.max(3000));
        }

        let mut registry = Self::new();
        for (name, cfg) in &config.public.providers {
            if !cfg.enable {
                continue;
            }
            let secret = config.provider_secrets.get(name);
            let http = BackendHttp::new(name, cfg, secret)?;
            let http = if matches!(name.as_str(), "wikipedia" | "wikidata" | "openlibrary") {
                http.with_shared_limits(
                    pools.shared("wikimedia-reference").expect("reference pool"),
                )
            } else if name == "arxiv" {
                http.with_shared_limits(pools.shared("arxiv").expect("arxiv pool"))
            } else {
                http
            };
            let backend = match name.as_str() {
                "duckduckgo" => backends::duckduckgo::DuckDuckGoBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "hn" => backends::hn::HnBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "stract" => backends::stract::StractBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "wiby" => backends::wiby::WibyBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "mdn" => backends::mdn::MdnBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "reddit" => backends::reddit::RedditBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "lobsters" => backends::lobsters::LobstersBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "searchmysite" => backends::searchmysite::SearchMySiteBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "marginalia" => backends::marginalia::MarginaliaBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "mwmbl" => backends::mwmbl::MwmblBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "wikipedia" => backends::wikipedia::WikipediaBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "wikidata" => backends::wikidata::WikidataBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "openlibrary" => backends::openlibrary::OpenLibraryBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "free_dictionary" => backends::freedictionary::FreeDictionaryBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "arxiv" => backends::arxiv::ArxivBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "crossref" => backends::crossref::CrossrefBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "semantic_scholar" => {
                    backends::semantic_scholar::SemanticScholarBackend::from_config(
                        cfg,
                        http,
                        secret,
                        config.public.search.per_backend_hit_cap,
                    )?
                }
                "pubmed" => backends::pubmed::PubmedBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "github" => backends::github::GithubBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "stackexchange" => backends::stackexchange::StackexchangeBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "npm" => backends::npm::NpmBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "crates_io" => backends::crates_io::CratesIoBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "gdelt" => backends::gdelt::GdeltBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "firecrawl" => backends::firecrawl::FirecrawlBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "brave" => backends::brave::BraveBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "mojeek" => backends::mojeek::MojeekBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                "searxng" => backends::searxng::SearxngBackend::from_config(
                    cfg,
                    http,
                    secret,
                    config.public.search.per_backend_hit_cap,
                )?,
                _ => unreachable!("validated provider name"),
            };
            if let Some(backend) = backend {
                registry.add(backend);
            }
        }
        Ok(registry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::resolve_inputs;
    use std::collections::BTreeMap;

    #[test]
    fn example_config_builds_expected_provider_registry() {
        let mut process = BTreeMap::new();
        process.insert("OPENROUTER_API_KEY".into(), "test-key".into());
        let config = resolve_inputs(
            include_str!("../../config.example.toml"),
            &BTreeMap::new(),
            &process,
        )
        .unwrap();

        let registry = BackendRegistry::from_config(&config).unwrap();
        let enabled = registry.enabled_names();
        let key_free = [
            "duckduckgo",
            "stract",
            "marginalia",
            "mwmbl",
            "wiby",
            "searchmysite",
            "wikipedia",
            "wikidata",
            "openlibrary",
            "free_dictionary",
            "arxiv",
            "crossref",
            "semantic_scholar",
            "pubmed",
            "hn",
            "github",
            "stackexchange",
            "npm",
            "crates_io",
            "mdn",
            "gdelt",
            "reddit",
            "lobsters",
        ];
        for name in key_free {
            assert!(enabled.contains(&name), "missing enabled provider {name}");
        }
        for name in ["brave", "mojeek", "searxng", "firecrawl"] {
            assert!(
                !enabled.contains(&name),
                "disabled provider was enabled: {name}"
            );
        }
    }
}
