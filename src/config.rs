//! Project-local configuration and isolated environment resolution.

use crate::contracts::error::AppError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

fn default_model() -> String {
    "openrouter/auto".into()
}
fn default_true() -> bool {
    true
}
fn default_provider_timeout() -> u64 {
    6
}
fn default_stage_budget() -> u64 {
    20
}
fn default_rank_timeout() -> u64 {
    20
}
fn default_hit_cap() -> usize {
    5
}
fn default_output_bytes() -> usize {
    6144
}
fn default_concurrency() -> usize {
    1
}
fn default_user_agent() -> String {
    "openrouter-chat-rust/0.1 (+https://github.com/nickth3man/rust-chat-agent)".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsConfig {
    #[serde(default = "default_model")]
    pub chat_id: String,
    #[serde(default = "default_model")]
    pub rank_id: String,
    #[serde(default = "default_model")]
    pub summarize_id: String,
    pub chat_context_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    #[serde(default = "default_stage_budget")]
    pub stage_budget_secs: u64,
    #[serde(default = "default_rank_timeout")]
    pub rank_timeout_secs: u64,
    #[serde(default = "default_hit_cap")]
    pub per_backend_hit_cap: usize,
    #[serde(default = "default_output_bytes")]
    pub model_output_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    pub enable: bool,
    pub api_key_env: Option<String>,
    pub optional_api_key_env: Option<String>,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default)]
    pub min_interval_ms: u64,
    #[serde(default = "default_provider_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_user_agent")]
    pub user_agent: String,
    pub base_url: Option<String>,
    pub base_url_env: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionConfig {
    pub log_directory: PathBuf,
    #[serde(default = "default_true")]
    pub redact_credentials: bool,
    #[serde(default = "default_true")]
    pub redact_auth_headers: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FetchConfig {
    pub timeout_secs: u64,
    pub max_bytes: usize,
    pub max_chars: usize,
    pub allowed_schemes: Vec<String>,
    pub allowed_media_types: Vec<String>,
    pub redirect_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    pub models: ModelsConfig,
    pub search: SearchConfig,
    pub providers: BTreeMap<String, ProviderConfig>,
    pub session: SessionConfig,
    pub fetch: FetchConfig,
}

#[derive(Debug, Clone)]
pub struct ProviderSecret {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

/// Resolved secrets intentionally live outside the serializable public config.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub public: AppConfig,
    pub openrouter_api_key: String,
    pub provider_secrets: BTreeMap<String, ProviderSecret>,
}

pub fn load() -> Result<ResolvedConfig, AppError> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config_path = root.join("config.toml");
    if !config_path.is_file() {
        return Err(AppError::Config(format!(
            "missing {}; copy config.example.toml to config.toml",
            config_path.display()
        )));
    }
    let config_text = fs::read_to_string(&config_path)
        .map_err(|e| AppError::Config(format!("read {}: {e}", config_path.display())))?;
    let dotenv = fs::read_to_string(root.join(".env")).unwrap_or_default();
    resolve_inputs(
        &config_text,
        &parse_dotenv(&dotenv),
        &std::env::vars().collect(),
    )
}

/// Resolve using explicit maps so tests never mutate process-global environment state.
pub fn resolve_inputs(
    config_text: &str,
    dotenv: &BTreeMap<String, String>,
    process: &BTreeMap<String, String>,
) -> Result<ResolvedConfig, AppError> {
    let public: AppConfig = toml::from_str(config_text)
        .map_err(|e| AppError::Config(format!("parse config.toml: {e}")))?;
    if !public.session.redact_credentials || !public.session.redact_auth_headers {
        return Err(AppError::Config(
            "session redaction policies must remain true".into(),
        ));
    }

    let value = |key: &str| process.get(key).or_else(|| dotenv.get(key));

    let openrouter_api_key = value("OPENROUTER_API_KEY")
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| AppError::MissingCredential {
            provider: "openrouter".into(),
            env_var: "OPENROUTER_API_KEY".into(),
        })?
        .clone();
    let mut provider_secrets = BTreeMap::new();
    for (name, provider) in &public.providers {
        if !provider.enable {
            continue;
        }
        let api_key = provider
            .api_key_env
            .as_deref()
            .filter(|key| !key.trim().is_empty())
            .and_then(value)
            .cloned();
        if let Some(env_var) = provider
            .api_key_env
            .as_deref()
            .filter(|key| !key.trim().is_empty())
            && api_key.as_deref().is_none_or(|key| key.trim().is_empty())
        {
            return Err(AppError::MissingCredential {
                provider: name.clone(),
                env_var: env_var.to_owned(),
            });
        }
        let optional_api_key = provider
            .optional_api_key_env
            .as_deref()
            .filter(|key| !key.trim().is_empty())
            .and_then(value)
            .filter(|key| !key.trim().is_empty())
            .cloned();
        let base_url_from_env = provider
            .base_url_env
            .as_deref()
            .filter(|key| !key.trim().is_empty())
            .and_then(value)
            .cloned()
            .or_else(|| {
                provider
                    .base_url
                    .as_deref()
                    .filter(|url| !url.trim().is_empty())
                    .map(str::to_owned)
            });
        if provider
            .base_url_env
            .as_deref()
            .is_some_and(|env_var| !env_var.trim().is_empty())
            && base_url_from_env.is_none()
        {
            return Err(AppError::Config(format!(
                "provider {name} requires non-empty endpoint environment variable {}",
                provider.base_url_env.as_deref().unwrap_or_default()
            )));
        }
        provider_secrets.insert(
            name.clone(),
            ProviderSecret {
                api_key: api_key.or(optional_api_key),
                base_url: base_url_from_env,
            },
        );
    }
    Ok(ResolvedConfig {
        public,
        openrouter_api_key,
        provider_secrets,
    })
}

pub fn parse_dotenv(contents: &str) -> BTreeMap<String, String> {
    contents
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            let value = value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
            Some((key.trim().to_string(), value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn maps() -> (BTreeMap<String, String>, BTreeMap<String, String>) {
        (BTreeMap::new(), BTreeMap::new())
    }
    #[test]
    fn example_config_parses() {
        let text = include_str!("../config.example.toml");
        let config: AppConfig = toml::from_str(text).unwrap();
        assert!(config.providers.contains_key("duckduckgo"));
        assert_eq!(config.models.chat_context_tokens, 12000);
    }

    #[test]
    fn removed_model_context_fields_are_rejected() {
        let text = include_str!("../config.example.toml");
        for field in ["rank_context_tokens", "summarize_context_tokens"] {
            let text = text.replacen(
                "chat_context_tokens = 12000",
                &format!("chat_context_tokens = 12000\n{field} = 8000"),
                1,
            );
            assert!(
                toml::from_str::<AppConfig>(&text).is_err(),
                "accepted {field}"
            );
        }
    }
    #[test]
    fn model_ids_are_toml_only_while_process_secrets_win() {
        let (mut dotenv, mut process) = maps();
        dotenv.insert("OPENROUTER_CHAT_MODEL".into(), "from-dotenv".into());
        dotenv.insert("OPENROUTER_RANK_MODEL".into(), "from-dotenv".into());
        dotenv.insert("OPENROUTER_SUMMARIZE_MODEL".into(), "from-dotenv".into());
        dotenv.insert("OPENROUTER_API_KEY".into(), "dotenv-secret".into());
        process.insert("OPENROUTER_CHAT_MODEL".into(), "from-process".into());
        process.insert("OPENROUTER_RANK_MODEL".into(), "from-process".into());
        process.insert("OPENROUTER_SUMMARIZE_MODEL".into(), "from-process".into());
        process.insert("OPENROUTER_API_KEY".into(), "process-secret".into());
        let resolved =
            resolve_inputs(include_str!("../config.example.toml"), &dotenv, &process).unwrap();
        assert_eq!(resolved.public.models.chat_id, "openrouter/auto");
        assert_eq!(resolved.public.models.rank_id, "openrouter/auto");
        assert_eq!(resolved.public.models.summarize_id, "openrouter/auto");
        assert_eq!(resolved.openrouter_api_key, "process-secret");
    }
    #[test]
    fn missing_key_names_provider_and_environment_variable() {
        let (mut dotenv, process) = maps();
        dotenv.insert("OPENROUTER_API_KEY".into(), "secret".into());
        let text = include_str!("../config.example.toml")
            .replacen("enable = false", "enable = true", 1)
            .replace("BRAVE_API_KEY", "BRAVE_KEY");
        let error = resolve_inputs(&text, &dotenv, &process)
            .unwrap_err()
            .to_string();
        assert!(error.contains("brave") && error.contains("BRAVE_KEY"));
    }
    #[test]
    fn provider_matrix_has_keyless_defaults_and_no_forbidden_entries() {
        let config: AppConfig = toml::from_str(include_str!("../config.example.toml")).unwrap();
        assert_eq!(config.search.stage_budget_secs, 20);
        assert_eq!(config.search.rank_timeout_secs, 20);
        assert_eq!(config.search.per_backend_hit_cap, 5);
        assert_eq!(config.search.model_output_bytes, 6144);
        let keyless = [
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
        for name in keyless {
            assert!(config.providers.get(name).unwrap().enable);
            assert_eq!(config.providers[name].timeout_secs, 6);
            assert_eq!(
                config.providers[name].user_agent,
                "openrouter-chat-rust/0.1 (+https://github.com/nickth3man/rust-chat-agent)"
            );
        }
        for name in ["brave", "mojeek", "searxng", "firecrawl"] {
            assert!(!config.providers.get(name).unwrap().enable);
        }
        assert_eq!(config.providers.len(), keyless.len() + 4);
        assert_eq!(config.providers["firecrawl"].timeout_secs, 15);
        for forbidden in [
            "openalex",
            "internet_archive",
            "wayback",
            "pypi",
            "tavily",
            "openrouter_search",
        ] {
            assert!(!config.providers.contains_key(forbidden));
        }
    }

    #[test]
    fn optional_credentials_are_best_effort() {
        let mut process = BTreeMap::new();
        process.insert("OPENROUTER_API_KEY".into(), "secret".into());
        let config = include_str!("../config.example.toml");
        let resolved = resolve_inputs(config, &BTreeMap::new(), &process).unwrap();
        assert_eq!(resolved.provider_secrets["github"].api_key, None);
        process.insert("GITHUB_TOKEN".into(), "token".into());
        let resolved = resolve_inputs(config, &BTreeMap::new(), &process).unwrap();
        assert_eq!(
            resolved.provider_secrets["github"].api_key.as_deref(),
            Some("token")
        );
    }

    #[test]
    fn enabled_endpoint_env_is_required() {
        let mut process = BTreeMap::new();
        process.insert("OPENROUTER_API_KEY".into(), "secret".into());
        let (before, after) = include_str!("../config.example.toml")
            .split_once("[providers.searxng]")
            .unwrap();
        let config = format!(
            "{before}[providers.searxng]{}",
            after.replacen("enable = false", "enable = true", 1)
        );
        let error = resolve_inputs(&config, &BTreeMap::new(), &process)
            .unwrap_err()
            .to_string();
        assert!(error.contains("searxng") && error.contains("SEARXNG_BASE_URL"));
    }
}
