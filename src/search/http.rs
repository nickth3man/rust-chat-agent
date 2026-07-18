use crate::config::{ProviderConfig, ProviderSecret};
use crate::contracts::ToolNetError;
use futures::StreamExt;
use governor::{Quota, RateLimiter};
use reqwest::{Client, Method, Response};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

const DEFAULT_MAX_BODY: usize = 2 * 1024 * 1024;

struct Slots(Arc<tokio::sync::Semaphore>);
impl Slots {
    async fn take(self: &Arc<Self>) -> Slot {
        Slot {
            _permit: self
                .0
                .clone()
                .acquire_owned()
                .await
                .expect("semaphore is never closed"),
        }
    }
}
struct Slot {
    _permit: tokio::sync::OwnedSemaphorePermit,
}

/// The concurrency and rate limits shared by a family of providers.
#[derive(Clone)]
pub struct SharedLimits {
    slots: Arc<Slots>,
    limiter: Arc<governor::DefaultDirectRateLimiter>,
}

impl SharedLimits {
    pub fn new(concurrency: usize, min_interval_ms: u64) -> Self {
        let quota = if min_interval_ms == 0 {
            Quota::per_second(std::num::NonZeroU32::new(1000).unwrap())
        } else {
            Quota::with_period(Duration::from_millis(min_interval_ms))
                .unwrap()
                .allow_burst(std::num::NonZeroU32::new(1).unwrap())
        };
        Self {
            slots: Arc::new(Slots(Arc::new(tokio::sync::Semaphore::new(
                concurrency.max(1),
            )))),
            limiter: Arc::new(RateLimiter::direct(quota)),
        }
    }
}

/// Cloneable transport. Authentication is deliberately not implicit: adapters
/// choose the exact header or query parameter required by their provider.
#[derive(Clone)]
pub struct BackendHttp {
    client: Client,
    provider: String,
    base_url: Option<String>,
    user_agent: String,
    retry_budget: Duration,
    retry_backoff: Duration,
    max_body: usize,
    limits: Arc<SharedLimits>,
}
impl std::fmt::Debug for BackendHttp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendHttp")
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl BackendHttp {
    pub fn new(
        provider: impl Into<String>,
        cfg: &ProviderConfig,
        secret: Option<&ProviderSecret>,
    ) -> Result<Self, ToolNetError> {
        let timeout = Duration::from_secs(cfg.timeout_secs.max(1));
        Self::build(
            provider,
            secret
                .and_then(|value| value.base_url.clone())
                .or_else(|| cfg.base_url.clone()),
            cfg.user_agent.clone(),
            timeout,
            SharedLimits::new(cfg.concurrency, cfg.min_interval_ms),
        )
    }

    fn build(
        provider: impl Into<String>,
        base_url: Option<String>,
        user_agent: String,
        timeout: Duration,
        limits: SharedLimits,
    ) -> Result<Self, ToolNetError> {
        let client = Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| ToolNetError::Network(e.to_string()))?;
        Ok(Self {
            client,
            provider: provider.into(),
            base_url,
            user_agent,
            retry_budget: timeout,
            retry_backoff: Duration::from_millis(100),
            max_body: DEFAULT_MAX_BODY,
            limits: Arc::new(limits),
        })
    }
    pub fn with_shared_limits(mut self, limits: Arc<SharedLimits>) -> Self {
        self.limits = limits;
        self
    }
    /// Test and embedding hook for millisecond timeouts and retry policies.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self, ToolNetError> {
        self.client = Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| ToolNetError::Network(e.to_string()))?;
        self.retry_budget = timeout;
        Ok(self)
    }
    pub fn with_limits(mut self, max_body: usize, retry_budget: Duration) -> Self {
        self.max_body = max_body;
        self.retry_budget = retry_budget;
        self
    }
    pub fn with_retry_backoff(mut self, backoff: Duration) -> Self {
        self.retry_backoff = backoff;
        self
    }
    pub fn provider(&self) -> &str {
        &self.provider
    }
    pub fn url(&self, path: &str) -> String {
        match &self.base_url {
            Some(base) => format!(
                "{}{}",
                base.trim_end_matches('/'),
                if path.starts_with('/') {
                    path.to_owned()
                } else {
                    format!("/{path}")
                }
            ),
            None => path.to_owned(),
        }
    }
    pub async fn get_json<T: DeserializeOwned>(
        &self,
        url: &str,
        query: &[(&str, &str)],
    ) -> Result<T, ToolNetError> {
        self.request_json(Method::GET, url, query, None::<&()>)
            .await
    }
    pub async fn get_text(
        &self,
        url: &str,
        query: &[(&str, &str)],
    ) -> Result<String, ToolNetError> {
        let response = self
            .request::<Vec<u8>>(Method::GET, url, query, None)
            .await?;
        String::from_utf8(response).map_err(|e| ToolNetError::Parse(e.to_string()))
    }
    pub async fn post_json<T: DeserializeOwned, B: Serialize>(
        &self,
        url: &str,
        body: &B,
    ) -> Result<T, ToolNetError> {
        self.request_json(Method::POST, url, &[], Some(body)).await
    }
    pub async fn post_form<T: DeserializeOwned>(
        &self,
        url: &str,
        form: &[(&str, &str)],
    ) -> Result<T, ToolNetError> {
        let bytes = self.request_form(url, form).await?;
        serde_json::from_slice(&bytes).map_err(|e| ToolNetError::Parse(e.to_string()))
    }
    pub async fn post_form_text(
        &self,
        url: &str,
        form: &[(&str, &str)],
    ) -> Result<String, ToolNetError> {
        let response = self.request_form(url, form).await?;
        String::from_utf8(response).map_err(|e| ToolNetError::Parse(e.to_string()))
    }

    /// GET with caller-supplied headers. Header names and values are validated by reqwest;
    /// secrets are never included in errors or debug output.
    pub async fn get_json_with_headers<T: DeserializeOwned>(
        &self,
        url: &str,
        query: &[(&str, &str)],
        headers: &[(&str, &str)],
    ) -> Result<T, ToolNetError> {
        let bytes = self
            .request_with_headers(
                Method::GET,
                url,
                query,
                None::<Vec<u8>>,
                "application/json",
                headers,
            )
            .await?;
        serde_json::from_slice(&bytes).map_err(|e| ToolNetError::Parse(e.to_string()))
    }

    pub async fn post_json_with_headers<T: DeserializeOwned, B: Serialize>(
        &self,
        url: &str,
        body: &B,
        headers: &[(&str, &str)],
    ) -> Result<T, ToolNetError> {
        let body = serde_json::to_vec(body).map_err(|e| ToolNetError::Content(e.to_string()))?;
        let bytes = self
            .request_with_headers(
                Method::POST,
                url,
                &[],
                Some(body),
                "application/json",
                headers,
            )
            .await?;
        serde_json::from_slice(&bytes).map_err(|e| ToolNetError::Parse(e.to_string()))
    }

    async fn request_json<T: DeserializeOwned, B: Serialize + Clone>(
        &self,
        method: Method,
        url: &str,
        query: &[(&str, &str)],
        body: Option<B>,
    ) -> Result<T, ToolNetError> {
        let bytes = self.request_body(method, url, query, body).await?;
        serde_json::from_slice(&bytes).map_err(|e| ToolNetError::Parse(e.to_string()))
    }
    async fn request_form(
        &self,
        url: &str,
        form: &[(&str, &str)],
    ) -> Result<Vec<u8>, ToolNetError> {
        let encoded = url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(form.iter().copied())
            .finish();
        self.request_with_type(
            Method::POST,
            url,
            &[],
            Some(encoded),
            "application/x-www-form-urlencoded",
        )
        .await
    }
    async fn request_body<B: Serialize + Clone>(
        &self,
        method: Method,
        url: &str,
        query: &[(&str, &str)],
        body: Option<B>,
    ) -> Result<Vec<u8>, ToolNetError> {
        let json = body
            .map(|b| serde_json::to_vec(&b).map_err(|e| ToolNetError::Content(e.to_string())))
            .transpose()?;
        self.request(method, url, query, json).await
    }
    async fn request<B: Into<reqwest::Body> + Clone>(
        &self,
        method: Method,
        url: &str,
        query: &[(&str, &str)],
        body: Option<B>,
    ) -> Result<Vec<u8>, ToolNetError> {
        self.request_with_type(method, url, query, body, "application/json")
            .await
    }
    async fn request_with_type<B: Into<reqwest::Body> + Clone>(
        &self,
        method: Method,
        url: &str,
        query: &[(&str, &str)],
        body: Option<B>,
        content_type: &str,
    ) -> Result<Vec<u8>, ToolNetError> {
        self.request_with_headers(method, url, query, body, content_type, &[])
            .await
    }
    async fn request_with_headers<B: Into<reqwest::Body> + Clone>(
        &self,
        method: Method,
        url: &str,
        query: &[(&str, &str)],
        body: Option<B>,
        content_type: &str,
        headers: &[(&str, &str)],
    ) -> Result<Vec<u8>, ToolNetError> {
        let started = Instant::now();
        let mut attempt = 0;
        loop {
            let _slot = self.limits.slots.take().await;
            let _rate = self.limits.limiter.until_ready().await;
            let mut req = self
                .client
                .request(method.clone(), url)
                .query(query)
                .header("user-agent", self.user_agent());
            for (name, value) in headers {
                req = req.header(*name, *value);
            }
            if let Some(body) = &body {
                req = req.header("content-type", content_type).body(body.clone());
            }
            let result = req.send().await;
            match result {
                Ok(response) => match self.read_response(response).await {
                    Ok(bytes) => return Ok(bytes),
                    Err(error) => {
                        if attempt == 0 && error.is_retryable() {
                            let delay = error.retry_after().unwrap_or(self.retry_backoff);
                            if delay.is_zero()
                                || started.elapsed().saturating_add(delay) <= self.retry_budget
                            {
                                if !delay.is_zero() {
                                    sleep(delay).await;
                                }
                                attempt += 1;
                                continue;
                            }
                        }
                        return Err(error);
                    }
                },
                Err(e) => {
                    let error = if e.is_timeout() {
                        ToolNetError::Timeout
                    } else {
                        ToolNetError::Network(e.to_string())
                    };
                    if attempt == 0 && self.retry_backoff.is_zero() {
                        return Err(error);
                    }
                    if attempt == 0
                        && started.elapsed().saturating_add(self.retry_backoff) <= self.retry_budget
                    {
                        sleep(self.retry_backoff).await;
                        attempt += 1;
                        continue;
                    }
                    return Err(error);
                }
            }
        }
    }
    async fn read_response(&self, response: Response) -> Result<Vec<u8>, ToolNetError> {
        let status = response.status().as_u16();
        let retry_after = ToolNetError::parse_retry_after(
            response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
        );
        let mut stream = response.bytes_stream();
        let mut body = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ToolNetError::Network(e.to_string()))?;
            if body.len().saturating_add(chunk.len()) > self.max_body {
                return Err(ToolNetError::BodyTooLarge {
                    limit: self.max_body,
                });
            }
            body.extend_from_slice(&chunk);
        }
        if !(200..300).contains(&status) {
            return Err(ToolNetError::HttpStatus {
                status,
                body: String::from_utf8_lossy(&body).into_owned(),
                retry_after,
            });
        }
        Ok(body)
    }
    fn user_agent(&self) -> &str {
        self.user_agent.as_str()
    }
}

/// Reusable provider transports. Clones retain the same semaphore and governor
/// state, so adapters registered under one key share both limits.
#[derive(Default, Clone)]
pub struct LimiterPools {
    limits: HashMap<String, Arc<SharedLimits>>,
}

impl LimiterPools {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, provider: impl Into<String>, http: BackendHttp) {
        self.limits.insert(provider.into(), http.limits);
    }

    pub fn insert_shared(
        &mut self,
        family: impl Into<String>,
        concurrency: usize,
        min_interval_ms: u64,
    ) {
        self.limits.insert(
            family.into(),
            Arc::new(SharedLimits::new(concurrency, min_interval_ms)),
        );
    }

    pub fn shared(&self, family: &str) -> Option<Arc<SharedLimits>> {
        self.limits.get(family).cloned()
    }
}
