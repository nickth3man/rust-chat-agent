// OpenRouter chat-completions helpers (OpenAI-compatible).
// Docs: https://openrouter.ai/docs/api-reference/overview
//
// Every call here is driven through `OpenRouterCtx`, which bundles the URL,
// bearer key, journal path, and retry knobs — all injectable so tests can
// point at a wiremock server instead of the real OpenRouter endpoint.

use crate::http::{journal_retry_and_sleep, post_json};
use crate::journal::journal;
use crate::{extract_answer_text, parse_rewrite_tool_arguments, Config, RewriteToolReject};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

// --- OpenRouter response types ---

/// `OpenRouter` (OpenAI-compatible) chat-completions response. Only the
/// fields we read are typed; the rest is ignored by serde.
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub function: ToolFunction,
}

#[derive(Debug, Deserialize)]
pub struct ToolFunction {
    /// JSON-encoded arguments string (per `OpenAI` tool-call spec).
    pub arguments: String,
}

/// Everything an `OpenRouter` call needs, bundled so call sites don't blow past
/// clippy's argument-count lint. Borrowed, so callers build one per `run()`
/// and reuse it across the rewrite/answer/re-search/citation-retry calls.
pub struct OpenRouterCtx<'a> {
    pub client: &'a reqwest::Client,
    pub url: &'a str,
    pub api_key: &'a str,
    pub journal_path: &'a str,
    pub network_max_attempts: u32,
    pub skip_sleep: bool,
}

pub fn first_message(resp: &ChatResponse) -> Option<&ChatMessage> {
    resp.choices.first().map(|c| &c.message)
}

/// Non-empty trimmed reasoning text from the first choice, if any.
pub fn reasoning_text(resp: &ChatResponse) -> Option<&str> {
    first_message(resp)
        .and_then(|m| m.reasoning.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Reasoning/thinking models return their chain-of-thought in the `reasoning`
/// field. When the model is configured as reasoning, journal that text if the
/// provider supplied one. No-op for non-reasoning models.
pub fn journal_reasoning(resp: &ChatResponse, config: &Config, journal_path: &str) {
    if !config.reasoning {
        return;
    }
    if let Some(text) = reasoning_text(resp) {
        journal(journal_path, json!({ "event": "reasoning", "text": text }));
    }
}

/// Common chat-completions body (model, temperature, `max_tokens`, messages)
/// shared by every LLM call. Adds the empty `reasoning` object that
/// reasoning-capable models expect when configured.
pub fn chat_body(config: &Config, max_tokens: u64, system: &str, user: &str) -> Value {
    let mut body = json!({
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": max_tokens,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ]
    });
    if config.reasoning {
        body["reasoning"] = json!({});
    }
    body
}

/// POST a chat-completions request body to `OpenRouter` and return the parsed
/// response. Delegates HTTP plumbing to `post_json` and deserializes the
/// body into a typed `ChatResponse`.
pub async fn openrouter_call(ctx: &OpenRouterCtx<'_>, body: &Value) -> Result<ChatResponse> {
    let v = post_json(
        ctx.client,
        ctx.url,
        ctx.api_key,
        body,
        ctx.journal_path,
        ctx.network_max_attempts,
        ctx.skip_sleep,
    )
    .await?;
    serde_json::from_value(v).context("failed to parse OpenRouter chat response")
}

/// Answering LLM call: returns the text reply from a plain chat completion.
/// Retries on empty `content` (Class B) — common with reasoning models that
/// occasionally finish their chain-of-thought but emit no final answer, or
/// transiently return an entirely empty turn. Bounded by `max_attempts`;
/// `max_attempts == 0` falls through to the final `bail!` untested-path
/// guard, matching `post_json`.
pub async fn llm(
    ctx: &OpenRouterCtx<'_>,
    config: &Config,
    system: &str,
    user: &str,
    max_attempts: u32,
) -> Result<String> {
    for attempt in 0..max_attempts {
        let resp: ChatResponse =
            openrouter_call(ctx, &chat_body(config, 1500, system, user)).await?;
        if let Some(text) =
            extract_answer_text(first_message(&resp).and_then(|m| m.content.as_deref()))
        {
            journal_reasoning(&resp, config, ctx.journal_path);
            return Ok(text);
        }
        // Content is empty/None. Distinguish "reasoning-only" (reasoning is
        // present, content is empty — common stochastic failure mode) from a
        // fully empty turn (both empty — could be a transient blank response,
        // worth one or two retries).
        let reason = if reasoning_text(&resp).is_some() {
            "reasoning-only"
        } else {
            "empty"
        };
        let n = attempt + 1;
        if n >= max_attempts {
            return Err(anyhow::Error::msg("no text in LLM response"))
                .context(format!("after {n} attempts"));
        }
        journal_retry_and_sleep(
            ctx.journal_path,
            "empty_answer_retry",
            n,
            reason,
            ctx.skip_sleep,
        )
        .await;
    }
    bail!("internal: llm retry loop exited without returning")
}

// ---------------------------------------------------------------------------
// Tool schema for the query rewrite step. Forces the model to respond with a
// structured tool call instead of free-form prose. Built with `json!` (not
// `from_str` on a string constant) so there is no infallible-parse `?` error
// region left uncovered.
// ---------------------------------------------------------------------------
pub fn rewrite_tool_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "generate_search_query",
            "description": "Generate a short web search query from the user's question",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (plain natural language, 1-6 words)"
                    }
                },
                "required": ["query"]
            }
        }
    })
}

/// Pull the search query out of a forced `generate_search_query` tool call.
pub fn rewrite_query_from_response(resp: &ChatResponse) -> Result<String, RewriteToolReject> {
    let arguments = first_message(resp)
        .and_then(|m| m.tool_calls.as_ref())
        .and_then(|tcs| tcs.first())
        .map(|tc| tc.function.arguments.as_str());
    parse_rewrite_tool_arguments(arguments)
}

/// Forced-tool-call LLM for query rewriting. Sends `tool_choice: "required"`
/// so the model must respond with a `generate_search_query` tool call.
/// Retries on any tool-call-shape failure (Class A) — no tool call at all,
/// unparseable arguments, missing `query`, or blank `query` — bounded by
/// `max_attempts`. Reasoning-capable models sometimes ignore
/// `tool_choice: "required"` and answer in prose instead.
pub async fn rewrite_llm(
    ctx: &OpenRouterCtx<'_>,
    config: &Config,
    system: &str,
    user: &str,
    max_attempts: u32,
) -> Result<String> {
    let mut body = chat_body(config, 250, system, user);
    body["tools"] = json!([rewrite_tool_schema()]);
    body["tool_choice"] = json!("required");

    for attempt in 0..max_attempts {
        let resp = openrouter_call(ctx, &body).await?;
        match rewrite_query_from_response(&resp) {
            Ok(query) => {
                journal_reasoning(&resp, config, ctx.journal_path);
                return Ok(query);
            }
            Err(fail) => {
                let n = attempt + 1;
                if n >= max_attempts {
                    return Err(fail.into_final_error(n));
                }
                journal_retry_and_sleep(
                    ctx.journal_path,
                    "rewrite_retry",
                    n,
                    fail.reason(),
                    ctx.skip_sleep,
                )
                .await;
            }
        }
    }
    bail!("internal: rewrite retry loop exited without returning")
}
