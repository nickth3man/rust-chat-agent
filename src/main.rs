//! OpenRouter chat agent — streaming CLI with token-aware history and retry.
//!
//! Architecture:
//! - `config` (this file): loads .env via dotenvy with asymmetric precedence.
//! - `client`:  transport — streaming SSE, HTTP status validation, retry.
//! - `conversation`: bounded, token-aware message history.
//! - `error`:   typed errors with retryability classification.

mod client;
mod conversation;
mod error;

use client::ChatClient;
use conversation::Conversation;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::io::{self, Write};
use std::time::Duration;

// ── Shared message type ────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

// ── Config defaults ────────────────────────────────────────────────────

/// Default context token budget for history trimming.
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 8_000;
/// Default maximum retry attempts for retryable failures.
const DEFAULT_MAX_RETRIES: u32 = 3;
/// Default request timeout.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// Tokens reserved for the model's response (held back from history budget).
const RESPONSE_RESERVE_TOKENS: usize = 1_024;

// ── .env loading (asymmetric precedence) ───────────────────────────────

fn load_dotenv() {
    // Resolve .env relative to CARGO_MANIFEST_DIR (agent-projects/rust -> agent-projects/.env)
    let dotenv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR has no parent")
        .join(".env");

    // Scan .env to discover which keys it defines (especially OPENROUTER_MODEL)
    let mut dotenv_model: Option<String> = None;
    if let Ok(contents) = std::fs::read_to_string(&dotenv_path) {
        for line in contents.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            if let Some(eq) = t.find('=') {
                let key = t[..eq].trim();
                let value = t[eq + 1..].trim().to_string();
                if key == "OPENROUTER_MODEL" {
                    dotenv_model = Some(value);
                }
            }
        }
    }

    // Load via dotenvy — standard behavior: does NOT override existing env vars.
    // This means OPENROUTER_API_KEY from the live environment always wins.
    let _ = dotenvy::from_path(&dotenv_path);

    // OPENROUTER_MODEL: .env is authoritative. Re-apply so it beats any ambient value.
    if let Some(model) = dotenv_model {
        std::env::set_var("OPENROUTER_MODEL", model);
    }
}

/// Read an optional numeric env var, falling back to a default.
fn env_or_default(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

// ── Entry point ────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    load_dotenv();

    // OPENROUTER_API_KEY: live env always wins (.env only fills the gap).
    let api_key = match env::var("OPENROUTER_API_KEY") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => {
            eprintln!("Set OPENROUTER_API_KEY before running this program.");
            std::process::exit(1);
        }
    };

    // OPENROUTER_MODEL: .env authoritative; fallback to openrouter/auto.
    let model = env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "openrouter/auto".to_string());

    let timeout_secs: u64 =
        env_or_default("OPENROUTER_TIMEOUT_SECS", &DEFAULT_TIMEOUT_SECS.to_string())
            .parse()
            .unwrap_or(DEFAULT_TIMEOUT_SECS);
    let max_retries: u32 =
        env_or_default("OPENROUTER_MAX_RETRIES", &DEFAULT_MAX_RETRIES.to_string())
            .parse()
            .unwrap_or(DEFAULT_MAX_RETRIES);
    let max_context_tokens: usize = env_or_default(
        "OPENROUTER_MAX_CONTEXT_TOKENS",
        &DEFAULT_MAX_CONTEXT_TOKENS.to_string(),
    )
    .parse()
    .unwrap_or(DEFAULT_MAX_CONTEXT_TOKENS);

    let http = Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()?;
    let chat_client = ChatClient::new(http, api_key, model.clone(), max_retries);

    let mut conversation = Conversation::new(
        "You are a helpful, concise assistant.".to_string(),
        max_context_tokens,
        RESPONSE_RESERVE_TOKENS,
    );

    println!("OpenRouter chat agent using {model}. Type 'exit' to quit.");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            println!();
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }

        conversation.push(Message {
            role: "user".to_string(),
            content: input.to_string(),
        });

        // Stream the completion, printing each delta as it arrives.
        print!("Assistant: ");
        io::stdout().flush()?;
        let messages = conversation.messages().to_vec();
        match chat_client
            .stream_complete(&messages, |delta| {
                print!("{delta}");
                let _ = io::stdout().flush();
            })
            .await
        {
            Ok(answer) => {
                println!("\n");
                conversation.push(Message {
                    role: "assistant".to_string(),
                    content: answer,
                });
            }
            Err(error) => {
                eprintln!("\nError: {error}");
            }
        }
    }

    Ok(())
}
