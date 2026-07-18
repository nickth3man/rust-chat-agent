//! Opt-in live checks for the three model IDs effective in the local config.
//!
//! These tests are ignored by default because they spend OpenRouter credits.  When
//! deliberately enabled, configuration and credentials are required rather than
//! being treated as a skip condition.

use futures::StreamExt;
use openrouter_chat_rust::{
    config,
    rank::{RankCandidate, Ranker, RankingRequest, RigRanker},
    summarize::{RigSummarizer, Summarizer},
};
use rig_core::{
    agent::MultiTurnStreamItem,
    client::CompletionClient,
    completion::Message,
    memory::InMemoryConversationMemory,
    providers::openrouter,
    streaming::{StreamedUserContent, StreamingPrompt},
};
use std::{collections::BTreeMap, time::Duration};

#[rig_core::tool_macro(
    name = "echo",
    description = "Return the supplied text unchanged.",
    params(text = "Text to return")
)]
fn echo(text: String) -> Result<String, rig_core::tool::ToolError> {
    Ok(text)
}

fn live_config() -> config::ResolvedConfig {
    config::load().expect("live integration configuration/OPENROUTER_API_KEY is required")
}

#[tokio::test]
#[ignore = "requires an intentional live OpenRouter run"]
async fn configured_chat_stream_executes_echo_and_finishes() {
    let resolved = live_config();
    let client = openrouter::Client::new(&resolved.openrouter_api_key)
        .expect("OpenRouter client must accept the configured credential");
    let agent = client
        .agent(resolved.public.models.chat_id)
        .preamble("When asked, call the echo tool exactly once with the supplied text, then answer briefly.")
        .tool(Echo)
        .memory(InMemoryConversationMemory::new())
        .build();

    let mut stream = agent
        .stream_prompt("Call echo with exactly ORBITAL-ECHO-17, then say done.")
        .conversation("configured-models-live-chat")
        .max_turns(2)
        .await;
    let mut execution_id = None;
    let mut matching_result = false;
    let mut final_response = false;
    tokio::time::timeout(Duration::from_secs(30), async {
        while let Some(item) = stream.next().await {
            match item.expect("configured chat stream must not fail") {
                MultiTurnStreamItem::ToolExecutionStart {
                    internal_call_id, ..
                } => execution_id = Some(internal_call_id),
                MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    internal_call_id,
                    ..
                }) => {
                    matching_result = execution_id.as_deref() == Some(internal_call_id.as_str());
                }
                MultiTurnStreamItem::FinalResponse(_) => final_response = true,
                _ => {}
            }
        }
    })
    .await
    .expect("configured chat stream must finish within 30 seconds");
    assert!(execution_id.is_some(), "chat must emit ToolExecutionStart");
    assert!(
        matching_result,
        "tool result must match the execution call ID"
    );
    assert!(final_response, "chat must emit FinalResponse");
}

#[tokio::test]
#[ignore = "requires an intentional live OpenRouter run"]
async fn configured_ranker_returns_each_candidate_once_and_selects_one() {
    let resolved = live_config();
    let client = openrouter::Client::new(&resolved.openrouter_api_key)
        .expect("OpenRouter client must accept the configured credential");
    let ranker = RigRanker::new(
        client,
        resolved.public.models.rank_id,
        Duration::from_secs(resolved.public.search.rank_timeout_secs),
    );
    let candidates = ["candidate-alpha", "candidate-beta"]
        .into_iter()
        .map(|candidate_id| RankCandidate {
            candidate_id: candidate_id.into(),
            hit: openrouter_chat_rust::contracts::types::SearchHit {
                title: format!("{candidate_id} title"),
                url: format!("https://{candidate_id}.example.invalid"),
                snippet: format!("Evidence for {candidate_id}"),
                published: None,
                native_rank: None,
                native_score: None,
                provider: "live-fixture".into(),
                backend_kind: openrouter_chat_rust::contracts::types::BackendKind::Web,
                source_subtype: None,
                metadata: BTreeMap::new(),
            },
        })
        .collect::<Vec<_>>();
    let result = ranker
        .rank(RankingRequest {
            query: "choose the stronger of two fixed candidates".into(),
            candidates,
        })
        .await
        .expect("configured ranker live request must succeed");
    let ids = result
        .decisions
        .iter()
        .map(|decision| decision.source_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 2);
    assert_eq!(ids.iter().filter(|id| **id == "candidate-alpha").count(), 1);
    assert_eq!(ids.iter().filter(|id| **id == "candidate-beta").count(), 1);
    assert!(result.decisions.iter().any(|decision| decision.selected));
}

#[tokio::test]
#[ignore = "requires an intentional live OpenRouter run"]
async fn configured_summarizer_preserves_distinctive_fact() {
    let resolved = live_config();
    let client = openrouter::Client::new(&resolved.openrouter_api_key)
        .expect("OpenRouter client must accept the configured credential");
    let summarizer = RigSummarizer::new(
        client,
        resolved.public.models.summarize_id,
        Duration::from_secs(30),
    );
    let messages = [
        Message::user("We decided to keep the migration small."),
        Message::assistant("The harmless reference fact is ORBITAL-CEDAR-17."),
    ];
    let output = summarizer
        .summarize("configured-models-live-summary", &messages)
        .await
        .expect("configured summarizer live request must succeed");
    assert!(!output.trim().is_empty());
    assert!(
        output.to_ascii_lowercase().contains("orbital-cedar-17"),
        "summary must preserve ORBITAL-CEDAR-17: {output}"
    );
}
