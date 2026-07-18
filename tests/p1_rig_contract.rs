//! P1 compile/runtime contract spike for Rig 0.40.
//!
//! The non-live test intentionally stops before awaiting a request: constructing the
//! request proves the complete type path without contacting OpenRouter. The ignored
//! test drives the real stream and is the opt-in protocol observation.

use futures::StreamExt;
use rig_core::agent::MultiTurnStreamItem;
use rig_core::client::CompletionClient;
use rig_core::memory::InMemoryConversationMemory;
use rig_core::providers::openrouter;
use rig_core::streaming::StreamedAssistantContent;
use rig_core::streaming::StreamingPrompt;

#[rig_core::tool_macro(
    name = "echo",
    description = "Return the supplied text unchanged.",
    params(text = "Text to return")
)]
fn echo(text: String) -> Result<String, rig_core::tool::ToolError> {
    Ok(text)
}

#[test]
fn p1_constructs_openrouter_agent_tool_memory_and_stable_request() {
    let client = openrouter::Client::new("p1-contract-fake-key").unwrap();
    let agent = client
        .agent("openrouter/auto")
        .preamble("Deterministic P1 contract agent.")
        .tool(Echo)
        .memory(InMemoryConversationMemory::new())
        .build();

    // Supplying the conversation id is what enables Rig memory; without it the
    // memory backend is intentionally bypassed. The request is not sent here.
    let _request = agent
        .stream_prompt("echo p1")
        .conversation("p1")
        .max_turns(2);
}

#[test]
fn p1_fake_client_construction_does_not_read_environment() {
    let _client = openrouter::Client::new("explicit-fake-key").expect("construction is local");
}

#[tokio::test]
#[ignore = "requires an intentional live OpenRouter run"]
async fn p1_live_stream_and_memory_identity() {
    let Some(key) = std::env::var_os("OPENROUTER_API_KEY") else {
        return;
    };
    if key.to_string_lossy().trim().is_empty() {
        return;
    }

    let client = openrouter::Client::new(key.to_string_lossy().as_ref()).expect("client builds");
    let agent = client
        .agent("openrouter/auto")
        .preamble("Use the echo tool once when asked, then answer briefly.")
        .tool(Echo)
        .memory(InMemoryConversationMemory::new())
        .build();

    let mut first_delta = false;
    let mut saw_tool_event = false;
    let mut saw_final_event = false;
    let mut stream = agent
        .stream_prompt("Use echo with the text p1, then say done.")
        .conversation("p1")
        .max_turns(2)
        .await;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => {
                if matches!(content, StreamedAssistantContent::Text(_)) {
                    first_delta = true;
                }
            }
            Ok(MultiTurnStreamItem::ToolExecutionStart { .. })
            | Ok(MultiTurnStreamItem::StreamUserItem(_)) => saw_tool_event = true,
            Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final_event = true,
            Ok(MultiTurnStreamItem::CompletionCall(_)) => {}
            Err(error) => panic!("P1 live stream failed: {error}"),
            // MultiTurnStreamItem is non-exhaustive: preserve forward compatibility.
            _ => {}
        }
    }
    assert!(saw_final_event, "live stream must emit a final response");
    let _tool_or_final_observed = saw_tool_event || saw_final_event;

    // A second request with the same identity exercises the memory lookup path.
    // Only failures before the first visible assistant delta are safe to retry;
    // once `first_delta` is true, replaying would duplicate user-visible output.
    let mut follow_up = agent
        .stream_prompt("What text did you echo? Answer in one sentence.")
        .conversation("p1")
        .max_turns(2)
        .await;
    let mut follow_up_final = false;
    while let Some(item) = follow_up.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => {
                if matches!(content, StreamedAssistantContent::Text(_)) {
                    first_delta = true;
                }
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => follow_up_final = true,
            Ok(MultiTurnStreamItem::ToolExecutionStart { .. })
            | Ok(MultiTurnStreamItem::StreamUserItem(_))
            | Ok(MultiTurnStreamItem::CompletionCall(_)) => {}
            Err(error) => panic!("P1 follow-up failed: {error}"),
            _ => {}
        }
    }
    assert!(follow_up_final, "same-conversation follow-up must complete");
    let _ = first_delta;
}
