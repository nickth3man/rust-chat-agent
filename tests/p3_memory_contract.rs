//! P3 offline contract spike for Rig 0.40 / rig-memory 0.40.
//!
//! `CompactingMemory` compacts only when its policy demotes messages during
//! `load`; there is no separate `compact()` method. The control wrapper below
//! therefore uses `load` for manual compaction and exposes `clear`/`forget`.

use rig_core::OneOrMany;
use rig_core::client::CompletionClient;
use rig_core::completion::Message;
use rig_core::memory::ConversationMemory;
use rig_core::message::{AssistantContent, UserContent};
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_memory::{
    CompactingMemory, Compactor, HeuristicTokenCounter, InMemoryConversationMemory, MemoryError,
    MemoryPolicy, TokenWindowMemory,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// A deterministic local compactor with the exact 0.40 `Compactor` contract.
/// It models one failed summarizer attempt, one retry, then an explicit fallback
/// artifact, so the agent-facing memory load never receives a `MemoryError`.
#[derive(Clone, Default)]
struct RetryThenFallbackCompactor {
    attempts: Arc<AtomicUsize>,
}

impl Compactor for RetryThenFallbackCompactor {
    type Artifact = Message;

    fn compact<'a>(
        &'a self,
        conversation_id: &'a str,
        evicted: &'a [Message],
        carry_over: Option<&'a Self::Artifact>,
    ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
        Box::pin(async move {
            let first_attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if first_attempt == 0 {
                // The real implementation would retry its summarizer here.
                let retry_attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
                let _ = retry_attempt;
                return Ok(Message::System {
                    content: format!(
                        "fallback summary for {conversation_id}: dropped {} messages",
                        evicted.len()
                    ),
                });
            }

            Ok(Message::System {
                content: format!(
                    "summary: prior={} dropped={}",
                    carry_over.is_some(),
                    evicted.len()
                ),
            })
        })
    }
}

type P3Memory =
    CompactingMemory<InMemoryConversationMemory, TokenWindowMemory, RetryThenFallbackCompactor>;

fn memory_with(compactor: RetryThenFallbackCompactor) -> Arc<P3Memory> {
    Arc::new(CompactingMemory::new(
        InMemoryConversationMemory::new(),
        TokenWindowMemory::new(2, HeuristicTokenCounter::default()),
        compactor,
    ))
}

#[derive(Clone)]
struct MemoryControl {
    memory: Arc<P3Memory>,
}

impl MemoryControl {
    /// Manual compaction is a policy-triggered load in rig-memory 0.40.
    async fn compact(&self, conversation_id: &str) -> Result<Vec<Message>, MemoryError> {
        self.memory.load(conversation_id).await
    }

    async fn clear(&self, conversation_id: &str) -> Result<(), MemoryError> {
        self.memory.clear(conversation_id).await
    }

    fn forget(&self, conversation_id: &str) {
        self.memory.forget(conversation_id);
    }
}

fn history() -> Vec<Message> {
    vec![
        Message::User {
            content: OneOrMany::one(UserContent::Text("first question".into())),
        },
        Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::Text("first answer".into())),
        },
        Message::User {
            content: OneOrMany::one(UserContent::Text("latest question".into())),
        },
    ]
}

#[tokio::test]
async fn custom_compactor_retry_fallback_and_artifact_compile() {
    let compactor = RetryThenFallbackCompactor::default();
    let attempts = compactor.attempts.clone();
    let memory = memory_with(compactor);
    memory.append("p3", history()).await.unwrap();

    let loaded = memory.load("p3").await.unwrap();
    assert!(
        matches!(loaded.first(), Some(Message::System { content }) if content.contains("fallback"))
    );
    assert_eq!(
        attempts.load(Ordering::SeqCst),
        2,
        "one failed attempt plus one retry"
    );
}

#[tokio::test]
async fn control_handle_reaches_agent_memory_clear_forget_and_manual_compact() {
    let memory = memory_with(RetryThenFallbackCompactor::default());
    let control = MemoryControl {
        memory: memory.clone(),
    };

    // This is the composition supplied to an Agent; no prompt is sent.
    let client = rig_core::providers::openrouter::Client::new("p3-fake-key").unwrap();
    let _agent = client
        .agent("openrouter/auto")
        .preamble("P3 offline agent")
        .memory(memory.clone())
        .build();

    memory.append("p3", history()).await.unwrap();
    let compacted = control.compact("p3").await.unwrap();
    assert!(compacted.iter().any(
        |message| matches!(message, Message::System { content } if content.contains("fallback"))
    ));

    control.clear("p3").await.unwrap();
    assert!(memory.load("p3").await.unwrap().is_empty());
    control.forget("p3");
    assert_eq!(memory.tracked_conversations(), 0);
}

#[test]
fn tool_turn_grouping_seam_is_explicit() {
    // Rig's ConversationMemory append receives the complete successful turn,
    // including assistant tool calls and user tool results; TokenWindowMemory
    // also drops a leading orphan tool result when truncating. It does not,
    // however, create an evidence ledger or rewrite raw tool payloads. Production
    // needs a custom MemoryPolicy (or a wrapper around ConversationMemory::append)
    // to group complete turns and replace tool payloads with ledger entries before
    // CompactingMemory applies its policy.
    let policy = TokenWindowMemory::new(1, |_message: &Message| 1);
    let kept = policy.apply(history()).unwrap();
    assert_eq!(kept.len(), 1);
}

#[tokio::test]
async fn template_compactor_is_an_offline_alternative() {
    let memory = CompactingMemory::new(
        InMemoryConversationMemory::new(),
        TokenWindowMemory::new(2, HeuristicTokenCounter::default()),
        rig_memory::TemplateCompactor::with_header("offline").with_max_bytes(256),
    );
    memory.append("template", history()).await.unwrap();
    let loaded = memory.load("template").await.unwrap();
    assert!(
        matches!(loaded.first(), Some(Message::System { content }) if content.starts_with("offline"))
    );
}
