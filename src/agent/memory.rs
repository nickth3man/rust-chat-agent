use crate::summarize::Summarizer;
use crate::tools::meta_search::normalize_url;
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_core::{
    OneOrMany,
    completion::Message,
    memory::ConversationMemory,
    message::{AssistantContent, Text, ToolResultContent, UserContent},
};
use rig_memory::{
    CompactingMemory, Compactor, HeuristicTokenCounter, InMemoryConversationMemory, MemoryError,
    MemoryPolicy, TokenWindowMemory,
};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::broadcast;

const PRESSURE: f32 = 0.82;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryEventKind {
    Started,
    Retry,
    Completed,
    Fallback,
    Cleared,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryEvent {
    pub kind: MemoryEventKind,
    pub conversation_id: String,
    pub detail: String,
}

#[derive(Clone)]
struct Activity {
    tx: broadcast::Sender<MemoryEvent>,
    log: Arc<Mutex<Vec<MemoryEvent>>>,
}

impl Activity {
    fn new() -> Self {
        let (tx, _) = broadcast::channel(64);
        Self {
            tx,
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }
    fn emit(&self, event: MemoryEvent) {
        if let Ok(mut log) = self.log.lock() {
            log.push(event.clone());
        }
        let _ = self.tx.send(event); // no receiver is a normal operating state
    }
}

/// Replace large web-tool payloads at the append seam, while retaining the
/// assistant call and its corresponding result as one provider-valid turn.
pub fn sanitize_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut out = Vec::with_capacity(messages.len());
    let mut pending_calls = HashMap::<String, String>::new();
    for message in messages {
        match &message {
            Message::Assistant { content, .. } => {
                for item in content.iter() {
                    if let AssistantContent::ToolCall(call) = item {
                        pending_calls.insert(call.id.clone(), call.function.name.clone());
                    }
                }
                out.push(message);
            }
            Message::User { content }
                if content
                    .iter()
                    .any(|c| matches!(c, UserContent::ToolResult(_))) =>
            {
                let mut kept = Vec::new();
                for item in content.iter() {
                    match item {
                        UserContent::ToolResult(result) => {
                            let tool_name = pending_calls.remove(&result.id);
                            if tool_name.as_deref().is_some_and(is_web_tool) {
                                kept.push(UserContent::ToolResult(compact_tool_result(result)));
                            } else {
                                // Non-web tools are deliberately lossless.
                                kept.push(item.clone());
                            }
                        }
                        other => kept.push(other.clone()),
                    }
                }
                if !kept.is_empty() {
                    out.push(Message::User {
                        content: OneOrMany::many(kept).expect("non-empty"),
                    });
                }
            }
            _ => out.push(message),
        }
    }
    out
}

fn compact_tool_result(result: &rig_core::message::ToolResult) -> rig_core::message::ToolResult {
    let raw = result
        .content
        .iter()
        .filter_map(|part| match part {
            ToolResultContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let value: serde_json::Value = serde_json::from_str(&raw).unwrap_or(serde_json::Value::Null);
    let body = compact_web_json(&value);
    let mut compact = result.clone();
    compact.content = OneOrMany::one(ToolResultContent::Text(Text::new(body)));
    compact
}

fn compact_web_json(value: &serde_json::Value) -> String {
    let mut lines = Vec::new();
    if let Some(selected) = value.get("selected").and_then(|v| v.as_array()) {
        let ranked = value
            .get("ranked_evidence")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        for hit in selected {
            let title = hit.get("title").and_then(|v| v.as_str()).unwrap_or("");
            let url = hit.get("url").and_then(|v| v.as_str()).unwrap_or("");
            let snippet = hit.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
            let providers = hit.get("providers").map(compact_value).unwrap_or_default();
            let normalized_url = normalize_url(url);
            let decision = ranked.iter().find(|entry| {
                entry.get("normalized_url").and_then(|v| v.as_str())
                    == Some(normalized_url.as_str())
            });
            let decision = decision
                .and_then(|entry| entry.get("decision"))
                .map(compact_value)
                .unwrap_or_else(|| "unknown".into());
            lines.push(format!(
                "title={} url={} snippet={} provider={} rank_decision={}",
                truncate(title, 160),
                truncate(url, 300),
                truncate(snippet, 400),
                truncate(&providers, 160),
                truncate(&decision, 120)
            ));
        }
    } else if value.get("content").is_some() || value.get("url").is_some() {
        let url = value.get("url").and_then(|v| v.as_str()).unwrap_or("");
        let content = value.get("content").and_then(|v| v.as_str()).unwrap_or("");
        lines.push(format!(
            "url={} summary_excerpt={}",
            truncate(url, 300),
            short_excerpt(&strip_markup(content))
        ));
    }
    if lines.is_empty() {
        "web evidence retained; raw tool payload omitted".into()
    } else {
        lines.join("\n")
    }
}

fn compact_value(value: &serde_json::Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn strip_markup(value: &str) -> String {
    value
        .replace("<web_content", "")
        .replace("</web_content>", "")
        .split('>')
        .next_back()
        .unwrap_or(value)
        .trim()
        .to_string()
}

/// Keep a useful lead without allowing a fetched document's later body to
/// leak back into conversation memory.  `char_indices`-free `chars().take`
/// truncation is Unicode-safe and intentionally keeps this ledger bounded.
fn short_excerpt(value: &str) -> String {
    const MAX_EXCERPT_CHARS: usize = 32;
    truncate(value.trim(), MAX_EXCERPT_CHARS)
}

fn is_web_tool(name: &str) -> bool {
    matches!(name, "meta_search" | "fetch_page")
}

fn truncate(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

struct PressurePolicy {
    window: TokenWindowMemory,
    force: Arc<AtomicBool>,
}

impl MemoryPolicy for PressurePolicy {
    fn apply(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        Ok(self.apply_with_demoted(messages)?.0)
    }
    fn apply_with_demoted(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Vec<Message>, Vec<Message>), MemoryError> {
        if !self.force.load(Ordering::Acquire) {
            return self.window.apply_with_demoted(messages);
        }
        if messages.len() <= 1 {
            return Ok((messages, Vec::new()));
        }
        let split = messages.len() - 1;
        let mut demoted = messages[..split].to_vec();
        let mut kept = messages[split..].to_vec();
        if matches!(kept.first(), Some(Message::User { content }) if matches!(content.first_ref(), UserContent::ToolResult(_)))
        {
            demoted.append(&mut kept);
        }
        Ok((kept, demoted))
    }
}

#[derive(Clone)]
pub struct LlmCompactor {
    summarizer: Arc<dyn Summarizer>,
    activity: Activity,
}

impl LlmCompactor {
    fn new(summarizer: Arc<dyn Summarizer>, activity: Activity) -> Self {
        Self {
            summarizer,
            activity,
        }
    }
}

impl Compactor for LlmCompactor {
    type Artifact = Message;
    fn compact<'a>(
        &'a self,
        conversation_id: &'a str,
        evicted: &'a [Message],
        carry_over: Option<&'a Message>,
    ) -> WasmBoxedFuture<'a, Result<Message, MemoryError>> {
        Box::pin(async move {
            self.activity.emit(MemoryEvent {
                kind: MemoryEventKind::Started,
                conversation_id: conversation_id.into(),
                detail: format!("{} messages", evicted.len()),
            });
            let mut input = Vec::with_capacity(evicted.len() + usize::from(carry_over.is_some()));
            if let Some(previous) = carry_over {
                input.push(previous.clone());
            }
            input.extend_from_slice(evicted);
            match self.summarizer.summarize(conversation_id, &input).await {
                Ok(text) => {
                    self.activity.emit(MemoryEvent {
                        kind: MemoryEventKind::Completed,
                        conversation_id: conversation_id.into(),
                        detail: "summary created".into(),
                    });
                    Ok(Message::System { content: text })
                }
                Err(first) => {
                    self.activity.emit(MemoryEvent {
                        kind: MemoryEventKind::Retry,
                        conversation_id: conversation_id.into(),
                        detail: first.to_string(),
                    });
                    match self.summarizer.summarize(conversation_id, &input).await {
                        Ok(text) => {
                            self.activity.emit(MemoryEvent {
                                kind: MemoryEventKind::Completed,
                                conversation_id: conversation_id.into(),
                                detail: "summary retry succeeded".into(),
                            });
                            Ok(Message::System { content: text })
                        }
                        Err(second) => {
                            self.activity.emit(MemoryEvent {
                                kind: MemoryEventKind::Fallback,
                                conversation_id: conversation_id.into(),
                                detail: second.to_string(),
                            });
                            Ok(Message::System {
                                content: format!(
                                    "Earlier context was compacted safely; {} messages omitted.",
                                    evicted.len()
                                ),
                            })
                        }
                    }
                }
            }
        })
    }
}

type InnerMemory = CompactingMemory<InMemoryConversationMemory, PressurePolicy, LlmCompactor>;

#[derive(Clone)]
pub struct ProductionMemory {
    inner: Arc<InnerMemory>,
    force: Arc<AtomicBool>,
    activity: Activity,
}

impl ProductionMemory {
    pub fn new(context_tokens: usize, summarizer: Arc<dyn Summarizer>) -> Self {
        Self::new_for_model(context_tokens, "unknown", summarizer)
    }

    /// Select the conservative provider-family heuristic when the model name
    /// is known; unknown families use the OpenAI-compatible default.
    pub fn new_for_model(
        context_tokens: usize,
        model_id: &str,
        summarizer: Arc<dyn Summarizer>,
    ) -> Self {
        let force = Arc::new(AtomicBool::new(false));
        let counter = if model_id.to_ascii_lowercase().contains("claude")
            || model_id.to_ascii_lowercase().contains("anthropic")
        {
            HeuristicTokenCounter::anthropic()
        } else if model_id.to_ascii_lowercase().contains("gemini")
            || model_id.to_ascii_lowercase().contains("google")
        {
            HeuristicTokenCounter::gemini()
        } else {
            HeuristicTokenCounter::default()
        };
        let policy = PressurePolicy {
            window: TokenWindowMemory::new(((context_tokens as f32) * PRESSURE) as usize, counter),
            force: force.clone(),
        };
        let activity = Activity::new();
        let inner = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            policy,
            LlmCompactor::new(summarizer, activity.clone()),
        );
        Self {
            inner: Arc::new(inner),
            force,
            activity,
        }
    }
    pub fn control(&self) -> MemoryControl {
        MemoryControl {
            memory: self.clone(),
        }
    }
    pub fn subscribe(&self) -> broadcast::Receiver<MemoryEvent> {
        self.activity.tx.subscribe()
    }
    pub fn take_events(&self) -> Vec<MemoryEvent> {
        self.activity
            .log
            .lock()
            .map(|mut l| std::mem::take(&mut *l))
            .unwrap_or_default()
    }
}

impl ConversationMemory for ProductionMemory {
    fn load<'a>(&'a self, id: &'a str) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        self.inner.load(id)
    }
    fn append<'a>(
        &'a self,
        id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        self.inner.append(id, sanitize_messages(messages))
    }
    fn clear<'a>(&'a self, id: &'a str) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        self.force.store(false, Ordering::Release);
        Box::pin(async move {
            self.inner.clear(id).await?;
            self.activity.emit(MemoryEvent {
                kind: MemoryEventKind::Cleared,
                conversation_id: id.into(),
                detail: "conversation cleared".into(),
            });
            Ok(())
        })
    }
}

#[derive(Clone)]
pub struct MemoryControl {
    memory: ProductionMemory,
}

impl MemoryControl {
    pub async fn compact(&self, id: &str) -> Result<Vec<Message>, MemoryError> {
        self.memory.force.store(true, Ordering::Release);
        let result = self.memory.load(id).await;
        self.memory.force.store(false, Ordering::Release);
        result
    }
    pub async fn clear(&self, id: &str) -> Result<(), MemoryError> {
        self.memory.clear(id).await
    }
    pub fn forget(&self, id: &str) {
        self.memory.inner.forget(id);
    }
    pub fn subscribe(&self) -> broadcast::Receiver<MemoryEvent> {
        self.memory.subscribe()
    }
    pub fn take_events(&self) -> Vec<MemoryEvent> {
        self.memory.take_events()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::error::AppError;
    use async_trait::async_trait;
    use std::sync::atomic::AtomicUsize;

    #[derive(Clone)]
    struct FakeSummarizer {
        calls: Arc<AtomicUsize>,
        failures: usize,
    }

    #[async_trait]
    impl Summarizer for FakeSummarizer {
        async fn summarize(&self, _: &str, _: &[Message]) -> Result<String, AppError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call < self.failures {
                Err(AppError::Compact("offline failure".into()))
            } else {
                Ok(format!("summary-{call}"))
            }
        }
    }

    fn memory(failures: usize, context: usize) -> (ProductionMemory, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let summarizer = Arc::new(FakeSummarizer {
            calls: calls.clone(),
            failures,
        });
        (ProductionMemory::new(context, summarizer), calls)
    }

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    #[tokio::test]
    async fn one_failure_retries_and_succeeds() {
        let (memory, calls) = memory(1, 20);
        memory
            .append(
                "c",
                vec![
                    user("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                    user("latest"),
                ],
            )
            .await
            .unwrap();
        let loaded = memory.load("c").await.unwrap();
        assert!(
            loaded
                .iter()
                .any(|m| matches!(m, Message::System { content } if content == "summary-1"))
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn two_failures_fallback_and_events_are_safe_without_subscriber() {
        let (memory, calls) = memory(2, 20);
        memory
            .append(
                "c",
                vec![
                    user("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                    user("latest"),
                ],
            )
            .await
            .unwrap();
        let loaded = memory.load("c").await.unwrap();
        assert!(
            loaded
                .iter()
                .any(|m| matches!(m, Message::System { content } if content.contains("safely")))
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let kinds = memory
            .take_events()
            .into_iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                MemoryEventKind::Started,
                MemoryEventKind::Retry,
                MemoryEventKind::Fallback
            ]
        );
    }

    #[test]
    fn web_payloads_become_compact_evidence_and_grouping_is_preserved() {
        let raw = r#"{"selected":[{"title":"Useful title","url":"https://example.test","snippet":"short evidence","providers":["alpha"]}],"ranked_evidence":[{"normalized_url":"https://example.test","decision":"keep","score":0.9}],"secret_raw":"DO NOT KEEP"}"#;
        let messages = vec![
            Message::Assistant {
                id: None,
                content: OneOrMany::one(AssistantContent::tool_call(
                    "call-1",
                    "meta_search",
                    serde_json::json!({"q":"x"}),
                )),
            },
            Message::User {
                content: OneOrMany::one(UserContent::tool_result(
                    "call-1",
                    OneOrMany::one(ToolResultContent::Text(Text::new(raw))),
                )),
            },
        ];
        let sanitized = sanitize_messages(messages);
        assert_eq!(sanitized.len(), 2);
        let rendered = serde_json::to_string(&sanitized).unwrap();
        assert!(rendered.contains("Useful title"));
        assert!(rendered.contains("rank_decision"));
        assert!(!rendered.contains("DO NOT KEEP"));
        assert!(matches!(sanitized[0], Message::Assistant { .. }));
        assert!(matches!(sanitized[1], Message::User { .. }));

        let fetch_raw = r#"{"url":"https://example.test/page","content":"short page excerpt followed by a very large private body that must not survive","original_chars":9999,"truncated":true}"#;
        let fetch_messages = vec![
            Message::Assistant {
                id: None,
                content: OneOrMany::one(AssistantContent::tool_call(
                    "call-2",
                    "fetch_page",
                    serde_json::json!({"url":"https://example.test/page"}),
                )),
            },
            Message::User {
                content: OneOrMany::one(UserContent::tool_result(
                    "call-2",
                    OneOrMany::one(ToolResultContent::Text(Text::new(fetch_raw))),
                )),
            },
        ];
        let fetch = serde_json::to_string(&sanitize_messages(fetch_messages)).unwrap();
        assert!(fetch.contains("https://example.test/page"));
        assert!(fetch.contains("short page excerpt"));
        assert!(!fetch.contains("very large private body"));
    }

    #[test]
    fn compact_evidence_joins_original_url_to_normalized_rank_decision() {
        let value = serde_json::json!({
            "selected": [{
                "title": "Useful title",
                "url": "https://www.example.test/a/?utm_source=x",
                "snippet": "short evidence",
                "providers": ["alpha"]
            }],
            "ranked_evidence": [{
                "normalized_url": "example.test/a",
                "decision": "keep"
            }]
        });

        let compact = compact_web_json(&value);

        assert!(compact.contains("rank_decision=keep"));
    }

    #[tokio::test]
    async fn forced_manual_compaction_works_below_threshold_and_threshold_compacts() {
        let (manual_memory, calls) = memory(0, 10_000);
        manual_memory
            .append("manual", vec![user("old"), user("new")])
            .await
            .unwrap();
        let control = manual_memory.control();
        let manual = control.compact("manual").await.unwrap();
        assert!(manual.iter().any(|m| matches!(m, Message::System { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let (small, _) = memory(0, 20);
        small
            .append(
                "threshold",
                vec![
                    user("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
                    user("latest"),
                ],
            )
            .await
            .unwrap();
        assert!(
            small
                .load("threshold")
                .await
                .unwrap()
                .iter()
                .any(|m| matches!(m, Message::System { .. }))
        );
    }

    #[tokio::test]
    async fn clear_and_forget_share_control_state() {
        let (memory, _) = memory(0, 20);
        let control = memory.control();
        memory
            .append("c", vec![user("old"), user("new")])
            .await
            .unwrap();
        control.clear("c").await.unwrap();
        assert!(memory.load("c").await.unwrap().is_empty());
        control.forget("c");
        assert!(memory.load("c").await.unwrap().is_empty());
        assert!(
            control
                .take_events()
                .iter()
                .any(|event| event.kind == MemoryEventKind::Cleared)
        );
    }
}
