//! Conversation history with token-aware bounded trimming.
//!
//! The system message (index 0) is always preserved. Older user/assistant
//! pairs are removed in FIFO order once the estimated token count exceeds
//! the configured budget. Trimming removes complete pairs so messages never
//! end up orphaned (an assistant reply without its preceding user turn).
//!
//! Token estimation uses a heuristic (~4 chars/token) to avoid pulling in a
//! tokenizer dependency. This is intentionally approximate; the budget has
//! built-in headroom via `reserve`.

use crate::Message;

/// Rough characters-per-token estimate for English/mixed text.
/// Real tokenizers vary, so this is conservative.
const CHARS_PER_TOKEN: usize = 4;
/// Per-message overhead (role tags, separators) in tokens.
const PER_MESSAGE_OVERHEAD: usize = 4;

/// A bounded conversation that keeps the system prompt and trims old pairs.
pub struct Conversation {
    messages: Vec<Message>,
    /// Soft cap on total context tokens (including the reserve).
    max_tokens: usize,
    /// Tokens reserved for the next model response; not available to history.
    reserve: usize,
}

impl Conversation {
    /// Create a new conversation starting with a system message.
    /// `max_tokens` is the context budget; `reserve` is held back for output.
    pub fn new(system_prompt: String, max_tokens: usize, reserve: usize) -> Self {
        Self {
            messages: vec![Message {
                role: "system".to_string(),
                content: system_prompt,
            }],
            max_tokens,
            reserve,
        }
    }

    /// Read-only view of the current message history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Append a message and trim if over budget.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
        self.trim();
    }

    /// Rough token estimate for a string.
    pub fn estimate_tokens(s: &str) -> usize {
        (s.len() / CHARS_PER_TOKEN).max(1) + PER_MESSAGE_OVERHEAD
    }

    /// Total estimated tokens across all messages.
    pub fn total_tokens(&self) -> usize {
        self.messages
            .iter()
            .map(|m| Self::estimate_tokens(&m.content))
            .sum()
    }

    /// Number of stored messages (including the system prompt).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Remove the oldest user/assistant pair (after the system message)
    /// until the budget is satisfied. The system message is never removed,
    /// and at least the system message plus the most recent turn are kept.
    fn trim(&mut self) {
        let budget = self.max_tokens.saturating_sub(self.reserve);
        // Keep system (index 0) + the latest two messages (most recent turn).
        while self.total_tokens() > budget && self.messages.len() > 3 {
            // Remove the oldest non-system message (index 1). Removing in
            // pairs keeps user/assistant turns coherent.
            self.messages.remove(1);
            if self.messages.len() > 3 {
                self.messages.remove(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }

    fn assistant(content: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn estimate_tokens_is_nonzero() {
        assert!(Conversation::estimate_tokens("") >= 1);
        assert!(Conversation::estimate_tokens("hello") >= 1);
    }

    #[test]
    fn system_message_always_preserved() {
        let mut c = Conversation::new("be concise".into(), 100, 10);
        // Push many large turns to force trimming.
        for _i in 0..50 {
            c.push(user(&"x".repeat(100)));
            c.push(assistant(&"y".repeat(100)));
        }
        assert_eq!(c.messages()[0].role, "system");
        assert_eq!(c.messages()[0].content, "be concise");
    }

    #[test]
    fn trims_when_over_budget() {
        // Small budget so a few turns overflow.
        let mut c = Conversation::new("sys".into(), 50, 10);
        for i in 0..20 {
            c.push(user(&format!("turn {i} with some padding text here")));
            c.push(assistant(&format!("reply {i} with padding text here")));
        }
        // Should have trimmed well below 40 messages.
        assert!(
            c.len() <= 5,
            "expected aggressive trim, got len {}",
            c.len()
        );
        assert!(c.total_tokens() <= 50);
    }

    #[test]
    fn keeps_system_plus_latest_turn_at_minimum() {
        let mut c = Conversation::new("sys".into(), 1, 1);
        c.push(user("hi"));
        c.push(assistant("hello"));
        // Even with a tiny budget, system + the latest turn are kept.
        assert!(c.len() >= 3);
        assert_eq!(c.messages().last().unwrap().role, "assistant");
    }

    #[test]
    fn under_budget_does_not_trim() {
        let mut c = Conversation::new("sys".into(), 100_000, 100);
        c.push(user("short"));
        c.push(assistant("reply"));
        assert_eq!(c.len(), 3);
    }
}
