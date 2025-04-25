use crate::token::TokenCounter;

/// Trait for compacting chat history
pub trait ChatHistoryCompactor: Send + Sync + Clone {
    /// Compacts the chat history to fit within a token budget
    ///
    /// This method should modify the history in place, removing
    /// messages as needed, and updating the token counter.
    fn compact(
        &self,
        history: &mut Vec<crate::message::Message>,
        counter: &mut TokenCounter,
        max_tokens: usize,
    );
}

/// Compactor that drops oldest messages first
#[derive(Debug, Default, Clone)]
pub struct DropOldestCompactor {}

impl ChatHistoryCompactor for DropOldestCompactor {
    fn compact(
        &self,
        history: &mut Vec<crate::message::Message>,
        counter: &mut TokenCounter,
        max_tokens: usize,
    ) {
        // If history is empty or we're already under budget, nothing to do
        if history.is_empty() || counter.under_budget(max_tokens) {
            return;
        }

        // While we're over budget, keep removing oldest messages
        while !counter.under_budget(max_tokens) && history.len() > 1 {
            // Remove the oldest message
            let removed_msg = history.remove(0);

            // Update the token count based on message type
            match &removed_msg {
                crate::message::Message::User { content, .. } => {
                    match content {
                        crate::message::Content::Text(text) => {
                            counter.subtract(text);
                        }
                        crate::message::Content::Parts(parts) => {
                            // For multimodal content, count all text parts
                            for part in parts {
                                if let crate::message::ContentPart::Text { text } = part {
                                    counter.subtract(text);
                                }
                            }
                        }
                    }
                }
                crate::message::Message::Assistant { content, .. } => {
                    if let Some(content_data) = content {
                        match content_data {
                            crate::message::Content::Text(text) => {
                                counter.subtract(text);
                            }
                            crate::message::Content::Parts(parts) => {
                                // For multimodal content, count all text parts
                                for part in parts {
                                    if let crate::message::ContentPart::Text { text } = part {
                                        counter.subtract(text);
                                    }
                                }
                            }
                        }
                    }
                }
                crate::message::Message::System { content, .. }
                | crate::message::Message::Tool { content, .. } => {
                    counter.subtract(content);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::message::Message;

    #[test]
    fn test_drop_oldest_compactor() {
        let compactor = DropOldestCompactor::default();
        let mut history = vec![
            Message::system("System message"),
            Message::user("First user message"),
            Message::assistant("First assistant message"),
        ];
        let mut counter = TokenCounter::default();
        counter.observe("System message");
        counter.observe("First user message");
        counter.observe("First assistant message");

        // Initial state
        assert_eq!(history.len(), 3);
        assert_eq!(counter.total(), 8); // "System" "message" "First" "user" "message" "First" "assistant" "message"

        // Compact to fit within 5 tokens
        compactor.compact(&mut history, &mut counter, 5);

        // Should have removed at least the system message
        assert!(history.len() < 3);
        assert!(counter.total() <= 5);
    }
}
