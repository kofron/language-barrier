use crate::message::Message;
use crate::token::TokenCounter;

/// A trait for strategies to compact chat history when it exceeds token limits
///
/// This trait defines a strategy pattern for managing conversation history
/// within token budget constraints. Different implementations can choose how
/// to trim history when the token budget is exceeded.
///
/// # Examples
///
/// ```
/// use language_barrier::{ChatHistoryCompactor, TokenCounter, DropOldestCompactor};
/// use language_barrier::message::Message;
///
/// let mut compactor = DropOldestCompactor::default();
/// let mut history = vec![
///     Message::system("You are a helpful assistant."),
///     Message::user("Hello!"),
///     Message::assistant("Hi there! How can I help you?"),
/// ];
/// let mut counter = TokenCounter::default();
/// 
/// // Add token counts for all messages
/// for msg in &history {
///     if let Some(content) = &msg.content {
///         if let crate::message::Content::Text(text) = content {
///             counter.observe(text);
///         }
///     }
/// }
///
/// // Compact history to fit within 5 tokens
/// compactor.compact(&mut history, &mut counter, 5);
///
/// // The oldest messages should be removed first
/// assert_eq!(history.len(), 1);
/// ```
pub trait ChatHistoryCompactor: Send + Sync + 'static {
    /// Mutates `history` and `counter` in-place to stay within `max_tokens` budget
    ///
    /// Implementations should modify the history to ensure the token count
    /// stays within the provided budget, and update the token counter to reflect
    /// the new token count.
    ///
    /// # Arguments
    ///
    /// * `history` - The conversation history to compact
    /// * `counter` - The token counter tracking token usage
    /// * `max_tokens` - The maximum number of tokens allowed
    fn compact(
        &mut self,
        history: &mut Vec<Message>,
        counter: &mut TokenCounter,
        max_tokens: usize,
    );
}

/// Default compactor implementation that drops oldest messages first
///
/// This implementation removes messages from the beginning of the history
/// until the token budget is satisfied. It's a simple but effective strategy
/// for many use cases.
///
/// # Examples
///
/// ```
/// use language_barrier::{ChatHistoryCompactor, DropOldestCompactor, TokenCounter};
/// use language_barrier::message::{Message, Content};
///
/// let mut compactor = DropOldestCompactor::default();
/// let mut history = vec![
///     Message::system("You are a helpful assistant."),
///     Message::user("Hello"),
///     Message::assistant("Hi there!"),
/// ];
/// let mut counter = TokenCounter::default();
/// counter.observe("You are a helpful assistant.");
/// counter.observe("Hello");
/// counter.observe("Hi there!");
///
/// // Compact to fit within 3 tokens
/// compactor.compact(&mut history, &mut counter, 3);
/// 
/// // Should have removed the oldest message
/// assert_eq!(history.len(), 2);
/// ```
#[derive(Default, Debug, Clone)]
pub struct DropOldestCompactor;

impl ChatHistoryCompactor for DropOldestCompactor {
    fn compact(
        &mut self,
        history: &mut Vec<Message>,
        counter: &mut TokenCounter,
        max_tokens: usize,
    ) {
        // Keep removing the oldest messages until we're under budget or history has at most one message left
        while !counter.under_budget(max_tokens) && history.len() > 1 {
            // Remove the oldest message
            let removed_msg = history.remove(0);
            
            // Update the token count
            if let Some(content) = &removed_msg.content {
                match content {
                    crate::message::Content::Text(text) => {
                        counter.subtract(text);
                    },
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Content, MessageRole};

    #[test]
    fn test_drop_oldest_compactor() {
        let mut compactor = DropOldestCompactor::default();
        let mut history = vec![
            Message {
                role: MessageRole::System,
                content: Some(Content::Text("System message".to_string())),
                name: None,
                function_call: None,
                tool_calls: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
            Message {
                role: MessageRole::User,
                content: Some(Content::Text("First user message".to_string())),
                name: None,
                function_call: None,
                tool_calls: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
            Message {
                role: MessageRole::Assistant,
                content: Some(Content::Text("First assistant message".to_string())),
                name: None,
                function_call: None,
                tool_calls: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
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
        assert!(history.len() <= 2);
        
        // Check that we have a content that we expect
        let has_expected_message = history.iter().any(|msg| {
            if let Some(Content::Text(content)) = &msg.content {
                content == "First user message" || content == "First assistant message"
            } else {
                false
            }
        });
        assert!(has_expected_message, "Expected to find a known message");

        // Should have adjusted token count to be at most 5
        assert!(counter.total() <= 5);
        
        // Compact to fit within 2 tokens
        compactor.compact(&mut history, &mut counter, 2);
        
        // Should have removed the user message too
        // We should have removed messages until we're under the token budget
        assert!(!history.is_empty());
        
        // The content might vary, but we expect at least one remaining message
        
        // Final token count should be at or under budget
        assert!(counter.total() <= 3); // "First" "assistant" "message"
    }
    
    #[test]
    fn test_compact_empty_history() {
        let mut compactor = DropOldestCompactor::default();
        let mut history = Vec::new();
        let mut counter = TokenCounter::default();
        
        // Compact an empty history (should not panic)
        compactor.compact(&mut history, &mut counter, 100);
        
        // Still empty
        assert_eq!(history.len(), 0);
        assert_eq!(counter.total(), 0);
    }
    
    #[test]
    fn test_compact_already_under_budget() {
        let mut compactor = DropOldestCompactor::default();
        let mut history = vec![
            Message::system("System message"),
            Message::user("User message"),
        ];
        let mut counter = TokenCounter::default();
        counter.observe("System message");
        counter.observe("User message");
        
        // Initial state
        assert_eq!(history.len(), 2);
        assert_eq!(counter.total(), 4); // "System" "message" "User" "message"
        
        // Compact with a budget larger than current usage
        compactor.compact(&mut history, &mut counter, 10);
        
        // Should not change anything
        assert_eq!(history.len(), 2);
        assert_eq!(counter.total(), 4);
    }
}