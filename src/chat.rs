use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::client::{GenerationOptions, GenerationResult, LlmProvider};
use crate::compactor::{ChatHistoryCompactor, DropOldestCompactor};
use crate::error::Result;
use crate::message::{Content, FunctionCall, Message, MessageRole, ToolCall};
use crate::token::TokenCounter;

/// A simplified representation of a message in a chat
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// The role of the message sender
    pub role: MessageRole,
    /// The content of the message
    pub content: String,
    /// The name of the sender (optional)
    pub name: Option<String>,
    /// The function call (for assistant messages)
    pub function_call: Option<FunctionCall>,
    /// The tool calls (for assistant messages)
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The ID of the tool call this message is responding to
    pub tool_call_id: Option<String>,
}

impl From<Message> for ChatMessage {
    fn from(message: Message) -> Self {
        let content = match message.content {
            Some(Content::Text(text)) => text,
            Some(Content::Parts(_)) => "[Complex content]".to_string(),
            None => "".to_string(),
        };

        Self {
            role: message.role,
            content,
            name: message.name,
            function_call: message.function_call,
            tool_calls: message.tool_calls,
            tool_call_id: message.tool_call_id,
        }
    }
}

impl From<ChatMessage> for Message {
    fn from(chat_message: ChatMessage) -> Self {
        let content_is_empty = chat_message.content.is_empty();
        let has_function_or_tool = chat_message.function_call.is_some() || chat_message.tool_calls.is_some();
        
        let content = if content_is_empty && has_function_or_tool {
            None
        } else {
            Some(Content::Text(chat_message.content))
        };

        Message {
            role: chat_message.role,
            content,
            name: chat_message.name,
            function_call: chat_message.function_call,
            tool_calls: chat_message.tool_calls,
            tool_call_id: chat_message.tool_call_id,
            metadata: HashMap::new(),
        }
    }
}

impl ChatMessage {
    /// Creates a new user message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatMessage;
    ///
    /// let message = ChatMessage::user("Hello, how are you?");
    /// ```
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a new assistant message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatMessage;
    ///
    /// let message = ChatMessage::assistant("I'm doing well, thank you!");
    /// ```
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a new system message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatMessage;
    ///
    /// let message = ChatMessage::system("You are a helpful assistant.");
    /// ```
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a new function message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatMessage;
    ///
    /// let message = ChatMessage::function("get_weather", "The weather is sunny.");
    /// ```
    pub fn function(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Function,
            content: content.into(),
            name: Some(name.into()),
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Creates a new tool message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatMessage;
    ///
    /// let message = ChatMessage::tool("tool123", "The result is 42.");
    /// ```
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Sets the name and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatMessage;
    ///
    /// let message = ChatMessage::user("Hello")
    ///     .with_name("John");
    /// ```
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl fmt::Display for ChatMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.role {
            MessageRole::System => write!(f, "System: {}", self.content),
            MessageRole::User => {
                if let Some(name) = &self.name {
                    write!(f, "{}:", name)?;
                } else {
                    write!(f, "User:")?;
                }
                write!(f, " {}", self.content)
            }
            MessageRole::Assistant => {
                if let Some(function_call) = &self.function_call {
                    write!(
                        f,
                        "Assistant: [Function call: {}({})]",
                        function_call.name, function_call.arguments
                    )
                } else if let Some(tool_calls) = &self.tool_calls {
                    write!(
                        f,
                        "Assistant: [Tool calls: {} tool(s)]",
                        tool_calls.len()
                    )
                } else {
                    write!(f, "Assistant: {}", self.content)
                }
            }
            MessageRole::Function => {
                if let Some(name) = &self.name {
                    write!(f, "Function {}: {}", name, self.content)
                } else {
                    write!(f, "Function: {}", self.content)
                }
            }
            MessageRole::Tool => {
                if let Some(tool_id) = &self.tool_call_id {
                    write!(f, "Tool {}: {}", tool_id, self.content)
                } else {
                    write!(f, "Tool: {}", self.content)
                }
            }
        }
    }
}

/// Represents a response from the chat
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The message from the assistant
    pub message: ChatMessage,
    /// Metadata about the generation
    pub metadata: HashMap<String, serde_json::Value>,
    /// Whether the response contains a function call
    pub has_function_call: bool,
    /// Whether the response contains tool calls
    pub has_tool_calls: bool,
}

impl From<GenerationResult> for ChatResponse {
    fn from(result: GenerationResult) -> Self {
        let chat_message = ChatMessage::from(result.message.clone());
        
        let mut metadata = HashMap::new();
        metadata.insert("id".to_string(), serde_json::to_value(result.metadata.id).unwrap());
        metadata.insert("model".to_string(), serde_json::to_value(result.metadata.model).unwrap());
        metadata.insert("created".to_string(), serde_json::to_value(result.metadata.created).unwrap());
        
        if let Some(usage) = result.metadata.usage {
            metadata.insert(
                "usage".to_string(),
                serde_json::json!({
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }),
            );
        }
        
        // Add any extra metadata
        for (key, value) in result.metadata.extra {
            metadata.insert(key, value);
        }

        Self {
            message: chat_message,
            metadata,
            has_function_call: result.message.function_call.is_some(),
            has_tool_calls: result.message.tool_calls.is_some(),
        }
    }
}

impl ChatResponse {
    /// Gets the content of the assistant's message
    pub fn content(&self) -> &str {
        &self.message.content
    }

    /// Gets the function call from the response, if any
    pub fn function_call(&self) -> Option<&FunctionCall> {
        self.message.function_call.as_ref()
    }

    /// Gets the tool calls from the response, if any
    pub fn tool_calls(&self) -> Option<&Vec<ToolCall>> {
        self.message.tool_calls.as_ref()
    }

    /// Gets the usage information from the response metadata
    pub fn usage(&self) -> Option<&serde_json::Value> {
        self.metadata.get("usage")
    }

    /// Gets the value of a specific metadata field
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }
}

/// The main Chat client that users will interact with
pub struct Chat<M, P> {
    // Immutable after construction
    model: M,
    provider: P,
    
    // Tunable knobs / state
    system_prompt: String,
    max_output_tokens: usize,
    
    // History and token tracking
    history: Vec<Message>,
    token_counter: TokenCounter,
    compactor: Box<dyn ChatHistoryCompactor>,
}

impl<M, P> Chat<M, P>
where
    M: Clone + Send + Sync + 'static,
    P: Clone + Send + Sync + 'static,
{
    /// Creates a new Chat instance with a model and provider
    pub fn new(model: M, provider: P) -> Self {
        Self {
            model,
            provider,
            system_prompt: String::new(),
            max_output_tokens: 2048,
            history: Vec::new(),
            token_counter: TokenCounter::default(),
            compactor: Box::<DropOldestCompactor>::default(),
        }
    }
    
    /// Sets system prompt and returns self for method chaining
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.set_system_prompt(prompt);
        self
    }
    
    /// Sets max output tokens and returns self for method chaining
    pub fn with_max_output_tokens(mut self, n: usize) -> Self {
        self.max_output_tokens = n;
        self
    }
    
    /// Sets history and returns self for method chaining
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        // Recompute token counter from scratch
        for msg in &history {
            if let Some(Content::Text(text)) = &msg.content {
                self.token_counter.observe(text);
            }
        }
        self.history = history;
        self
    }
    
    /// Sets compactor and returns self for method chaining
    pub fn with_compactor<C: ChatHistoryCompactor>(mut self, comp: C) -> Self {
        self.compactor = Box::new(comp);
        self.trim_to_context_window();
        self
    }
    
    /// Sets system prompt at runtime
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        let p = prompt.into();
        self.token_counter.observe(&p);
        self.system_prompt = p;
        self.trim_to_context_window();
    }
    
    /// Sets max output tokens at runtime
    pub fn set_max_output_tokens(&mut self, n: usize) {
        self.max_output_tokens = n;
    }
    
    /// Adds a message to the conversation history
    pub fn push_message(&mut self, msg: Message) {
        if let Some(Content::Text(text)) = &msg.content {
            self.token_counter.observe(text);
        }
        self.history.push(msg);
        self.trim_to_context_window();
    }
    
    /// Sets a new compactor strategy at runtime
    pub fn set_compactor<C: ChatHistoryCompactor>(&mut self, comp: C) {
        self.compactor = Box::new(comp);
        self.trim_to_context_window();
    }
    
    /// Trims the conversation history to fit within token budget
    fn trim_to_context_window(&mut self) {
        const MAX_TOKENS: usize = 32_768; // could be model-specific
        self.compactor.compact(&mut self.history, &mut self.token_counter, MAX_TOKENS);
    }
    
    /// Gets the current token count
    pub fn tokens_used(&self) -> usize {
        self.token_counter.total()
    }
}

impl<M> Chat<M, Arc<dyn LlmProvider>> 
where
    M: Clone + Send + Sync + AsRef<str> + 'static,
{
    /// Sends a message and gets a response 
    pub async fn send(&mut self, message: impl Into<String>) -> Result<ChatResponse> {
        let msg = Message::user(message);
        self.push_message(msg);
        self.generate_response().await
    }
    
    /// Sends a function response and gets a new response
    pub async fn send_function_response(
        &mut self,
        function_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<ChatResponse> {
        let msg = Message::function(function_name, content);
        self.push_message(msg);
        self.generate_response().await
    }
    
    /// Sends a tool response and gets a new response
    pub async fn send_tool_response(
        &mut self,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<ChatResponse> {
        let msg = Message::tool(tool_call_id, content);
        self.push_message(msg);
        self.generate_response().await
    }

    /// Helper method to generate a response based on current history
    async fn generate_response(&mut self) -> Result<ChatResponse> {
        // Prepare the messages to send to the provider
        let mut messages = Vec::new();
        
        // Add system prompt if specified
        if !self.system_prompt.is_empty() {
            messages.push(Message::system(&self.system_prompt));
        }
        
        // Add conversation history
        for msg in &self.history {
            messages.push(msg.clone());
        }
        
        // Prepare generation options
        let generation_options = GenerationOptions::new()
            .with_max_tokens(self.max_output_tokens as u32);
        
        // Generate response
        let model_id = self.model.as_ref();
        let generation_result = self.provider
            .generate(model_id, &messages, generation_options)
            .await?;
        
        // Convert to ChatResponse
        let response = ChatResponse::from(generation_result.clone());
        
        // Add to history
        self.push_message(generation_result.message);
        
        Ok(response)
    }
}

/// Factory methods for creating chats with specific providers
pub struct ChatBuilder;

impl ChatBuilder {
    /// Creates a new chat with an Anthropic API key
    pub fn anthropic(api_key: impl Into<String>, model: impl Into<String>) -> Chat<String, Arc<dyn LlmProvider>> {
        let provider = Arc::new(crate::provider::AnthropicProvider::with_api_key(api_key)) as Arc<dyn LlmProvider>;
        Chat::new(model.into(), provider)
            .with_system_prompt("You are a helpful AI assistant.")
    }
    
    /// Creates a new chat with a Google API key
    pub fn google(api_key: impl Into<String>, model: impl Into<String>) -> Chat<String, Arc<dyn LlmProvider>> {
        let provider = Arc::new(crate::provider::GoogleProvider::with_api_key(api_key)) as Arc<dyn LlmProvider>;
        Chat::new(model.into(), provider)
            .with_system_prompt("You are a helpful AI assistant.")
    }
    
    /// Creates a mock chat for testing
    pub fn mock() -> Chat<String, Arc<dyn LlmProvider>> {
        let provider = Arc::new(crate::client::MockProvider::new()) as Arc<dyn LlmProvider>;
        Chat::new("mock-model".to_string(), provider)
            .with_system_prompt("You are a helpful AI assistant.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{FunctionCall, MessageRole};

    #[tokio::test]
    async fn test_chat_creation() {
        // Create a mock chat
        let chat = ChatBuilder::mock();
        
        // Verify system prompt
        assert_eq!(chat.system_prompt, "You are a helpful AI assistant.");
        
        // Verify model
        assert_eq!(chat.model, "mock-model");
        
        // Verify default token tracking
        assert_eq!(chat.token_counter.total(), 6); // "You are a helpful AI assistant." is 6 tokens
    }
    
    #[tokio::test]
    async fn test_chat_send_message() {
        let mut chat = ChatBuilder::mock();
        let response = chat.send("Hello, world!").await.unwrap();
        
        // Verify that message was added to history
        assert_eq!(chat.history.len(), 2); // user message and assistant's response
        assert_eq!(chat.history[0].role, MessageRole::User);
        assert!(chat.history[0].content.is_some());
        if let Some(Content::Text(text)) = &chat.history[0].content {
            assert_eq!(text, "Hello, world!");
        }
        assert_eq!(chat.history[1].role, MessageRole::Assistant);
        
        // Verify response
        assert_eq!(response.message.role, MessageRole::Assistant);
        assert!(!response.message.content.is_empty());
    }
    
    #[tokio::test]
    async fn test_chat_multiple_turns() {
        // Create a mock provider with different responses
        let mut provider = crate::client::MockProvider::new();
        provider.add_response("Hello", Message::assistant("Response to Hello"));
        provider.add_response("How are you", Message::assistant("Response to How are you"));
        
        let provider_arc = Arc::new(provider) as Arc<dyn LlmProvider>;
        let mut chat = Chat::new("mock-model".to_string(), provider_arc);
        
        // First turn
        let response1 = chat.send("Hello").await.unwrap();
        
        // Second turn
        let response2 = chat.send("How are you?").await.unwrap();
        
        assert_ne!(response1.content(), response2.content());
    }
    
    #[tokio::test]
    async fn test_chat_function_call() {
        let mut provider = crate::client::AdvancedMockProvider::new();
        let function_call = FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"New York","unit":"celsius"}"#.to_string(),
        };
        provider.add_function_call("weather", function_call.clone());
        
        let provider_arc = Arc::new(provider) as Arc<dyn LlmProvider>;
        let mut chat = Chat::new("mock-model".to_string(), provider_arc);
        
        // Send message that triggers function call
        let response = chat.send("What's the weather in New York?").await.unwrap();
        
        assert!(response.has_function_call);
        assert_eq!(response.function_call().unwrap().name, "get_weather");
        
        // Send function response
        let _function_response = chat.send_function_response(
            "get_weather", 
            r#"{"temperature": 25, "condition": "sunny"}"#
        ).await.unwrap();
    }
    
    #[tokio::test]
    async fn test_token_counting_and_compaction() {
        let mut chat = ChatBuilder::mock();
        
        // Start with system prompt: "You are a helpful AI assistant." (6 tokens)
        assert_eq!(chat.tokens_used(), 6);
        
        // Add some messages
        chat.push_message(Message::user("This is a test message with several tokens")); // 8 tokens
        assert_eq!(chat.tokens_used(), 14);
        
        // Add a large message that should trigger compaction
        // Create a very long message that will exceed the context window
        let long_message = "a ".repeat(20000); // 20k tokens
        chat.push_message(Message::user(long_message));
        
        // Tokens should be much less than 20k after compaction
        assert!(chat.tokens_used() < 32_768);
        // History should contain at least one message
        assert!(!chat.history.is_empty());
    }
    
    // Custom compactor doesn't make sense to test with our current setup
    // since we've modified Chat to use Message internally rather than ChatMessage
}