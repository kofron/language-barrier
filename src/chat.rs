use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::client::{GenerationOptions, GenerationResult, LlmProvider};
use crate::error::Result;
use crate::message::{Content, FunctionCall, Message, MessageRole, ToolCall};
use crate::model::Model;

/// Options for configuring a chat session
#[derive(Debug, Clone)]
pub struct ChatOptions {
    /// The model to use for the chat
    pub model: String,
    /// System message to set context for the conversation
    pub system_message: Option<String>,
    /// Generation options for the model
    pub generation_options: GenerationOptions,
    /// Maximum number of messages to keep in the conversation history
    pub max_history_size: Option<usize>,
    /// Tool definitions that can be used in this chat session
    pub tool_definitions: Option<Vec<serde_json::Value>>,
    /// Tool names available in this chat session
    pub tool_names: Option<Vec<String>>,
    /// References to actual tool objects (not cloneable)
    #[doc(hidden)]
    pub tools: std::sync::Arc<std::sync::Mutex<Vec<Box<dyn crate::tool::Tool>>>>,
    /// Tool choice strategy
    pub tool_choice: Option<crate::client::ToolChoice>,
    /// Additional metadata for the chat session
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            model: "".to_string(),
            system_message: None,
            generation_options: GenerationOptions::default(),
            max_history_size: None,
            tool_definitions: None,
            tool_names: None,
            tools: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            tool_choice: None,
            metadata: HashMap::new(),
        }
    }
}

impl ChatOptions {
    /// Creates a new ChatOptions with the specified model
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatOptions;
    ///
    /// let options = ChatOptions::new("gpt-4");
    /// ```
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Sets the system message and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatOptions;
    ///
    /// let options = ChatOptions::new("gpt-4")
    ///     .with_system_message("You are a helpful assistant.");
    /// ```
    pub fn with_system_message(mut self, message: impl Into<String>) -> Self {
        self.system_message = Some(message.into());
        self
    }

    /// Sets the generation options and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{ChatOptions, client::GenerationOptions};
    ///
    /// let gen_options = GenerationOptions::new().with_temperature(0.7);
    /// let options = ChatOptions::new("gpt-4")
    ///     .with_generation_options(gen_options);
    /// ```
    pub fn with_generation_options(mut self, options: GenerationOptions) -> Self {
        self.generation_options = options;
        self
    }

    /// Sets the maximum history size and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatOptions;
    ///
    /// let options = ChatOptions::new("gpt-4")
    ///     .with_max_history_size(20);
    /// ```
    pub fn with_max_history_size(mut self, size: usize) -> Self {
        self.max_history_size = Some(size);
        self
    }

    /// Adds metadata and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatOptions;
    /// use serde_json::json;
    ///
    /// let options = ChatOptions::new("gpt-4")
    ///     .with_metadata("session_id", json!("abc123"));
    /// ```
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
    
    /// Sets the tools and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatOptions;
    /// use language_barrier::tool::calculator;
    ///
    /// let options = ChatOptions::new("gpt-4")
    ///     .with_tools(vec![Box::new(calculator())]);
    /// ```
    pub fn with_tools(mut self, tools: Vec<Box<dyn crate::tool::Tool>>) -> Self {
        // Store the tool definitions and names separately from the tool objects
        let mut definitions = Vec::new();
        let mut names = Vec::new();
        
        for tool in &tools {
            definitions.push(tool.to_provider_format());
            names.push(tool.name().to_string());
        }
        
        self.tool_definitions = Some(definitions);
        self.tool_names = Some(names);
        
        // Store the tools in the Arc<Mutex<Vec<...>>> for thread-safe access
        {
            let mut tools_lock = self.tools.lock().unwrap();
            *tools_lock = tools;
        }
        
        self
    }
    
    /// Sets the tool choice strategy and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::ChatOptions;
    /// use language_barrier::client::ToolChoice;
    ///
    /// let options = ChatOptions::new("gpt-4")
    ///     .with_tool_choice(ToolChoice::Auto);
    /// ```
    pub fn with_tool_choice(mut self, tool_choice: crate::client::ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }
}

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
    ///
    /// # Examples
    ///
    /// ```
    /// # use language_barrier::ChatResponse;
    /// # use language_barrier::ChatMessage;
    /// # use language_barrier::client::GenerationMetadata;
    /// # use language_barrier::client::UsageInfo;
    /// # use language_barrier::client::GenerationResult;
    /// # use language_barrier::message::Message;
    /// # use language_barrier::message::MessageRole;
    /// # use language_barrier::message::Content;
    /// # use std::collections::HashMap;
    /// #
    /// # let message = Message {
    /// #     role: MessageRole::Assistant,
    /// #     content: Some(Content::Text("Hello there!".to_string())),
    /// #     name: None,
    /// #     function_call: None,
    /// #     tool_calls: None,
    /// #     tool_call_id: None,
    /// #     metadata: HashMap::new(),
    /// # };
    /// #
    /// # let metadata = GenerationMetadata {
    /// #     id: "test".to_string(),
    /// #     model: "test-model".to_string(),
    /// #     created: 0,
    /// #     usage: Some(UsageInfo {
    /// #         prompt_tokens: 10,
    /// #         completion_tokens: 20,
    /// #         total_tokens: 30,
    /// #     }),
    /// #     extra: HashMap::new(),
    /// # };
    /// #
    /// # let result = GenerationResult {
    /// #     message,
    /// #     metadata,
    /// # };
    /// #
    /// # let response = ChatResponse::from(result);
    /// let content = response.content();
    /// assert_eq!(content, "Hello there!");
    /// ```
    pub fn content(&self) -> &str {
        &self.message.content
    }

    /// Gets the function call from the response, if any
    ///
    /// # Examples
    ///
    /// ```
    /// # use language_barrier::ChatResponse;
    /// # use language_barrier::ChatMessage;
    /// # use language_barrier::client::GenerationMetadata;
    /// # use language_barrier::client::UsageInfo;
    /// # use language_barrier::client::GenerationResult;
    /// # use language_barrier::message::Message;
    /// # use language_barrier::message::MessageRole;
    /// # use language_barrier::message::Content;
    /// # use language_barrier::message::FunctionCall;
    /// # use std::collections::HashMap;
    /// #
    /// # let function_call = FunctionCall {
    /// #     name: "get_weather".to_string(),
    /// #     arguments: r#"{"location":"New York"}"#.to_string(),
    /// # };
    /// #
    /// # let message = Message {
    /// #     role: MessageRole::Assistant,
    /// #     content: None,
    /// #     name: None,
    /// #     function_call: Some(function_call.clone()),
    /// #     tool_calls: None,
    /// #     tool_call_id: None,
    /// #     metadata: HashMap::new(),
    /// # };
    /// #
    /// # let metadata = GenerationMetadata {
    /// #     id: "test".to_string(),
    /// #     model: "test-model".to_string(),
    /// #     created: 0,
    /// #     usage: Some(UsageInfo {
    /// #         prompt_tokens: 10,
    /// #         completion_tokens: 20,
    /// #         total_tokens: 30,
    /// #     }),
    /// #     extra: HashMap::new(),
    /// # };
    /// #
    /// # let result = GenerationResult {
    /// #     message,
    /// #     metadata,
    /// # };
    /// #
    /// # let response = ChatResponse::from(result);
    /// if let Some(function_call) = response.function_call() {
    ///     println!("Function name: {}", function_call.name);
    ///     println!("Arguments: {}", function_call.arguments);
    /// }
    /// ```
    pub fn function_call(&self) -> Option<&FunctionCall> {
        self.message.function_call.as_ref()
    }

    /// Gets the tool calls from the response, if any
    ///
    /// # Examples
    ///
    /// ```
    /// # use language_barrier::ChatResponse;
    /// # use language_barrier::ChatMessage;
    /// # use language_barrier::client::GenerationMetadata;
    /// # use language_barrier::client::UsageInfo;
    /// # use language_barrier::client::GenerationResult;
    /// # use language_barrier::message::Message;
    /// # use language_barrier::message::MessageRole;
    /// # use language_barrier::message::Content;
    /// # use language_barrier::message::FunctionCall;
    /// # use language_barrier::message::ToolCall;
    /// # use std::collections::HashMap;
    /// #
    /// # let tool_call = ToolCall {
    /// #     id: "call_123".to_string(),
    /// #     tool_type: "function".to_string(),
    /// #     function: FunctionCall {
    /// #         name: "get_weather".to_string(),
    /// #         arguments: r#"{"location":"New York"}"#.to_string(),
    /// #     },
    /// # };
    /// #
    /// # let message = Message {
    /// #     role: MessageRole::Assistant,
    /// #     content: None,
    /// #     name: None,
    /// #     function_call: None,
    /// #     tool_calls: Some(vec![tool_call.clone()]),
    /// #     tool_call_id: None,
    /// #     metadata: HashMap::new(),
    /// # };
    /// #
    /// # let metadata = GenerationMetadata {
    /// #     id: "test".to_string(),
    /// #     model: "test-model".to_string(),
    /// #     created: 0,
    /// #     usage: Some(UsageInfo {
    /// #         prompt_tokens: 10,
    /// #         completion_tokens: 20,
    /// #         total_tokens: 30,
    /// #     }),
    /// #     extra: HashMap::new(),
    /// # };
    /// #
    /// # let result = GenerationResult {
    /// #     message,
    /// #     metadata,
    /// # };
    /// #
    /// # let response = ChatResponse::from(result);
    /// if let Some(tool_calls) = response.tool_calls() {
    ///     for tool_call in tool_calls {
    ///         println!("Tool ID: {}", tool_call.id);
    ///         println!("Function name: {}", tool_call.function.name);
    ///     }
    /// }
    /// ```
    pub fn tool_calls(&self) -> Option<&Vec<ToolCall>> {
        self.message.tool_calls.as_ref()
    }

    /// Gets the usage information from the response metadata
    ///
    /// # Examples
    ///
    /// ```
    /// # use language_barrier::ChatResponse;
    /// # use language_barrier::ChatMessage;
    /// # use language_barrier::client::GenerationMetadata;
    /// # use language_barrier::client::UsageInfo;
    /// # use language_barrier::client::GenerationResult;
    /// # use language_barrier::message::Message;
    /// # use language_barrier::message::MessageRole;
    /// # use language_barrier::message::Content;
    /// # use std::collections::HashMap;
    /// #
    /// # let message = Message {
    /// #     role: MessageRole::Assistant,
    /// #     content: Some(Content::Text("Hello there!".to_string())),
    /// #     name: None,
    /// #     function_call: None,
    /// #     tool_calls: None,
    /// #     tool_call_id: None,
    /// #     metadata: HashMap::new(),
    /// # };
    /// #
    /// # let metadata = GenerationMetadata {
    /// #     id: "test".to_string(),
    /// #     model: "test-model".to_string(),
    /// #     created: 0,
    /// #     usage: Some(UsageInfo {
    /// #         prompt_tokens: 10,
    /// #         completion_tokens: 20,
    /// #         total_tokens: 30,
    /// #     }),
    /// #     extra: HashMap::new(),
    /// # };
    /// #
    /// # let result = GenerationResult {
    /// #     message,
    /// #     metadata,
    /// # };
    /// #
    /// # let response = ChatResponse::from(result);
    /// if let Some(usage) = response.usage() {
    ///     println!("Total tokens: {}", usage["total_tokens"]);
    /// }
    /// ```
    pub fn usage(&self) -> Option<&serde_json::Value> {
        self.metadata.get("usage")
    }

    /// Gets the value of a specific metadata field
    ///
    /// # Examples
    ///
    /// ```
    /// # use language_barrier::ChatResponse;
    /// # use language_barrier::ChatMessage;
    /// # use language_barrier::client::GenerationMetadata;
    /// # use language_barrier::client::UsageInfo;
    /// # use language_barrier::client::GenerationResult;
    /// # use language_barrier::message::Message;
    /// # use language_barrier::message::MessageRole;
    /// # use language_barrier::message::Content;
    /// # use std::collections::HashMap;
    /// #
    /// # let message = Message {
    /// #     role: MessageRole::Assistant,
    /// #     content: Some(Content::Text("Hello there!".to_string())),
    /// #     name: None,
    /// #     function_call: None,
    /// #     tool_calls: None,
    /// #     tool_call_id: None,
    /// #     metadata: HashMap::new(),
    /// # };
    /// #
    /// # let metadata = GenerationMetadata {
    /// #     id: "test".to_string(),
    /// #     model: "test-model".to_string(),
    /// #     created: 0,
    /// #     usage: Some(UsageInfo {
    /// #         prompt_tokens: 10,
    /// #         completion_tokens: 20,
    /// #         total_tokens: 30,
    /// #     }),
    /// #     extra: HashMap::new(),
    /// # };
    /// #
    /// # let result = GenerationResult {
    /// #     message,
    /// #     metadata,
    /// # };
    /// #
    /// # let response = ChatResponse::from(result);
    /// let model = response.get_metadata("model").unwrap();
    /// println!("Model used: {}", model);
    /// ```
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }
}

/// The main Chat client that users will interact with
pub struct Chat {
    provider: Arc<dyn LlmProvider>,
    options: ChatOptions,
    history: Vec<ChatMessage>,
    available_models: Vec<Model>,
}

impl Chat {
    /// Creates a new Chat instance with the specified provider and options
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::sync::Arc;
    /// use language_barrier::{Chat, ChatOptions};
    /// use language_barrier::client::MockProvider;
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let provider = Arc::new(MockProvider::new());
    /// let options = ChatOptions::new("mock-model")
    ///     .with_system_message("You are a helpful assistant.");
    ///
    /// let chat = Chat::new(provider, options).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(
        provider: Arc<dyn LlmProvider>,
        options: ChatOptions,
    ) -> Result<Self> {
        // Verify that the specified model is available
        let available_models = provider.list_models().await?;
        
        // Check if the model is available
        if !options.model.is_empty() && !available_models.iter().any(|m| m.id == options.model) {
            return Err(crate::error::Error::UnsupportedModel(format!(
                "Model '{}' not found",
                options.model
            )));
        }

        let mut history = Vec::new();
        
        // Add the system message if provided
        if let Some(system_message) = &options.system_message {
            history.push(ChatMessage::system(system_message));
        }

        Ok(Self {
            provider,
            options,
            history,
            available_models,
        })
    }

    /// Creates a new Chat instance with a mock provider for testing
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions, ChatMessage};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// let response = chat.send("Hello, world!").await?;
    /// println!("Response: {}", response.content());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new_mock() -> Result<Self> {
        let provider = Arc::new(crate::client::MockProvider::new());
        let options = ChatOptions::new("mock-model")
            .with_system_message("You are a helpful assistant.");
        
        Self::new(provider, options).await
    }

    /// Gets a list of available models from the provider
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let chat = Chat::new_mock().await?;
    /// let models = chat.available_models();
    /// println!("Available models: {}", models.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn available_models(&self) -> &[Model] {
        &self.available_models
    }

    /// Gets the current chat options
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let chat = Chat::new_mock().await?;
    /// let options = chat.options();
    /// println!("Current model: {}", options.model);
    /// # Ok(())
    /// # }
    /// ```
    pub fn options(&self) -> &ChatOptions {
        &self.options
    }

    /// Updates the chat options
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// let new_options = ChatOptions::new("mock-model")
    ///     .with_system_message("You are an AI assistant.");
    /// chat.set_options(new_options);
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_options(&mut self, options: ChatOptions) {
        if self.options.system_message != options.system_message {
            // Update the system message in the history
            if let Some(system_message) = &options.system_message {
                if let Some(first_message) = self.history.first_mut() {
                    if first_message.role == MessageRole::System {
                        first_message.content = system_message.clone();
                    } else {
                        self.history.insert(0, ChatMessage::system(system_message));
                    }
                } else {
                    self.history.push(ChatMessage::system(system_message));
                }
            } else {
                // Remove the system message if it exists
                if !self.history.is_empty() && self.history[0].role == MessageRole::System {
                    self.history.remove(0);
                }
            }
        }
        
        self.options = options;
    }

    /// Gets the current conversation history
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// let response = chat.send("Hello").await?;
    /// let history = chat.history();
    /// println!("Conversation has {} messages", history.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Clears the conversation history except for the system message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// let response = chat.send("Hello").await?;
    /// chat.clear_history();
    /// assert_eq!(chat.history().len(), 1); // Only the system message remains
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_history(&mut self) {
        if !self.history.is_empty() && self.history[0].role == MessageRole::System {
            let system_message = self.history[0].clone();
            self.history.clear();
            self.history.push(system_message);
        } else {
            self.history.clear();
            
            // Restore system message if in options
            if let Some(system_message) = &self.options.system_message {
                self.history.push(ChatMessage::system(system_message));
            }
        }
    }

    /// Adds a message to the conversation history
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions, ChatMessage};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// chat.add_message(ChatMessage::user("Hello"));
    /// chat.add_message(ChatMessage::assistant("Hi there!"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_message(&mut self, message: ChatMessage) {
        self.history.push(message);
        
        // Trim history if needed
        if let Some(max_size) = self.options.max_history_size {
            if self.history.len() > max_size {
                let has_system = !self.history.is_empty() && self.history[0].role == MessageRole::System;
                
                if has_system {
                    // Keep system message and the most recent messages up to max_size
                    let system_message = self.history[0].clone();
                    let remaining_capacity = max_size - 1;
                    let new_start = self.history.len() - remaining_capacity;
                    
                    // Store the most recent messages
                    let recent_messages: Vec<_> = self.history.drain(new_start..).collect();
                    
                    // Clear and rebuild history
                    self.history.clear();
                    self.history.push(system_message);
                    self.history.extend(recent_messages);
                } else {
                    // Just keep the most recent max_size messages
                    let new_start = self.history.len() - max_size;
                    self.history = self.history.drain(new_start..).collect();
                }
            }
        }
    }

    /// Sends a message and gets a response
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// let response = chat.send("Hello, how are you?").await?;
    /// println!("Response: {}", response.content());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send(&mut self, message: impl Into<String>) -> Result<ChatResponse> {
        let chat_message = ChatMessage::user(message);
        self.add_message(chat_message);
        self.generate_response().await
    }

    /// Sends a function response and gets a new response
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// // Assume the previous response had a function call
    /// let response = chat.send_function_response(
    ///     "get_weather", 
    ///     r#"{"temperature": 72, "condition": "sunny"}"#
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send_function_response(
        &mut self,
        function_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<ChatResponse> {
        let chat_message = ChatMessage::function(function_name, content);
        self.add_message(chat_message);
        self.generate_response().await
    }
    
    /// Sends a tool response and gets a new response
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// // Assume the previous response had a tool call
    /// let response = chat.send_tool_response(
    ///     "call_123", 
    ///     r#"{"temperature": 72, "condition": "sunny"}"#
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send_tool_response(
        &mut self,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<ChatResponse> {
        let chat_message = ChatMessage::tool(tool_call_id, content);
        self.add_message(chat_message);
        self.generate_response().await
    }

    /// Sends a custom message and gets a response
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions, ChatMessage};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// let message = ChatMessage::user("Tell me a joke").with_name("John");
    /// let response = chat.send_message(message).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send_message(&mut self, message: ChatMessage) -> Result<ChatResponse> {
        self.add_message(message);
        self.generate_response().await
    }

    /// Send a message and automatically handle any tool calls
    ///
    /// This method sends a message, and if the response contains tool calls,
    /// it will automatically execute them and continue the conversation.
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    /// use language_barrier::tool::calculator;
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// // Add a calculator tool
    /// chat.add_tool(Box::new(calculator()));
    /// 
    /// // Send a message that might trigger a tool call
    /// let response = chat.send_message_with_tools("Calculate 5 + 3").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send_message_with_tools(&mut self, content: impl Into<String>) -> Result<ChatResponse> {
        let result = self.send(content).await?;
        
        // If there are tool calls, handle them automatically
        if let Some(tool_calls) = result.tool_calls() {
            // Process the first tool call
            if let Some(tool_call) = tool_calls.iter().next() {
                // Extract parameters
                let tool_name = &tool_call.function.name;
                let arguments = &tool_call.function.arguments;
                
                // Parse arguments into JSON
                let parameters: serde_json::Value = serde_json::from_str(arguments)?;
                
                // Execute the tool
                let tool_result = self.provider.execute_tool(tool_name, parameters).await?;
                
                // Add the tool result message to the conversation
                let tool_message = ChatMessage::tool(
                    &tool_call.id,
                    serde_json::to_string(&tool_result).unwrap_or_else(|_| tool_result.to_string()),
                );
                
                self.add_message(tool_message);
                
                // Get the assistant's response to the tool result
                let final_response = self.generate_response().await?;
                return Ok(final_response);
            }
        }
        
        Ok(result)
    }
    
    /// Add a tool to the chat session
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    /// use language_barrier::tool::calculator;
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// chat.add_tool(Box::new(calculator()));
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_tool(&mut self, tool: Box<dyn crate::tool::Tool>) -> &mut Self {
        // Add to tool definitions and names
        let definition = tool.to_provider_format();
        let name = tool.name().to_string();
        
        // Update tool definitions
        let mut definitions = self.options.tool_definitions.take().unwrap_or_default();
        definitions.push(definition);
        self.options.tool_definitions = Some(definitions);
        
        // Update tool names
        let mut names = self.options.tool_names.take().unwrap_or_default();
        names.push(name);
        self.options.tool_names = Some(names);
        
        // Add the actual tool to the mutex-protected vector
        {
            let mut tools = self.options.tools.lock().unwrap();
            tools.push(tool);
        }
        
        self
    }
    
    /// Set the tool choice strategy
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::{Chat, ChatOptions};
    /// use language_barrier::client::ToolChoice;
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let mut chat = Chat::new_mock().await?;
    /// chat.with_tool_choice(ToolChoice::Auto);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tool_choice(&mut self, tool_choice: crate::client::ToolChoice) -> &mut Self {
        self.options.tool_choice = Some(tool_choice);
        self
    }
    
    // Helper method to generate a response based on current history
    async fn generate_response(&mut self) -> Result<ChatResponse> {
        // Convert ChatMessages to Messages
        let messages = self.history
            .iter()
            .map(|m| Message::from(m.clone()))
            .collect::<Vec<_>>();
        
        // Copy tool information to generation options
        let mut generation_options = self.options.generation_options.clone();
        
        // Copy tool definitions and names
        if let Some(definitions) = &self.options.tool_definitions {
            generation_options.tool_definitions = Some(definitions.clone());
        }
        
        if let Some(names) = &self.options.tool_names {
            generation_options.tool_names = Some(names.clone());
        }
        
        if let Some(tool_choice) = &self.options.tool_choice {
            generation_options.tool_choice = Some(tool_choice.clone());
        }
        
        // Generate response
        let generation_result = self.provider
            .generate(&self.options.model, &messages, generation_options)
            .await?;
        
        // Convert to ChatResponse
        let response = ChatResponse::from(generation_result);
        
        // Add to history
        self.add_message(response.message.clone());
        
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::AdvancedMockProvider;
    use crate::message::{FunctionCall, MessageRole, ToolCall};

    #[tokio::test]
    async fn test_chat_creation() {
        let provider = Arc::new(crate::client::MockProvider::new());
        let options = ChatOptions::new("mock-model")
            .with_system_message("You are a helpful assistant.");
        
        let chat = Chat::new(provider, options).await.unwrap();
        
        assert_eq!(chat.history().len(), 1);
        assert_eq!(chat.history()[0].role, MessageRole::System);
        assert_eq!(chat.history()[0].content, "You are a helpful assistant.");
    }
    
    #[tokio::test]
    async fn test_chat_send_message() {
        let mut chat = Chat::new_mock().await.unwrap();
        let response = chat.send("Hello, world!").await.unwrap();
        
        assert_eq!(chat.history().len(), 3); // System, user, assistant
        assert_eq!(chat.history()[1].role, MessageRole::User);
        assert_eq!(chat.history()[1].content, "Hello, world!");
        assert_eq!(chat.history()[2].role, MessageRole::Assistant);
        
        assert_eq!(response.message.role, MessageRole::Assistant);
        assert!(!response.message.content.is_empty());
    }
    
    #[tokio::test]
    async fn test_chat_multiple_turns() {
        // Create a mock provider with different responses
        let mut provider = crate::client::MockProvider::new();
        provider.add_response("Hello", Message::assistant("Response to Hello"));
        provider.add_response("How are you", Message::assistant("Response to How are you"));
        
        let options = ChatOptions::new("mock-model");
        let mut chat = Chat::new(Arc::new(provider), options).await.unwrap();
        
        // First turn
        let response1 = chat.send("Hello").await.unwrap();
        assert_eq!(chat.history().len(), 2); // User and assistant (no system message)
        
        // Second turn
        let response2 = chat.send("How are you?").await.unwrap();
        assert_eq!(chat.history().len(), 4);
        
        assert_ne!(response1.content(), response2.content());
    }
    
    #[tokio::test]
    async fn test_chat_custom_message() {
        let mut chat = Chat::new_mock().await.unwrap();
        let message = ChatMessage::user("Custom message").with_name("John");
        
        let _response = chat.send_message(message).await.unwrap();
        
        assert_eq!(chat.history().len(), 3);
        assert_eq!(chat.history()[1].role, MessageRole::User);
        assert_eq!(chat.history()[1].name.as_deref(), Some("John"));
    }
    
    #[tokio::test]
    async fn test_chat_clear_history() {
        let mut chat = Chat::new_mock().await.unwrap();
        
        // Add some messages
        let _ = chat.send("Hello").await.unwrap();
        let _ = chat.send("How are you?").await.unwrap();
        assert_eq!(chat.history().len(), 5);
        
        // Clear history
        chat.clear_history();
        assert_eq!(chat.history().len(), 1); // Only system message remains
        assert_eq!(chat.history()[0].role, MessageRole::System);
    }
    
    #[tokio::test]
    async fn test_chat_function_call() {
        let mut provider = AdvancedMockProvider::new();
        let function_call = FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"New York","unit":"celsius"}"#.to_string(),
        };
        provider.add_function_call("weather", function_call.clone());
        
        let options = ChatOptions::new("mock-model");
        let mut chat = Chat::new(Arc::new(provider), options).await.unwrap();
        
        // Send message that triggers function call
        let response = chat.send("What's the weather in New York?").await.unwrap();
        
        assert!(response.has_function_call);
        assert_eq!(response.function_call().unwrap().name, "get_weather");
        
        // Send function response
        let _function_response = chat.send_function_response(
            "get_weather", 
            r#"{"temperature": 25, "condition": "sunny"}"#
        ).await.unwrap();
        
        assert_eq!(chat.history().len(), 4);
        assert_eq!(chat.history()[2].role, MessageRole::Function);
        assert_eq!(chat.history()[2].name.as_deref(), Some("get_weather"));
    }
    
    #[tokio::test]
    async fn test_chat_tool_call() {
        let mut provider = AdvancedMockProvider::new();
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"New York","unit":"celsius"}"#.to_string(),
            },
        };
        provider.add_tool_calls("weather", vec![tool_call.clone()]);
        
        let options = ChatOptions::new("mock-model");
        let mut chat = Chat::new(Arc::new(provider), options).await.unwrap();
        
        // Send message that triggers tool call
        let response = chat.send("What's the weather in New York?").await.unwrap();
        
        assert!(response.has_tool_calls);
        assert_eq!(response.tool_calls().unwrap()[0].id, "call_123");
        
        // Send tool response
        let _tool_response = chat.send_tool_response(
            "call_123", 
            r#"{"temperature": 25, "condition": "sunny"}"#
        ).await.unwrap();
        
        assert_eq!(chat.history().len(), 4);
        assert_eq!(chat.history()[2].role, MessageRole::Tool);
        assert_eq!(chat.history()[2].tool_call_id.as_deref(), Some("call_123"));
    }
    
    #[tokio::test]
    async fn test_chat_message_display() {
        let user_msg = ChatMessage::user("Hello");
        assert_eq!(format!("{}", user_msg), "User: Hello");
        
        let named_user_msg = ChatMessage::user("Hi").with_name("John");
        assert_eq!(format!("{}", named_user_msg), "John: Hi");
        
        let system_msg = ChatMessage::system("You are an assistant");
        assert_eq!(format!("{}", system_msg), "System: You are an assistant");
        
        let assistant_msg = ChatMessage::assistant("I can help you");
        assert_eq!(format!("{}", assistant_msg), "Assistant: I can help you");
        
        let function_msg = ChatMessage::function("get_weather", "It's sunny");
        assert_eq!(format!("{}", function_msg), "Function get_weather: It's sunny");
        
        let tool_msg = ChatMessage::tool("call_123", "Result: sunny");
        assert_eq!(format!("{}", tool_msg), "Tool call_123: Result: sunny");
    }
    
    #[tokio::test]
    async fn test_chat_metadata() {
        let mut chat = Chat::new_mock().await.unwrap();
        let response = chat.send("Hello").await.unwrap();
        
        assert!(response.metadata.contains_key("id"));
        assert!(response.metadata.contains_key("model"));
        assert!(response.metadata.contains_key("created"));
        assert!(response.usage().is_some());
    }
    
    #[tokio::test]
    async fn test_chat_history_limit() {
        // Create a chat with a mock provider and no system message
        let provider = crate::client::MockProvider::new();
        let options = ChatOptions::new("mock-model")
            .with_max_history_size(4); // No system message
        
        let mut chat = Chat::new(Arc::new(provider), options).await.unwrap();
        
        // Add enough messages to exceed the limit
        for i in 1..=5 {
            let _ = chat.send(format!("Message {}", i)).await.unwrap();
        }
        
        // History should be trimmed to the limit (4)
        assert_eq!(chat.history().len(), 4);
        
        // Check the messages
        assert_eq!(chat.history()[0].content, "Message 4");
        assert_eq!(chat.history()[0].role, MessageRole::User);
        
        assert_eq!(chat.history()[1].role, MessageRole::Assistant);
        
        assert_eq!(chat.history()[2].content, "Message 5");
        assert_eq!(chat.history()[2].role, MessageRole::User);
        
        assert_eq!(chat.history()[3].role, MessageRole::Assistant);
    }
    
    #[tokio::test]
    async fn test_chat_with_calculator_tool() {
        // Create a chat with the mock provider
        let mut chat = Chat::new_mock().await.unwrap();
        
        // Add the calculator tool
        chat.add_tool(Box::new(crate::tool::calculator()));
        
        // Send a message that should trigger the calculator tool
        let response = chat.send_message_with_tools("calculate 5 + 3").await.unwrap();
        
        // Print the response content for debugging
        println!("Response content: {}", response.content());
        
        // Check the conversation history - should include the tool call and result
        assert!(chat.history().len() >= 4); // User, Assistant with tool call, Tool, Assistant with result
        
        // Find the tool message
        let tool_message = chat.history().iter().find(|m| m.role == MessageRole::Tool);
        assert!(tool_message.is_some());
        
        // Less strict assertion that doesn't depend on exact content
        assert!(response.content().len() > 0);
    }
}