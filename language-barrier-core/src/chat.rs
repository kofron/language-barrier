use crate::compactor::{ChatHistoryCompactor, DropOldestCompactor};
use crate::message::{Content, Message};
use crate::token::TokenCounter;
use crate::tool::{LlmToolInfo, ToolChoice};
use crate::{Result, ToolDefinition};

/// The main Chat client that users will interact with.
/// All methods return a new instance rather than mutating the existing one,
/// following the immutable builder pattern.
#[derive(Clone)]
pub struct Chat {
    // Tunable knobs / state
    pub system_prompt: String,
    pub max_output_tokens: usize,

    // History and token tracking
    pub history: Vec<Message>,
    token_counter: TokenCounter,

    // Registry for type-safe tool definitions (optional)
    pub tools: Option<Vec<LlmToolInfo>>,

    // Tool execution settings
    pub tool_choice: Option<ToolChoice>,
}

impl Chat {
    /// Creates a new Chat instance with a model and provider
    pub fn new() -> Self {
        Self {
            system_prompt: String::new(),
            max_output_tokens: 2048,
            history: Vec::new(),
            token_counter: TokenCounter::default(),
            tools: None,
            tool_choice: None,
        }
    }

    /// Sets system prompt and returns a new instance
    #[must_use]
    pub fn with_system_prompt(self, prompt: impl Into<String>) -> Self {
        let p = prompt.into();
        let mut token_counter = self.token_counter.clone();
        token_counter.observe(&p);

        let mut new_chat = Self {
            system_prompt: p,
            token_counter,
            ..self
        };

        new_chat = new_chat.trim_to_context_window();
        new_chat
    }

    /// Sets max output tokens and returns a new instance
    #[must_use]
    pub fn with_max_output_tokens(self, n: usize) -> Self {
        Self {
            max_output_tokens: n,
            ..self
        }
    }

    /// Sets history and returns a new instance
    #[must_use]
    pub fn with_history(self, history: Vec<Message>) -> Self {
        // Create a new token counter from scratch
        let mut token_counter = TokenCounter::default();

        // Count tokens in system prompt
        token_counter.observe(&self.system_prompt);

        // Count tokens in message history
        for msg in &history {
            match msg {
                Message::User { content, .. } => {
                    if let Content::Text(text) = content {
                        token_counter.observe(text);
                    }
                }
                Message::Assistant { content, .. } => {
                    if let Some(Content::Text(text)) = content {
                        token_counter.observe(text);
                    }
                }
                Message::System { content, .. } | Message::Tool { content, .. } => {
                    token_counter.observe(content);
                }
            }
        }

        let mut new_chat = Self {
            history,
            token_counter,
            ..self
        };

        new_chat = new_chat.trim_to_context_window();
        new_chat
    }

    /// Adds a message to the conversation history and returns a new instance
    #[must_use]
    pub fn add_message(self, msg: Message) -> Self {
        let mut token_counter = self.token_counter.clone();
        let mut history = self.history.clone();

        // Count tokens based on message type
        match &msg {
            Message::User { content, .. } => {
                if let Content::Text(text) = content {
                    token_counter.observe(text);
                }
            }
            Message::Assistant { content, .. } => {
                if let Some(Content::Text(text)) = content {
                    token_counter.observe(text);
                }
            }
            Message::System { content, .. } | Message::Tool { content, .. } => {
                token_counter.observe(content);
            }
        }

        history.push(msg);

        let mut new_chat = Self {
            history,
            token_counter,
            ..self
        };

        new_chat = new_chat.trim_to_context_window();
        new_chat
    }

    /// Alias for `add_message` for backward compatibility
    #[must_use]
    pub fn push_message(self, msg: Message) -> Self {
        self.add_message(msg)
    }

    /// Trims the conversation history to fit within token budget and returns a new instance
    #[must_use]
    fn trim_to_context_window(self) -> Self {
        const MAX_TOKENS: usize = 32_768; // could be model-specific

        let mut history = self.history.clone();
        let mut token_counter = self.token_counter.clone();

        // Create a fresh compactor of the same default type
        // Note: In a real implementation, you would want a way to clone the compactor
        // or to properly reconstruct the specific type that was being used.
        let new_compactor = Box::<DropOldestCompactor>::default();

        // Use the compactor to trim history
        new_compactor.compact(&mut history, &mut token_counter, MAX_TOKENS);

        Self {
            history,
            token_counter,
            ..self
        }
    }

    /// Gets the current token count
    pub fn tokens_used(&self) -> usize {
        self.token_counter.total()
    }

    /// Add a tool and returns a new instance with the tool added
    #[must_use = "This returns a new Chat with the tool added"]
    pub fn with_tool(self, tool: impl ToolDefinition) -> Result<Self> {
        let info = LlmToolInfo {
            name: tool.name(),
            description: tool.description(),
            parameters: tool.schema()?,
        };

        let tools = match self.tools {
            Some(mut tools) => {
                tools.push(info);
                Some(tools)
            }
            None => Some(vec![info]),
        };

        let new_chat = Self { tools, ..self };

        Ok(new_chat)
    }

    /// Add multiple tools at once and return a new instance with the tools added
    #[must_use = "This returns a new Chat with the tools added"]
    pub fn with_tools(self, tools: Vec<LlmToolInfo>) -> Self {
        let new_tools = match self.tools {
            Some(mut existing_tools) => {
                existing_tools.extend(tools);
                Some(existing_tools)
            }
            None => Some(tools),
        };

        Self {
            tools: new_tools,
            ..self
        }
    }

    /// Sets the tool choice strategy and returns a new instance
    ///
    /// This method allows configuring how the model should choose tools:
    /// - `ToolChoice::Auto` - Model can choose whether to use a tool (default)
    /// - `ToolChoice::Any` - Model must use one of the available tools
    /// - `ToolChoice::None` - Model must not use any tools
    /// - `ToolChoice::Specific(name)` - Model must use the specified tool
    ///
    /// Different providers implement this with slightly different terminology:
    /// - OpenAI/Mistral use "auto", "required", "none"
    /// - Anthropic uses "auto", "any", "none"
    /// - Gemini uses function_calling_config with modes
    ///
    /// The library transparently handles these differences, providing a
    /// consistent API regardless of which provider you're using.
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier_core::{Chat, tool::ToolChoice};
    ///
    /// // Require using a tool
    /// let chat = Chat::new()
    ///     .with_tool_choice(ToolChoice::Any);
    ///
    /// // Specify a tool by name
    /// let chat = Chat::new()
    ///     .with_tool_choice(ToolChoice::Specific("weather_tool".to_string()));
    ///
    /// // Disable tools for this conversation
    /// let chat = Chat::new()
    ///     .with_tool_choice(ToolChoice::None);
    /// ```
    #[must_use]
    pub fn with_tool_choice(self, choice: ToolChoice) -> Self {
        Self {
            tool_choice: Some(choice),
            ..self
        }
    }

    /// Removes tool choice configuration and returns a new instance
    ///
    /// This resets to the default behavior, where the model can choose whether to use tools.
    #[must_use]
    pub fn without_tool_choice(self) -> Self {
        Self {
            tool_choice: None,
            ..self
        }
    }
}
