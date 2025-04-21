use crate::compactor::{ChatHistoryCompactor, DropOldestCompactor};
use crate::message::{Content, Message};
use crate::token::TokenCounter;
use crate::tool::LlmToolInfo;
use crate::{ModelInfo, Result, ToolDefinition};

/// The main Chat client that users will interact with
pub struct Chat<M: ModelInfo> {
    // Immutable after construction
    pub model: M,

    // Tunable knobs / state
    pub system_prompt: String,
    pub max_output_tokens: usize,

    // History and token tracking
    pub history: Vec<Message>,
    token_counter: TokenCounter,
    compactor: Box<dyn ChatHistoryCompactor>,

    // Registry for type-safe tool definitions (optional)
    pub tools: Option<Vec<LlmToolInfo>>,
}

impl<M> Chat<M>
where
    M: ModelInfo,
{
    /// Creates a new Chat instance with a model and provider
    pub fn new(model: M) -> Self {
        Self {
            model,
            system_prompt: String::new(),
            max_output_tokens: 2048,
            history: Vec::new(),
            token_counter: TokenCounter::default(),
            compactor: Box::<DropOldestCompactor>::default(),
            tools: None,
        }
    }

    /// Sets system prompt and returns self for method chaining
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.set_system_prompt(prompt);
        self
    }

    /// Sets max output tokens and returns self for method chaining
    #[must_use]
    pub fn with_max_output_tokens(mut self, n: usize) -> Self {
        self.max_output_tokens = n;
        self
    }

    /// Sets history and returns self for method chaining
    #[must_use]
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        // Recompute token counter from scratch
        for msg in &history {
            match msg {
                Message::User { content, .. } => {
                    if let Content::Text(text) = content {
                        self.token_counter.observe(text);
                    }
                }
                Message::Assistant { content, .. } => {
                    if let Some(Content::Text(text)) = content {
                        self.token_counter.observe(text);
                    }
                }
                Message::System { content, .. } | Message::Tool { content, .. } => {
                    self.token_counter.observe(content);
                }
            }
        }
        self.history = history;
        self
    }

    /// Sets compactor and returns self for method chaining
    #[must_use]
    pub fn with_compactor<C: ChatHistoryCompactor + 'static>(mut self, comp: C) -> Self {
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
        // Count tokens based on message type
        match &msg {
            Message::User { content, .. } => {
                if let Content::Text(text) = content {
                    self.token_counter.observe(text);
                }
            }
            Message::Assistant { content, .. } => {
                if let Some(Content::Text(text)) = content {
                    self.token_counter.observe(text);
                }
            }
            Message::System { content, .. } | Message::Tool { content, .. } => {
                self.token_counter.observe(content);
            }
        }
        self.history.push(msg);
        self.trim_to_context_window();
    }

    /// Alias for `push_message` for better readability
    pub fn add_message(&mut self, msg: Message) {
        self.push_message(msg);
    }

    /// Sets a new compactor strategy at runtime
    pub fn set_compactor<C: ChatHistoryCompactor + 'static>(&mut self, comp: C) {
        self.compactor = Box::new(comp);
        self.trim_to_context_window();
    }

    /// Trims the conversation history to fit within token budget
    fn trim_to_context_window(&mut self) {
        const MAX_TOKENS: usize = 32_768; // could be model-specific
        self.compactor
            .compact(&mut self.history, &mut self.token_counter, MAX_TOKENS);
    }

    /// Gets the current token count
    pub fn tokens_used(&self) -> usize {
        self.token_counter.total()
    }

    /// Add a tool
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
            },
            None => Some(vec![info]),
        };
        
        let new_chat = Self {
            tools,
            ..self
        };

        Ok(new_chat)
    }
}
