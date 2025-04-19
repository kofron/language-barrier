use crate::ModelInfo;
use crate::compactor::{ChatHistoryCompactor, DropOldestCompactor};
use crate::message::{Content, Message};
use crate::token::TokenCounter;
use crate::tool::{Toolbox, ToolDescription};

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
    
    // Optional toolbox for function/tool calling
    toolbox: Option<Box<dyn Toolbox>>,
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
            toolbox: None,
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
    
    /// Alias for push_message for better readability
    pub fn add_message(&mut self, msg: Message) {
        self.push_message(msg);
    }

    /// Sets a new compactor strategy at runtime
    pub fn set_compactor<C: ChatHistoryCompactor>(&mut self, comp: C) {
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
    
    /// Sets a toolbox for function/tool calling
    pub fn with_toolbox<T: Toolbox + 'static>(mut self, toolbox: T) -> Self {
        self.toolbox = Some(Box::new(toolbox));
        self
    }
    
    /// Sets a toolbox at runtime
    pub fn set_toolbox<T: Toolbox + 'static>(&mut self, toolbox: T) {
        self.toolbox = Some(Box::new(toolbox));
    }
    
    /// Gets the tool descriptions from the current toolbox
    pub fn tool_descriptions(&self) -> Vec<ToolDescription> {
        match &self.toolbox {
            Some(toolbox) => toolbox.describe(),
            None => Vec::new(),
        }
    }
    
    /// Processes tool calls from a message and adds tool response messages
    /// 
    /// This function examines an assistant message for tool calls, executes them
    /// using the current toolbox, and adds the tool response messages to the history.
    pub fn process_tool_calls(&mut self, assistant_message: &Message) -> crate::error::Result<()> {
        // If there's no toolbox or no tool calls, there's nothing to do
        if self.toolbox.is_none() || assistant_message.tool_calls.is_none() {
            return Ok(());
        }
        
        let tool_calls = assistant_message.tool_calls.as_ref().unwrap();
        
        // Process each tool call and collect responses first
        let mut responses = Vec::new();
        
        for tool_call in tool_calls {
            // Execute the tool call
            let result = self.toolbox.as_ref().unwrap().execute(
                &tool_call.function.name, 
                serde_json::from_str(&tool_call.function.arguments)?
            )?;
            
            // Create a tool response message
            let tool_message = Message::tool(tool_call.id.clone(), result);
            responses.push(tool_message);
        }
        
        // Now add all the responses to history
        for message in responses {
            self.add_message(message);
        }
        
        Ok(())
    }
    
    /// Clears the toolbox
    pub fn clear_toolbox(&mut self) {
        self.toolbox = None;
    }
    
    /// Returns true if the chat has a toolbox configured
    pub fn has_toolbox(&self) -> bool {
        self.toolbox.is_some()
    }
}
