use crate::ModelInfo;
use crate::compactor::{ChatHistoryCompactor, DropOldestCompactor};

use crate::message::{Content, Message};
use crate::token::TokenCounter;
use crate::tool::Tool;

/// The main Chat client that users will interact with
pub struct Chat<M> {
    // Immutable after construction
    model: M,

    // Tunable knobs / state
    system_prompt: String,
    max_output_tokens: usize,
    tools: Vec<Box<dyn Tool>>,

    // History and token tracking
    history: Vec<Message>,
    token_counter: TokenCounter,
    compactor: Box<dyn ChatHistoryCompactor>,
}

impl<M> Chat<M>
where
    M: Clone + Send + Sync + 'static,
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
            tools: Vec::new(),
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
        self.compactor
            .compact(&mut self.history, &mut self.token_counter, MAX_TOKENS);
    }

    /// Gets the current token count
    pub fn tokens_used(&self) -> usize {
        self.token_counter.total()
    }
}
