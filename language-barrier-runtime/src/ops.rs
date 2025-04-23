use std::collections::HashMap;
use std::fmt;

use language_barrier_core::{
    chat::Chat,
    error::Result,
    message::{Content, Message, ToolCall},
};
use std::marker::Send;

/// Tool execution result
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}

pub enum ExecuteToolBehavior {
    Break,
    AutoContinue,
}

/// Free monad operations for the LLM runtime
pub enum LlmOp<Next> {
    GenerateNextMessage {
        chat: Chat,
        next: Box<dyn FnOnce(Result<Chat>) -> Next + Send>,
    },
    /// Execute a specific tool call
    ExecuteTool {
        tool_call: ToolCall,
        next: Box<dyn FnOnce(Result<ToolResult>) -> Next + Send>,
    },
    /// Terminal operation
    Done { result: Result<Chat> },
}

impl<Next: fmt::Debug> fmt::Debug for LlmOp<Next> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmOp::GenerateNextMessage { chat, .. } => f
                .debug_struct("GenerateNextMessage")
                .field("chat", chat)
                .field("next", &"<function>")
                .finish(),
            LlmOp::ExecuteTool { tool_call, .. } => f
                .debug_struct("ExecuteTool")
                .field("tool_call", tool_call)
                .field("next", &"<function>")
                .finish(),
            LlmOp::Done { result } => f.debug_struct("Done").field("result", result).finish(),
        }
    }
}

/// The free monad wrapper
#[derive(Debug)]
pub struct LlmM<A> {
    pub op: Option<LlmOp<LlmM<A>>>,
    pub result: Option<A>,
}

impl<A: 'static> LlmM<A> {
    /// Creates a new LlmM with an operation
    pub fn new(op: LlmOp<LlmM<A>>) -> Self {
        Self {
            op: Some(op),
            result: None,
        }
    }

    /// Creates a new LlmM with a result value
    pub fn pure(value: A) -> Self {
        Self {
            op: None,
            result: Some(value),
        }
    }

    /// Monadic bind operation
    pub fn and_then<B: 'static, F>(self, f: F) -> LlmM<B>
    where
        F: 'static + FnOnce(A) -> LlmM<B> + Send,
    {
        match (self.op, self.result) {
            (None, Some(result)) => f(result),
            (Some(op), None) => match op {
                LlmOp::GenerateNextMessage { chat, next } => {
                    LlmM::new(LlmOp::GenerateNextMessage {
                        chat,
                        next: Box::new(move |res| next(res).and_then(f)),
                    })
                }
                LlmOp::ExecuteTool { tool_call, next } => LlmM::new(LlmOp::ExecuteTool {
                    tool_call,
                    next: Box::new(move |res| next(res).and_then(f)),
                }),
                LlmOp::Done { result } => LlmM::new(LlmOp::Done { result }),
            },
            _ => panic!("Invalid LlmM state: both op and result are None or Some"),
        }
    }

    /// Maps a function over the result
    pub fn map<B: 'static, F>(self, f: F) -> LlmM<B>
    where
        F: 'static + FnOnce(A) -> B + Send,
    {
        self.and_then(move |a| LlmM::pure(f(a)))
    }
}

// Helper functions to create operations
pub fn generate_next_message(chat: Chat) -> LlmM<Result<Chat>> {
    LlmM::new(LlmOp::GenerateNextMessage {
        chat,
        next: Box::new(LlmM::pure),
    })
}

pub fn execute_tool(tool_call: ToolCall) -> LlmM<Result<ToolResult>> {
    LlmM::new(LlmOp::ExecuteTool {
        tool_call,
        next: Box::new(LlmM::pure),
    })
}

pub fn done(result: Result<Chat>) -> LlmM<Result<Chat>> {
    LlmM::new(LlmOp::Done { result })
}

/// Helper to create a user message from a string
pub fn user_message(text: impl Into<String>) -> Message {
    Message::User {
        content: Content::Text(text.into()),
        metadata: HashMap::new(),
        name: None,
    }
}

/// Operation to add a message to a chat
pub fn add_message(chat: Chat, message: Message) -> LlmM<Result<Chat>> {
    LlmM::pure(Ok(chat.add_message(message)))
}
