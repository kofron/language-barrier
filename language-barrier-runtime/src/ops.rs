use std::fmt;

use language_barrier_core::{
    chat::Chat,
    error::Result,
    message::{Content, Message, ToolCall},
    model::ModelInfo,
    tool::LlmToolInfo,
};
use std::{collections::HashMap, marker::Send};

/// Tool execution result
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}

/// Free monad operations for the LLM runtime
pub enum LlmOp<Next> {
    /// Send a message to the LLM with optional tools
    Chat {
        messages: Vec<Message>,
        tools: Option<Vec<LlmToolInfo>>,
        next: Box<dyn FnOnce(Result<Message>) -> Next + Send>,
    },
    /// Execute a specific tool call
    ExecuteTool {
        tool_call: ToolCall,
        next: Box<dyn FnOnce(Result<ToolResult>) -> Next + Send>,
    },
    /// Add a message to an existing Chat instance
    AddMessage {
        chat: Box<dyn std::any::Any + Send>,
        message: Message,
        next: Box<dyn FnOnce(Result<Box<dyn std::any::Any + Send>>) -> Next + Send>,
    },
    /// Terminal operation
    Done { result: Result<Message> },
}

impl<Next: fmt::Debug> fmt::Debug for LlmOp<Next> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmOp::Chat {
                messages, tools, ..
            } => f
                .debug_struct("Chat")
                .field("messages", messages)
                .field("tools", tools)
                .field("next", &"<function>")
                .finish(),
            LlmOp::ExecuteTool { tool_call, .. } => f
                .debug_struct("ExecuteTool")
                .field("tool_call", tool_call)
                .field("next", &"<function>")
                .finish(),
            LlmOp::AddMessage { message, .. } => f
                .debug_struct("AddMessage")
                .field("message", message)
                .field("chat", &"<boxed_chat>")
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
                LlmOp::Chat {
                    messages,
                    tools,
                    next,
                } => LlmM::new(LlmOp::Chat {
                    messages,
                    tools,
                    next: Box::new(move |res| next(res).and_then(f)),
                }),
                LlmOp::ExecuteTool { tool_call, next } => LlmM::new(LlmOp::ExecuteTool {
                    tool_call,
                    next: Box::new(move |res| next(res).and_then(f)),
                }),
                LlmOp::AddMessage {
                    chat,
                    message,
                    next,
                } => LlmM::new(LlmOp::AddMessage {
                    chat,
                    message,
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
pub fn chat(messages: Vec<Message>, tools: Option<Vec<LlmToolInfo>>) -> LlmM<Result<Message>> {
    LlmM::new(LlmOp::Chat {
        messages,
        tools,
        next: Box::new(LlmM::pure),
    })
}

pub fn execute_tool(tool_call: ToolCall) -> LlmM<Result<ToolResult>> {
    LlmM::new(LlmOp::ExecuteTool {
        tool_call,
        next: Box::new(LlmM::pure),
    })
}

/// Add a message to an existing Chat instance
///
/// This operation takes a Chat instance and a Message, returning the updated Chat instance
/// with the message added immutably. This keeps all Chat interactions immutable.
///
/// # Arguments
///
/// * `chat` - The Chat instance to add the message to
/// * `message` - The Message to add to the Chat
///
/// # Returns
///
/// A Result containing the updated Chat instance
///
/// # Examples
///
/// ```
/// use language_barrier_core::{Chat, model::Claude, message::Message};
/// use language_barrier_runtime::ops;
///
/// let chat = Chat::new(Claude::default())
///     .with_system_prompt("You are helpful assistant");
///
/// // Create a user message
/// let message = Message::user("Hello, how can you help me?");
///
/// // Add the message to the chat
/// let program = ops::add_message(chat, message);
/// ```
pub fn add_message<M: 'static + ModelInfo + Clone + Send>(
    chat: Chat<M>,
    message: Message,
) -> LlmM<Result<Chat<M>>> {
    // First box the chat so we can send it through the operation
    let boxed_chat: Box<dyn std::any::Any + Send> = Box::new(chat);

    // Create the operation
    LlmM::new(LlmOp::AddMessage {
        chat: boxed_chat,
        message,
        next: Box::new(move |result: Result<Box<dyn std::any::Any + Send>>| {
            // Unbox the chat and return it
            match result {
                Ok(boxed) => {
                    // Try to downcast to the specific Chat type
                    match boxed.downcast::<Chat<M>>() {
                        Ok(chat) => LlmM::pure(Ok(*chat)),
                        Err(_) => LlmM::pure(Err(language_barrier_core::error::Error::Other(
                            "Could not downcast chat".into(),
                        ))),
                    }
                }
                Err(e) => LlmM::pure(Err(e)),
            }
        }),
    })
}

pub fn done(result: Result<Message>) -> LlmM<Result<Message>> {
    LlmM::new(LlmOp::Done { result })
}

// Message creation helpers

/// Create a user message with the given text content
///
/// This is a helper function for creating a user message with text content.
///
/// # Arguments
///
/// * `text` - The text content of the message
///
/// # Returns
///
/// A user message with the given text content
///
/// # Examples
///
/// ```
/// use language_barrier_runtime::ops;
///
/// let message = ops::user_message("Hello, how can you help me?");
/// ```
pub fn user_message(text: impl Into<String>) -> Message {
    Message::User {
        content: Content::Text(text.into()),
        metadata: HashMap::new(),
        name: None,
    }
}

/// Create a user message with the given text content and name
///
/// This is a helper function for creating a user message with text content and a name.
///
/// # Arguments
///
/// * `text` - The text content of the message
/// * `name` - The name to associate with the message
///
/// # Returns
///
/// A user message with the given text content and name
///
/// # Examples
///
/// ```
/// use language_barrier_runtime::ops;
///
/// let message = ops::named_user_message("Hello, how can you help me?", "John");
/// ```
pub fn named_user_message(text: impl Into<String>, name: impl Into<String>) -> Message {
    Message::User {
        content: Content::Text(text.into()),
        metadata: HashMap::new(),
        name: Some(name.into()),
    }
}

/// Create a system message with the given text content
///
/// This is a helper function for creating a system message with text content.
///
/// # Arguments
///
/// * `text` - The text content of the message
///
/// # Returns
///
/// A system message with the given text content
///
/// # Examples
///
/// ```
/// use language_barrier_runtime::ops;
///
/// let message = ops::system_message("You are a helpful assistant.");
/// ```
pub fn system_message(text: impl Into<String>) -> Message {
    Message::System {
        content: text.into(),
        metadata: HashMap::new(),
    }
}
