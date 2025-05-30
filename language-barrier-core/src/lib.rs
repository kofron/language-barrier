// This is the main library file that re-exports the public API
// and defines the module structure.

pub mod chat;
pub mod compactor;
pub mod error;
pub mod message;
pub mod model;
pub mod provider;
pub mod secret;
pub mod token;
pub mod tool;

// Re-export the main types for convenient usage
pub use chat::Chat;
pub use compactor::{ChatHistoryCompactor, DropOldestCompactor};
pub use error::{Error, Result, ToolError};
pub use llm_service::{HTTPLlmService, LLMService};
pub use message::{Content, Message, ToolCall};
pub use model::{Claude, Gemini, Mistral, ModelInfo, OpenAi};
pub use secret::Secret;
pub use token::TokenCounter;
pub use tool::{LlmToolInfo, Tool, ToolDefinition};
pub mod llm_service;
