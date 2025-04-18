// This is the main library file that re-exports the public API
// and defines the module structure.

pub mod chat;
pub mod client;
pub mod error;
pub mod message;
pub mod model;
pub mod provider;
pub mod secret;
pub mod token;
pub mod compactor;

// Re-export the main types for convenient usage
pub use chat::{Chat, ChatBuilder, ChatMessage, ChatResponse};
pub use client::{LlmProvider, ToolChoice};
pub use error::Error;
pub use message::{Content, Message, MessageRole};
pub use model::{Model, ModelCapability, ModelFamily};
pub use provider::Provider;
pub use secret::Secret;
pub use token::TokenCounter;
pub use compactor::{ChatHistoryCompactor, DropOldestCompactor};