// This is the main library file that re-exports the public API
// and defines the module structure.

pub mod chat;
pub mod compactor;
pub mod error;
pub mod executor;
pub mod message;
pub mod model;
pub mod provider;
pub mod secret;
pub mod token;
pub mod tool;

// Re-export the main types for convenient usage
pub use chat::Chat;
pub use compactor::{ChatHistoryCompactor, DropOldestCompactor};
pub use error::{Error, Result};
pub use executor::SingleRequestExecutor;
pub use message::{Content, Message, ToolCall};
pub use model::{Claude, GPT, Gemini, Mistral, ModelInfo};
pub use secret::Secret;
pub use token::TokenCounter;
pub use tool::{Tool, ToolCallView, ToolDescription, Toolbox, TypedToolbox};
