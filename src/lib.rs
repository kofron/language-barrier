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
pub mod transport;

// Re-export the main types for convenient usage
pub use chat::Chat;
pub use compactor::{ChatHistoryCompactor, DropOldestCompactor};
pub use error::Error;
pub use message::{Content, Message, MessageRole};
pub use model::{Claude, ModelInfo};

pub use secret::Secret;
pub use token::TokenCounter;
