// This is the main library file that re-exports the public API
// and defines the module structure.

pub mod error;
pub mod message;
pub mod model;
pub mod provider;

// Re-export the main types for convenient usage
pub use error::Error;
pub use message::{Content, Message, MessageRole};
pub use model::{Model, ModelCapability, ModelFamily};
pub use provider::Provider;