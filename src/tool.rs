use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Defines the contract for tools that can be used by LLMs
///
/// Tools provide a standardized way to extend language models with custom functionality.
/// Each tool defines its name, description, parameter schema, and execution logic.f
/// ```
#[async_trait]
pub trait Tool: Send + Sync + Debug {
    type Input<'a>: Serialize + Deserialize<'a> + Send + Sync;
    type Output<'a>: Serialize + Deserialize<'a> + Send + Sync;
    type Error: std::error::Error + Send + Sync;

    /// Returns the name of the tool
    fn name(&self) -> &str;

    /// Returns the description of the tool
    fn description(&self) -> &str;
}
