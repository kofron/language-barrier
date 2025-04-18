use schemars::JsonSchema;
use serde_json::Value;

use crate::error::Result;

/// Defines the contract for tools that can be used by LLMs
///
/// Tools provide a standardized way to extend language models with custom functionality.
/// Each tool defines its name, description, parameter schema, and execution logic.
pub trait Tool
where
    Self: JsonSchema,
{
    /// Returns the name of the tool
    fn name(&self) -> &str;

    /// Returns the description of the tool
    fn description(&self) -> &str;
}

/// This is the actual description that gets attached to tool
pub struct ToolDescription {
    name: String,
    description: String,
    parameters: Value,
}

pub trait Toolbox<T> {
    fn describe(&self) -> Vec<ToolDescription>;
    fn tool(&self, input: Value) -> Result<T>;
}
