use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::marker::PhantomData;

use crate::error::Result;
use crate::message::{Message, ToolCall};

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

/// This is the actual description that gets attached to a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescription {
    /// The name of the tool
    pub name: String,
    /// The description of the tool
    pub description: String,
    /// The JSON schema for the tool parameters
    pub parameters: Value,
}

/// A non-generic toolbox trait that works with untyped JSON values
pub trait Toolbox {
    /// Returns descriptions of all tools in this toolbox
    fn describe(&self) -> Vec<ToolDescription>;
    
    /// Executes a tool call with the given input
    fn execute(&self, name: &str, arguments: Value) -> Result<String>;
}

/// A typed toolbox that works with specific tool types
pub trait TypedToolbox<T: DeserializeOwned>: Toolbox {
    /// Parses a tool call into a strongly typed representation
    fn parse_tool_call(&self, tool_call: &ToolCall) -> Result<T>;
    
    /// Executes a typed tool request and returns a typed response
    fn execute_typed(&self, request: T) -> Result<String>;
}

/// A view over a message that provides typed access to tool calls
pub struct ToolCallView<'a, T, TB>
where
    T: DeserializeOwned,
    TB: TypedToolbox<T>,
{
    /// The toolbox for parsing and executing tool calls
    toolbox: &'a TB,
    /// The message containing tool calls
    message: &'a Message,
    /// PhantomData to use the type parameter T
    _phantom: PhantomData<T>,
}

impl<'a, T, TB> ToolCallView<'a, T, TB>
where
    T: DeserializeOwned,
    TB: TypedToolbox<T>,
{
    /// Creates a new view over a message
    pub fn new(toolbox: &'a TB, message: &'a Message) -> Self {
        Self { 
            toolbox, 
            message,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Returns typed tool calls from this message
    pub fn tool_calls(&self) -> Result<Vec<T>> {
        match &self.message.tool_calls {
            Some(calls) => {
                let mut result = Vec::with_capacity(calls.len());
                for call in calls {
                    result.push(self.toolbox.parse_tool_call(call)?);
                }
                Ok(result)
            }
            None => Ok(Vec::new()),
        }
    }
    
    /// Returns true if this message contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.message.tool_calls.as_ref().map_or(false, |calls| !calls.is_empty())
    }
}
