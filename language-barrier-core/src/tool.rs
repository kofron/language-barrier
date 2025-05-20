#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::error::Result;

/// Defines the contract for tools that can be used by LLMs
///
/// Tools provide a standardized way to extend language models with custom functionality.
/// Each tool defines its name, description, parameter schema, and execution logic.
///
/// # Type Parameters
///
/// The `Tool` trait requires that implementors also implement `JsonSchema`, which is
/// used to automatically generate JSON schemas for tool parameters.
///
/// # Examples
///
/// ```
/// use language_barrier_core::Tool;
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
/// struct WeatherRequest {
///     location: String,
///     units: Option<String>,
/// }
///
/// impl Tool for WeatherRequest {
///     fn name(&self) -> &str {
///         "get_weather"
///     }
///
///     fn description(&self) -> &str {
///         "Get current weather for a location"
///     }
/// }
/// ```
#[cfg(feature = "schema")]
pub trait Tool
where
    Self: JsonSchema,
{
    /// Returns the name of the tool
    ///
    /// This name is used to identify the tool in the LLM's API.
    fn name(&self) -> &str;

    /// Returns the description of the tool
    ///
    /// This description is sent to the LLM to help it understand when and how
    /// to use the tool. It should be clear and concise.
    fn description(&self) -> &str;
}

#[cfg(not(feature = "schema"))]
pub trait Tool {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
}

/// Core trait for defining tools with associated input and output types
///
/// This trait provides a more flexible and type-safe way to define tools compared
/// to the original `Tool` trait. It uses associated types to specify the input and output
/// types for a tool, allowing for better compile-time type checking.
///
/// # Type Parameters
///
/// * `Input` - The type of input that this tool accepts, must implement `DeserializeOwned` and `JsonSchema`
/// * `Output` - The type of output that this tool produces, must implement `Serialize`
///
/// # Examples
///
/// ```
/// use language_barrier_core::ToolDefinition;
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Clone, Deserialize, JsonSchema)]
/// struct WeatherRequest {
///     location: String,
///     units: Option<String>,
/// }
///
/// #[derive(Debug, Clone, Serialize)]
/// struct WeatherResponse {
///     temperature: f64,
///     condition: String,
///     location: String,
///     units: String,
/// }
///
/// struct WeatherTool;
///
/// impl ToolDefinition for WeatherTool {
///     type Input = WeatherRequest;
///     type Output = WeatherResponse;
///
///     fn name(&self) -> String {
///         "get_weather".to_string()
///     }
///
///     fn description(&self) -> String {
///         "Get current weather for a location".to_string()
///     }
/// }
/// ```
#[cfg(feature = "schema")]
pub trait ToolDefinition {
    /// The input type that this tool accepts
    type Input: DeserializeOwned + JsonSchema + Send + Sync + 'static;

    /// The output type that this tool produces
    type Output: Serialize + Send + Sync + 'static;

    /// Returns the name of the tool
    fn name(&self) -> String;

    /// Returns the description of the tool
    fn description(&self) -> String;

    /// Helper to generate the JSON schema for the input type
    fn schema(&self) -> Result<Value> {
        let schema = schemars::schema_for!(Self::Input);
        serde_json::to_value(schema.schema)
            .map_err(|e| crate::Error::Other(format!("Schema generation failed: {}", e)))
    }
}

#[cfg(not(feature = "schema"))]
pub trait ToolDefinition {
    type Input: DeserializeOwned + Send + Sync + 'static;
    type Output: Serialize + Send + Sync + 'static;
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn schema(&self) -> Result<Value> {
        Err(crate::Error::Other(
            "Schema generation requires the `schema` feature".to_string(),
        ))
    }
}

/// LLM-facing representation of a tool
#[derive(Serialize, Debug, Clone)]
pub struct LlmToolInfo {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Represents the tool choice strategy for LLMs
///
/// This enum provides a provider-agnostic API for tool choice, mapping to
/// different provider-specific parameters:
///
/// - OpenAI/Mistral: Maps to "auto", "required", "none", or a function object
/// - Anthropic: Maps to "auto", "any", "none", or a function object
/// - Gemini: Maps to function_calling_config modes and allowed_function_names
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// Allow the model to choose which tool to use (or none)
    /// 
    /// - OpenAI/Mistral: "auto"
    /// - Anthropic: "auto"
    /// - Gemini: mode="auto"
    Auto,
    /// Require the model to use one of the available tools
    /// 
    /// - OpenAI/Mistral: "required"
    /// - Anthropic: "any"
    /// - Gemini: mode="any"
    Any,
    /// Force the model not to use any tools
    /// 
    /// - OpenAI/Mistral: "none"
    /// - Anthropic: "none"
    /// - Gemini: mode="none"
    None,
    /// Require the model to use a specific tool by name
    /// 
    /// - OpenAI/Mistral: Object with type="function" and function.name
    /// - Anthropic: Object with type="function" and function.name
    /// - Gemini: mode="auto" with allowed_function_names=[name]
    Specific(String),
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Auto
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tool_choice_default() {
        let choice = ToolChoice::default();
        assert_eq!(choice, ToolChoice::Auto);
    }
    
    #[test]
    fn test_tool_choice_specific() {
        let choice = ToolChoice::Specific("weather".to_string());
        assert!(matches!(choice, ToolChoice::Specific(name) if name == "weather"));
    }
}
