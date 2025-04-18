use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{Error, Result};

/// Represents the schema for a parameter in a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    /// The type of the parameter (string, number, boolean, etc.)
    pub r#type: String,
    /// Optional description of the parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// For enum types, the possible values
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Whether the parameter is required
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
}

/// Represents the schema for a tool's parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParametersSchema {
    /// The type of the parameters object (usually "object")
    pub r#type: String,
    /// The properties of the parameters object
    pub properties: HashMap<String, ParameterSchema>,
    /// The list of required parameters
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub required: Vec<String>,
}

/// Defines the contract for tools that can be used by LLMs
///
/// Tools provide a standardized way to extend language models with custom functionality.
/// Each tool defines its name, description, parameter schema, and execution logic.
///
/// # Examples
///
/// ```
/// use language_barrier::tool::{Tool, ToolParametersSchema, calculator};
/// use language_barrier::error::Result;
/// use std::collections::HashMap;
/// use serde_json::json;
///
/// # async fn example() -> Result<()> {
/// // Create a calculator tool
/// let calc = calculator();
///
/// // Check its metadata
/// assert_eq!(calc.name(), "calculator");
/// assert!(calc.description().contains("arithmetic"));
///
/// // Execute the tool with parameters
/// let params = json!({
///     "operation": "add",
///     "a": 5,
///     "b": 3
/// });
///
/// let result = calc.execute(params).await?;
/// assert_eq!(result["result"], 8.0);
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait Tool: Send + Sync + Debug {
    /// Returns the name of the tool
    fn name(&self) -> &str;
    
    /// Returns the description of the tool
    fn description(&self) -> &str;
    
    /// Returns the parameters schema for the tool
    fn parameters_schema(&self) -> ToolParametersSchema;
    
    /// Executes the tool with the given parameters
    async fn execute(&self, parameters: serde_json::Value) -> Result<serde_json::Value>;
    
    /// Converts the tool to a provider-specific format (defaults to OpenAI format)
    fn to_provider_format(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": self.parameters_schema(),
            }
        })
    }
}

/// A simple calculator tool that can perform basic arithmetic operations
///
/// # Examples
///
/// ```
/// use language_barrier::tool::{calculator, Tool};
/// use language_barrier::client::{LlmProvider, MockProvider, GenerationOptions};
/// use language_barrier::message::{Message, MessageRole, Content};
/// use language_barrier::chat::{Chat, ChatOptions};
/// use std::sync::Arc;
///
/// # async fn example() -> language_barrier::error::Result<()> {
/// // Create a mock provider
/// let mut provider = MockProvider::new();
///
/// // Register the calculator tool
/// provider.register_tool(Box::new(calculator())).await?;
///
/// // Use the tool in a generation request
/// let messages = vec![
///     Message::user("Please calculate 10 / 2"),
/// ];
///
/// let mut options = GenerationOptions::default();
/// // Create tool definitions for the request
/// let calc = calculator();
/// options.tool_definitions = Some(vec![calc.to_provider_format()]);
/// options.tool_names = Some(vec![calc.name().to_string()]);
///
/// let result = provider.generate("mock-model", &messages, options).await?;
///
/// // For high-level usage, use the Chat interface
/// let provider = Arc::new(provider);
/// let mut options = ChatOptions::default();
/// // Add the tool to the Chat
/// options = options.with_tools(vec![Box::new(calculator())]);
///
/// let mut chat = Chat::new(provider, options).await?;
/// let response = chat.send_message_with_tools("Calculate 5 * 3").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Calculator;

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }
    
    fn description(&self) -> &str {
        "Performs basic arithmetic calculations. Supports add, subtract, multiply, and divide operations."
    }
    
    fn parameters_schema(&self) -> ToolParametersSchema {
        let mut properties = HashMap::new();
        
        properties.insert(
            "operation".to_string(),
            ParameterSchema {
                r#type: "string".to_string(),
                description: Some("The operation to perform (add, subtract, multiply, divide)".to_string()),
                enum_values: Some(vec![
                    "add".to_string(),
                    "subtract".to_string(),
                    "multiply".to_string(),
                    "divide".to_string(),
                ]),
                required: Some(true),
            },
        );
        
        properties.insert(
            "a".to_string(),
            ParameterSchema {
                r#type: "number".to_string(),
                description: Some("The first number".to_string()),
                enum_values: None,
                required: Some(true),
            },
        );
        
        properties.insert(
            "b".to_string(),
            ParameterSchema {
                r#type: "number".to_string(),
                description: Some("The second number".to_string()),
                enum_values: None,
                required: Some(true),
            },
        );
        
        ToolParametersSchema {
            r#type: "object".to_string(),
            properties,
            required: vec!["operation".to_string(), "a".to_string(), "b".to_string()],
        }
    }
    
    async fn execute(&self, parameters: serde_json::Value) -> Result<serde_json::Value> {
        let operation = parameters["operation"]
            .as_str()
            .ok_or_else(|| Error::InvalidToolParameter("operation must be a string".to_string()))?;
            
        let a = parameters["a"]
            .as_f64()
            .ok_or_else(|| Error::InvalidToolParameter("a must be a number".to_string()))?;
            
        let b = parameters["b"]
            .as_f64()
            .ok_or_else(|| Error::InvalidToolParameter("b must be a number".to_string()))?;
            
        let result = match operation {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Err(Error::InvalidToolParameter("Cannot divide by zero".to_string()));
                }
                a / b
            },
            _ => return Err(Error::InvalidToolParameter(format!("Unknown operation: {}", operation))),
        };
        
        Ok(serde_json::json!({ "result": result }))
    }
}

/// Helper function to create a calculator tool
pub fn calculator() -> Calculator {
    Calculator
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_calculator_schema() {
        let calc = Calculator;
        let schema = calc.parameters_schema();
        
        assert_eq!(schema.r#type, "object");
        assert_eq!(schema.properties.len(), 3);
        assert!(schema.properties.contains_key("operation"));
        assert!(schema.properties.contains_key("a"));
        assert!(schema.properties.contains_key("b"));
        assert_eq!(schema.required.len(), 3);
    }
    
    #[tokio::test]
    async fn test_calculator_execution() {
        let calc = Calculator;
        
        // Test addition
        let add_params = serde_json::json!({
            "operation": "add",
            "a": 5,
            "b": 3
        });
        
        let result = calc.execute(add_params).await.unwrap();
        assert_eq!(result["result"], 8.0);
        
        // Test division
        let div_params = serde_json::json!({
            "operation": "divide",
            "a": 10,
            "b": 2
        });
        
        let result = calc.execute(div_params).await.unwrap();
        assert_eq!(result["result"], 5.0);
        
        // Test division by zero
        let div_zero_params = serde_json::json!({
            "operation": "divide",
            "a": 10,
            "b": 0
        });
        
        let result = calc.execute(div_zero_params).await;
        assert!(result.is_err());
        
        // Test invalid operation
        let invalid_params = serde_json::json!({
            "operation": "power",
            "a": 2,
            "b": 3
        });
        
        let result = calc.execute(invalid_params).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_tool_provider_format() {
        let calc = Calculator;
        let provider_format = calc.to_provider_format();
        
        assert_eq!(provider_format["type"], "function");
        assert_eq!(provider_format["function"]["name"], "calculator");
        assert!(provider_format["function"]["description"].as_str().unwrap().contains("arithmetic"));
        assert!(provider_format["function"]["parameters"].is_object());
    }
}