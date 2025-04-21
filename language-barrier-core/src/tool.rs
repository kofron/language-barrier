use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::{collections::HashMap, marker::PhantomData};
use thiserror::Error;

use crate::error::Result;
use crate::message::{Message, ToolCall};

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
pub trait Tool
where
    Self: JsonSchema,
{
    /// Returns the name of the tool
    ///
    /// This name is used to identify the tool in the LLM's API and must be unique
    /// within a toolbox.
    fn name(&self) -> &str;

    /// Returns the description of the tool
    ///
    /// This description is sent to the LLM to help it understand when and how
    /// to use the tool. It should be clear and concise.
    fn description(&self) -> &str;
}

/// This is the actual description that gets attached to a tool
///
/// `ToolDescription` contains all the information needed for an LLM to understand
/// and use a tool, including its name, description, and parameter schema.
///
/// This struct is primarily used internally by the `Toolbox` trait and is
/// converted to provider-specific formats when sending requests to the LLM.
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
///
/// The `Toolbox` trait provides a central place for defining and executing tools
/// without requiring generic type parameters, which allows it to be used easily
/// throughout the library.
///
/// This is the primary trait that the `Chat` struct works with, and it's used for
/// registering tools with LLMs and executing tool calls.
///
/// # Examples
///
/// ```
/// use language_barrier_core::{Tool, ToolDescription, Toolbox, Result};
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
/// use serde_json::Value;
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
///
/// struct MyToolbox;
///
/// impl Toolbox for MyToolbox {
///     fn describe(&self) -> Vec<ToolDescription> {
///         let weather_schema = schemars::schema_for!(WeatherRequest);
///         let weather_schema_value = serde_json::to_value(weather_schema.schema).unwrap();
///
///         vec![
///             ToolDescription {
///                 name: "get_weather".to_string(),
///                 description: "Get current weather for a location".to_string(),
///                 parameters: weather_schema_value,
///             },
///         ]
///     }
///
///     fn execute(&self, name: &str, arguments: Value) -> Result<String> {
///         match name {
///             "get_weather" => {
///                 let request: WeatherRequest = serde_json::from_value(arguments)?;
///                 let units = request.units.unwrap_or_else(|| "celsius".to_string());
///                 Ok(format!("Weather in {}: 22 degrees {}", request.location, units))
///             }
///             _ => Err(language_barrier_core::Error::ToolNotFound(name.to_string())),
///         }
///     }
/// }
/// ```
pub trait Toolbox {
    /// Returns descriptions of all tools in this toolbox
    ///
    /// These descriptions are used to register the tools with the LLM API.
    /// Each description should include a name, human-readable description,
    /// and JSON schema for the tool's parameters.
    fn describe(&self) -> Vec<ToolDescription>;

    /// Executes a tool call with the given input
    ///
    /// # Parameters
    ///
    /// * `name` - The name of the tool to execute
    /// * `arguments` - The arguments for the tool, as a JSON value
    ///
    /// # Returns
    ///
    /// The result of executing the tool, as a string that will be sent back to the LLM.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found or if execution fails.
    fn execute(&self, name: &str, arguments: Value) -> Result<String>;
}

/// A typed toolbox that works with specific tool types
///
/// The `TypedToolbox` trait extends `Toolbox` with type-safe methods for
/// parsing and executing tool calls. It's generic over a type parameter `T`,
/// which typically represents a union of all possible tool request types,
/// often implemented as an enum.
///
/// This trait is used in conjunction with the `ToolCallView` struct to provide
/// a type-safe interface for working with tool calls.
///
/// # Type Parameters
///
/// * `T` - The type that represents tool requests, typically an enum that can represent
///   all possible tool requests.
///
/// # Examples
///
/// ```
/// use language_barrier_core::{Tool, ToolCall, ToolDescription, Toolbox, TypedToolbox, Result};
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
/// use serde_json::Value;
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
///
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// enum MyToolRequest {
///     Weather(WeatherRequest),
/// }
///
/// struct MyToolbox;
///
/// impl Toolbox for MyToolbox {
///     fn describe(&self) -> Vec<ToolDescription> {
///         // Implementation details...
///         vec![]
///     }
///
///     fn execute(&self, name: &str, arguments: Value) -> Result<String> {
///         // Implementation details...
///         Ok("".to_string())
///     }
/// }
///
/// impl TypedToolbox<MyToolRequest> for MyToolbox {
///     fn parse_tool_call(&self, tool_call: &ToolCall) -> Result<MyToolRequest> {
///         let name = &tool_call.function.name;
///         let arguments = serde_json::from_str(&tool_call.function.arguments)?;
///
///         match name.as_str() {
///             "get_weather" => {
///                 let request: WeatherRequest = serde_json::from_value(arguments)?;
///                 Ok(MyToolRequest::Weather(request))
///             }
///             _ => Err(language_barrier_core::Error::ToolNotFound(name.clone())),
///         }
///     }
///
///     fn execute_typed(&self, request: MyToolRequest) -> Result<String> {
///         match request {
///             MyToolRequest::Weather(weather_req) => {
///                 let units = weather_req.units.unwrap_or_else(|| "celsius".to_string());
///                 Ok(format!("Weather in {}: 22 degrees {}", weather_req.location, units))
///             }
///         }
///     }
/// }
/// ```
pub trait TypedToolbox<T: DeserializeOwned>: Toolbox {
    /// Parses a tool call into a strongly typed representation
    ///
    /// This method converts an untyped `ToolCall` into a strongly typed `T`,
    /// which allows for pattern matching and type-safe processing.
    ///
    /// # Parameters
    ///
    /// * `tool_call` - The untyped tool call to parse
    ///
    /// # Returns
    ///
    /// The parsed tool call as a strongly typed `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails or if the tool is not recognized.
    fn parse_tool_call(&self, tool_call: &ToolCall) -> Result<T>;

    /// Executes a typed tool request and returns a typed response
    ///
    /// This method executes a strongly typed tool request and returns a string
    /// response that will be sent back to the LLM.
    ///
    /// # Parameters
    ///
    /// * `request` - The strongly typed tool request to execute
    ///
    /// # Returns
    ///
    /// The result of executing the tool, as a string that will be sent back to the LLM.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails.
    fn execute_typed(&self, request: T) -> Result<String>;
}

/// Errors that can occur when working with tools
#[derive(Error, Debug)]
pub enum ToolError {
    /// Tool with the specified name was not found in the registry
    #[error("Tool not found: {0}")]
    NotFound(String),

    /// Error generating JSON schema for tool
    #[error("Failed to generate schema for tool '{0}': {1}")]
    SchemaGenerationError(String, serde_json::Error),

    /// Error parsing arguments for tool
    #[error("Failed to parse arguments for tool '{0}': {1}")]
    ArgumentParsingError(String, serde_json::Error),

    /// Tool execution encountered an error
    #[error("Tool execution failed for tool '{0}': {1}")]
    ExecutionError(String, Box<dyn std::error::Error + Send + Sync>),

    /// Expected output type did not match actual output type
    #[error("Type mismatch after execution for tool '{0}': Expected different output type")]
    OutputTypeMismatch(String),
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

/// LLM-facing representation of a tool
#[derive(Serialize, Debug, Clone)]
pub struct LlmToolInfo {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Registry for storing tool definitions
pub struct ToolRegistry {
    pub(crate) tools: HashMap<String, ToolEntry>,
}

/// Internal structure to store a registered tool
pub(crate) struct ToolEntry {
    pub name: String,
    pub description: String,
    pub schema: Value,
}

impl ToolRegistry {
    /// Creates a new, empty tool registry
    pub fn new() -> Self {
        ToolRegistry {
            tools: HashMap::new(),
        }
    }

    /// Registers a tool with the registry
    pub fn register_tool<T>(&mut self, tool: T) -> Result<()>
    where
        T: ToolDefinition + Send + Sync + 'static,
    {
        let name = tool.name();
        let description = tool.description();
        let schema = tool.schema()?;

        let entry = ToolEntry {
            name: name.clone(),
            description,
            schema,
        };

        self.tools.insert(name, entry);
        Ok(())
    }

    /// Gets LLM-friendly descriptions of all registered tools
    pub fn get_tool_descriptions(&self) -> Vec<LlmToolInfo> {
        self.tools
            .values()
            .map(|entry| LlmToolInfo {
                name: entry.name.clone(),
                description: entry.description.clone(),
                parameters: entry.schema.clone(),
            })
            .collect()
    }

    /// Returns true if the registry has a tool with the given name
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Returns the number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns true if the registry has no registered tools
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Legacy wrapper for toolbox implementation over ToolRegistry
pub struct ToolRegistryAdapter {
    registry: ToolRegistry,
    // Maps for tools that have been registered with execution functions
    // This is temporary until we fully migrate to the runtime executor
    executors:
        HashMap<String, Box<dyn Fn(Value) -> std::result::Result<String, ToolError> + Send + Sync>>,
}

impl ToolRegistryAdapter {
    pub fn new(registry: ToolRegistry) -> Self {
        ToolRegistryAdapter {
            registry,
            executors: HashMap::new(),
        }
    }

    /// Registers an executor function for a tool name
    pub fn register_executor(
        &mut self,
        name: &str,
        executor: impl Fn(Value) -> std::result::Result<String, ToolError> + Send + Sync + 'static,
    ) {
        self.executors.insert(name.to_string(), Box::new(executor));
    }
}

impl Toolbox for ToolRegistryAdapter {
    fn describe(&self) -> Vec<ToolDescription> {
        self.registry
            .get_tool_descriptions()
            .into_iter()
            .map(|info| ToolDescription {
                name: info.name,
                description: info.description,
                parameters: info.parameters,
            })
            .collect()
    }

    fn execute(&self, name: &str, arguments: Value) -> Result<String> {
        match self.executors.get(name) {
            Some(executor) => executor(arguments)
                .map_err(|e| crate::Error::ToolExecutionError(format!("{}: {}", name, e))),
            None => Err(crate::Error::ToolNotFound(name.to_string())),
        }
    }
}

/// A view over a message that provides typed access to tool calls
///
/// The `ToolCallView` struct provides a type-safe way to work with tool calls
/// in a message. It takes a reference to a message and a toolbox, and provides
/// methods for extracting and working with strongly typed tool calls.
///
/// This implements the "view pattern", allowing type-safe access to tool calls
/// without requiring the core message types to be generic.
///
/// # Type Parameters
///
/// * `T` - The type that represents tool requests, typically an enum that can represent
///   all possible tool requests.
/// * `TB` - The type of the toolbox, which must implement `TypedToolbox<T>`.
///
/// # Examples
///
/// ```
/// use language_barrier_core::{Message, ToolCall, ToolCallView, TypedToolbox};
/// use serde::{Deserialize, Serialize};
///
/// // Assuming MyToolbox and MyToolRequest are defined as in previous examples
/// struct MyToolbox;
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// enum MyToolRequest {
///     // ...
/// }
///
/// impl TypedToolbox<MyToolRequest> for MyToolbox {
///     // Implementation details...
///     # fn parse_tool_call(&self, _: &ToolCall) -> language_barrier_core::Result<MyToolRequest> { unimplemented!() }
///     # fn execute_typed(&self, _: MyToolRequest) -> language_barrier_core::Result<String> { unimplemented!() }
/// }
///
/// impl language_barrier_core::Toolbox for MyToolbox {
///     # fn describe(&self) -> Vec<language_barrier_core::ToolDescription> { vec![] }
///     # fn execute(&self, _: &str, _: serde_json::Value) -> language_barrier_core::Result<String> { unimplemented!() }
/// }
///
/// // Create a message with tool calls
/// let mut message = Message::assistant("I'll check the weather for you.");
///
/// // Create a toolbox
/// let toolbox = MyToolbox;
///
/// // Create a view over the message
/// let view = ToolCallView::<MyToolRequest, _>::new(&toolbox, &message);
///
/// // Check if the message has tool calls
/// if view.has_tool_calls() {
///     // Process the tool calls
///     // let tool_requests = view.tool_calls().unwrap();
///     // ...
/// }
/// ```
pub struct ToolCallView<'a, T, TB>
where
    T: DeserializeOwned,
    TB: TypedToolbox<T>,
{
    /// The toolbox for parsing and executing tool calls
    toolbox: &'a TB,
    /// The message containing tool calls
    message: &'a Message,
    /// `PhantomData` to use the type parameter T
    _phantom: PhantomData<T>,
}

impl<'a, T, TB> ToolCallView<'a, T, TB>
where
    T: DeserializeOwned,
    TB: TypedToolbox<T>,
{
    /// Creates a new view over a message
    ///
    /// # Parameters
    ///
    /// * `toolbox` - The toolbox to use for parsing tool calls
    /// * `message` - The message to create a view over
    ///
    /// # Returns
    ///
    /// A new `ToolCallView` instance that provides typed access to the tool calls in the message.
    pub fn new(toolbox: &'a TB, message: &'a Message) -> Self {
        Self {
            toolbox,
            message,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns typed tool calls from this message
    ///
    /// This method extracts all tool calls from the message and parses them into
    /// strongly typed representations using the toolbox.
    ///
    /// # Returns
    ///
    /// A vector of strongly typed tool requests (`T`), or an empty vector if the message
    /// doesn't contain any tool calls.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing any of the tool calls fails.
    pub fn tool_calls(&self) -> Result<Vec<T>> {
        match self.message {
            Message::Assistant { tool_calls, .. } => {
                let mut result = Vec::with_capacity(tool_calls.len());
                for call in tool_calls {
                    result.push(self.toolbox.parse_tool_call(call)?);
                }
                Ok(result)
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Returns true if this message contains tool calls
    ///
    /// This method is a convenient way to check if a message contains any tool calls
    /// before attempting to process them.
    ///
    /// # Returns
    ///
    /// `true` if the message contains at least one tool call, `false` otherwise.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        match self.message {
            Message::Assistant { tool_calls, .. } => !tool_calls.is_empty(),
            _ => false,
        }
    }
}
