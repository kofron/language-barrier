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
///
/// # Type Parameters
///
/// The `Tool` trait requires that implementors also implement `JsonSchema`, which is
/// used to automatically generate JSON schemas for tool parameters.
///
/// # Examples
///
/// ```
/// use language_barrier::Tool;
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
/// use language_barrier::{Tool, ToolDescription, Toolbox, Result};
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
///             _ => Err(language_barrier::Error::ToolNotFound(name.to_string())),
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
/// use language_barrier::{Tool, ToolCall, ToolDescription, Toolbox, TypedToolbox, Result};
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
///             _ => Err(language_barrier::Error::ToolNotFound(name.clone())),
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
/// use language_barrier::{Message, MessageRole, ToolCall, ToolCallView, TypedToolbox};
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
///     # fn parse_tool_call(&self, _: &ToolCall) -> language_barrier::Result<MyToolRequest> { unimplemented!() }
///     # fn execute_typed(&self, _: MyToolRequest) -> language_barrier::Result<String> { unimplemented!() }
/// }
///
/// impl language_barrier::Toolbox for MyToolbox {
///     # fn describe(&self) -> Vec<language_barrier::ToolDescription> { vec![] }
///     # fn execute(&self, _: &str, _: serde_json::Value) -> language_barrier::Result<String> { unimplemented!() }
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
    /// PhantomData to use the type parameter T
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
    ///
    /// This method is a convenient way to check if a message contains any tool calls
    /// before attempting to process them.
    ///
    /// # Returns
    ///
    /// `true` if the message contains at least one tool call, `false` otherwise.
    pub fn has_tool_calls(&self) -> bool {
        self.message.tool_calls.as_ref().map_or(false, |calls| !calls.is_empty())
    }
}