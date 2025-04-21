use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::{
    any::Any, // Import Any
    collections::HashMap,
    sync::Arc,
};
use thiserror::Error;

// --- Error Type (mostly unchanged) ---
#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),
    #[error("Failed to generate schema for tool '{0}': {1}")]
    SchemaGenerationError(String, serde_json::Error),
    #[error("Failed to parse arguments for tool '{0}': {1}")]
    ArgumentParsingError(String, serde_json::Error),
    // No longer needed: OutputSerializationError
    // #[error("Failed to serialize output for tool '{0}': {1}")]
    // OutputSerializationError(String, serde_json::Error),
    #[error("Tool execution failed for tool '{0}': {1}")]
    ExecutionError(String, Box<dyn std::error::Error + Send + Sync>),
    #[error("Type mismatch after execution for tool '{0}': Expected different output type")]
    OutputTypeMismatch(String), // New error for downcast failure
}

// --- Core Tool Definition (unchanged, but Output needs 'static) ---
pub trait ToolDefinition {
    type Input: DeserializeOwned + JsonSchema + Send + Sync + 'static;
    // *** Important: Output now needs to be 'static to be used with Any ***
    type Output: Serialize + Send + Sync + 'static;

    fn name(&self) -> String;
    fn description(&self) -> String;
}

// --- LLM Facing Representation (unchanged) ---
#[derive(Serialize, Debug, Clone)] // Clone might be useful
pub struct LlmToolInfo {
    name: String,
    description: String,
    parameters: Value,
}

// --- Execution Abstraction (Modified) ---
// *** Add `Any` as a supertrait ***
trait DynExecutableTool: Send + Sync + Any {
    // <- Added Any
    fn name(&self) -> &str;
    // *** Return Box<dyn Any + Send + Sync> instead of Value ***
    fn execute(&self, args: Value) -> Result<Box<dyn Any + Send + Sync>, ToolError>;
    fn llm_info(&self) -> Result<LlmToolInfo, ToolError>;

    // Helper method for downcasting self (optional but can be convenient)
    fn as_any(&self) -> &dyn Any;
}

// --- Concrete Tool Runner (Modified) ---
struct ToolRunner<T: ToolDefinition + Send + Sync + 'static> {
    // T needs 'static
    definition: Arc<T>,
    func: Arc<
        dyn Fn(T::Input) -> Result<T::Output, Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync,
    >,
    // Store name/info for easier access
    llm_info: LlmToolInfo,
}

impl<T: ToolDefinition + Send + Sync + 'static> ToolRunner<T> {
    pub fn new<E>(
        definition: T,
        func: impl Fn(T::Input) -> Result<T::Output, E> + Send + Sync + 'static,
    ) -> Result<Self, ToolError>
    // Return Result to handle schema errors
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let name = definition.name();
        let description = definition.description();
        let schema = schema_for!(T::Input); // Generate schema once
        let parameters = serde_json::to_value(schema)
            .map_err(|e| ToolError::SchemaGenerationError(name.clone(), e))?;

        let llm_info = LlmToolInfo {
            name: name.clone(), // Clone name here
            description,
            parameters,
        };

        Ok(ToolRunner {
            definition: Arc::new(definition),
            func: Arc::new(move |input| {
                func(input).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }),
            llm_info, // Store pre-generated info
        })
    }
}

// --- Implement DynExecutableTool for ToolRunner (Modified) ---
impl<T: ToolDefinition + Send + Sync + 'static> DynExecutableTool for ToolRunner<T> {
    fn name(&self) -> &str {
        &self.llm_info.name // Use stored name
    }

    fn llm_info(&self) -> Result<LlmToolInfo, ToolError> {
        Ok(self.llm_info.clone()) // Return stored info
    }

    fn execute(&self, args: Value) -> Result<Box<dyn Any + Send + Sync>, ToolError> {
        let tool_name = self.name();

        // 1. Parse Input
        let input: T::Input = serde_json::from_value(args)
            .map_err(|e| ToolError::ArgumentParsingError(tool_name.to_string(), e))?;

        // 2. Execute Function
        let output: T::Output =
            (self.func)(input).map_err(|e| ToolError::ExecutionError(tool_name.to_string(), e))?;

        // 3. *** Box the concrete output type into Box<dyn Any> ***
        Ok(Box::new(output))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// --- Tool Registry/Dispatcher (Modified execute_tool return type) ---
pub struct ToolRegistry {
    // Store Box<dyn DynExecutableTool> directly
    tools: HashMap<String, Box<dyn DynExecutableTool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        ToolRegistry {
            tools: HashMap::new(),
        }
    }

    pub fn add_tool<T, E>(
        &mut self,
        definition: T,
        func: impl Fn(T::Input) -> Result<T::Output, E> + Send + Sync + 'static,
    ) -> Result<(), ToolError>
    where
        T: ToolDefinition + Send + Sync + 'static, // T needs 'static
        E: std::error::Error + Send + Sync + 'static,
    {
        // Use ToolRunner::new which returns Result
        let runner = ToolRunner::new(definition, func)?;
        let name = runner.name().to_string(); // Get name from runner
        if self.tools.contains_key(&name) {
            eprintln!(
                "Warning: Tool with name '{}' already exists. Overwriting.",
                name
            );
        }
        // Store the Box<dyn DynExecutableTool>
        self.tools.insert(name, Box::new(runner));
        Ok(())
    }

    pub fn get_llm_tool_infos(&self) -> Result<Vec<LlmToolInfo>, ToolError> {
        self.tools.values().map(|tool| tool.llm_info()).collect()
    }

    // *** Changed return type ***
    pub fn execute_tool(
        &self,
        name: &str,
        args: Value,
    ) -> Result<Box<dyn Any + Send + Sync>, ToolError> {
        match self.tools.get(name) {
            Some(tool) => tool.execute(args), // Directly call execute
            None => Err(ToolError::NotFound(name.to_string())),
        }
    }

    // Optional: Helper for typed execution if you know the type
    pub fn execute_typed<Output: Any + Send + Sync + 'static>(
        &self,
        name: &str,
        args: Value,
    ) -> Result<Output, ToolError> {
        let result_any: Box<dyn Any + Send + Sync> = self.execute_tool(name, args)?;
        // Attempt to downcast the Box<dyn Any> into Box<Output> then unbox
        result_any
            .downcast::<Output>()
            .map(|boxed_output| *boxed_output) // Dereference the Box to get Output
            .map_err(|_e| ToolError::OutputTypeMismatch(name.to_string()))
    }
}

// --- Example Usage (Modified) ---

#[derive(JsonSchema, Deserialize, Debug, Clone)] // Added Clone
struct MyToolArgs {
    val: f64,
}

// Make Output Clone + Debug + PartialEq for easier example checks
#[derive(Serialize, Debug, Clone, PartialEq)]
struct MyToolOutput {
    doubled: f64,
}

#[derive(Error, Debug)]
enum MyToolFuncError {
    #[error("Input value cannot be negative")]
    NegativeInput,
}

fn run_my_tool(args: MyToolArgs) -> Result<MyToolOutput, MyToolFuncError> {
    println!("Executing my_tool with args: {:?}", args);
    if args.val < 0.0 {
        Err(MyToolFuncError::NegativeInput)
    } else {
        Ok(MyToolOutput {
            doubled: args.val * 2.0,
        })
    }
}

struct MyToolDefinition;
impl ToolDefinition for MyToolDefinition {
    type Input = MyToolArgs;
    type Output = MyToolOutput; // Needs to be 'static (structs usually are)

    fn name(&self) -> String {
        "my_tool".to_string()
    }
    fn description(&self) -> String {
        "Doubles the input floating point number.".to_string()
    }
}

fn main() -> Result<(), ToolError> {
    let mut registry = ToolRegistry::new();
    registry.add_tool(MyToolDefinition, run_my_tool)?;

    let llm_infos = registry.get_llm_tool_infos()?;
    println!(
        "Tool Info for LLM: {}",
        serde_json::to_string_pretty(&llm_infos).unwrap()
    );

    let llm_tool_name = "my_tool";
    let llm_args_ok = serde_json::json!({ "val": 21.0 });
    let llm_args_neg = serde_json::json!({ "val": -5.0 });

    println!("\n--- Executing OK (Manual Downcast) ---");
    match registry.execute_tool(llm_tool_name, llm_args_ok.clone()) {
        Ok(result_any) => {
            // We know "my_tool" should return MyToolOutput
            match result_any.downcast::<MyToolOutput>() {
                Ok(boxed_output) => {
                    let output: MyToolOutput = *boxed_output; // Unbox
                    println!("Execution successful (typed): {:?}", output);
                    assert_eq!(output, MyToolOutput { doubled: 42.0 });
                }
                Err(_) => {
                    eprintln!("Execution failed: Output type mismatch!");
                    // This would indicate a programming error if we expected MyToolOutput
                }
            }
        }
        Err(e) => eprintln!("Execution failed: {}", e),
    }

    println!("\n--- Executing OK (Helper Method) ---");
    match registry.execute_typed::<MyToolOutput>(llm_tool_name, llm_args_ok) {
        Ok(output) => {
            println!("Execution successful (typed helper): {:?}", output);
            assert_eq!(output, MyToolOutput { doubled: 42.0 });
        }
        Err(e) => eprintln!("Execution failed: {}", e),
    }

    println!("\n--- Executing Negative Input (Error Handling) ---");
    match registry.execute_tool(llm_tool_name, llm_args_neg) {
        Ok(_) => eprintln!("Execution unexpectedly succeeded?"),
        Err(e) => {
            eprintln!("Execution failed as expected: {}", e);
            // Optionally, you could try downcasting the error in ToolError::ExecutionError
            if let ToolError::ExecutionError(_, boxed_err) = e {
                if let Some(specific_err) = boxed_err.downcast_ref::<MyToolFuncError>() {
                    println!("   (Caught specific error type: {:?})", specific_err);
                    assert!(matches!(specific_err, MyToolFuncError::NegativeInput));
                }
            }
        }
    }

    Ok(())
}
