# Language Barrier Runtime

NB: This is an EXPERIMENTAL crate.  All of the code is 100% subject to change and may or may not actually work.  You have been warned.

A flexible, composable runtime for language model operations based on free monads and Tower middleware.

## Overview

Language Barrier Runtime is an extension to the Language Barrier Core library that provides a more flexible and composable way to build LLM-powered applications. It uses free monads to represent operations as data and Tower middleware for execution.

Key features:

- **Free Monad Operations**: Define LLM workflows as data that can be analyzed, transformed, and executed
- **Tower Middleware**: Compose operations through a stack of middleware components
- **Provider Agnostic**: Works with any provider supported by Language Barrier Core
- **Type-Safe Tool System**: Strong typing for tools using the new ToolRegistry system
- **Extensible**: Easy to add new operations and middleware
- **Tracing**: Built-in tracing for observability

## Architecture

The runtime is built on two core concepts:
1. **Free Monads** - Representing LLM operations as pure data
2. **Tower Middleware** - Providing a composable execution pipeline

### Free Monad Operations

Operations are represented as data structures that can be composed and interpreted:

```rust
pub enum LlmOp<Next> {
    // Send a message to the LLM with optional tools
    Chat { /* ... */ },
    // Execute a specific tool call
    ExecuteTool { /* ... */ },
    // Terminal operation
    Done { /* ... */ },
}
```

### Tower Middleware

The middleware stack allows for specialized handlers:

1. **ChatMiddleware** - For basic chat sessions
2. **BreakOnToolMiddleware** - For stopping when a specific tool is called
3. **ToolExecutorMiddleware** - For executing tools using the new ToolRegistry system
4. **AutoToolExecutorMiddleware** - For automatically executing tools and continuing conversations

## Usage Examples

### Basic Chat

```rust
// Create middleware stack for simple chat
let service = ServiceBuilder::new()
    .layer_fn(|inner| ChatMiddleware::new(inner, chat))
    .service(FinalInterpreter::new());

// Create program
let program = chat(messages, None);

// Execute
let result = service.call(program).await??;
```

### Using the New ToolRegistry System

```rust
// Create tools
struct WeatherTool;

#[derive(Deserialize, JsonSchema)]
struct WeatherInput {
    location: String,
    unit: String,
}

#[derive(Serialize)]
struct WeatherOutput {
    temperature: i32,
    unit: String,
    conditions: String,
}

impl ToolDefinition for WeatherTool {
    type Input = WeatherInput;
    type Output = WeatherOutput;

    fn name(&self) -> String {
        "get_weather".to_string()
    }

    fn description(&self) -> String {
        "Get current weather for a location".to_string()
    }
}

// Register tool in registry
let mut registry = ToolRegistry::new();
registry.register_with_executor(WeatherTool, |tool, value| {
    // Parse input
    let input: WeatherInput = serde_json::from_value(value)?;

    // Execute tool
    let output = WeatherOutput {
        temperature: 22,
        unit: input.unit,
        conditions: "sunny".to_string(),
    };

    // Serialize output
    let json = serde_json::to_string(&output)?;
    Ok(json)
});

// Create service with tool executor
let service = ServiceBuilder::new()
    .layer_fn(|inner| ToolExecutorMiddleware::new(inner, registry.clone()))
    .layer_fn(|inner| ChatMiddleware::new(inner, chat))
    .service(FinalInterpreter::new());

// Create program with tool descriptions
let tool_descriptions = registry.get_tool_descriptions();
let program = chat(messages, Some(tool_descriptions));

// Execute
let result = service.call(program).await??;
```

### Auto Tool Execution with ToolRegistry

```rust
// Create middleware stack that automatically executes tools
let service = ServiceBuilder::new()
    .layer_fn(|inner| AutoToolExecutorMiddleware::new(inner, registry.clone()))
    .layer_fn(|inner| ChatMiddleware::new(inner, chat))
    .service(FinalInterpreter::new());

// Get the tools from the registry
let tool_descriptions = registry.get_tool_descriptions();
let program = chat(messages, Some(tool_descriptions));

// Execute
let result = service.call(program).await??;
```

### Break on Tool Call with ToolRegistry

```rust
// Create middleware stack that breaks on a specific tool
let service = ServiceBuilder::new()
    .layer_fn(|inner| BreakOnToolMiddleware::new(inner, "get_weather".to_string()))
    .layer_fn(|inner| ToolExecutorMiddleware::new(inner, registry.clone()))
    .layer_fn(|inner| ChatMiddleware::new(inner, chat))
    .service(FinalInterpreter::new());
```

## Running Examples

The runtime comes with a simple interactive chat application that demonstrates using the Tower middleware architecture with Language Barrier Core:

```bash
# First, set your API key in the environment
export ANTHROPIC_API_KEY=your_api_key_here

# Run the chat application
cargo run --bin language-barrier-runtime

# Type messages and press Enter to send
# Type 'exit' to quit
```

The chat application demonstrates:
1. Setting up a Tower middleware stack
2. Using the Chat and ops operations to manage conversation state
3. Sending and receiving messages
4. Maintaining conversation history

This provides a simple but complete example of building a real-world application with the Language Barrier Runtime.

The runtime can be extended to support additional examples like:

```bash
# Basic chat without tools
cargo run --bin language-barrier-runtime -- basic

# Chat with automatic tool execution using the old toolbox
cargo run --bin language-barrier-runtime -- auto-tool

# Chat that breaks on specific tool calls using the old toolbox
cargo run --bin language-barrier-runtime -- tool-break

# Chat with automatic tool execution using the new ToolRegistry
cargo run --bin language-barrier-runtime -- registry-tool

# Chat that breaks on specific tool calls using the new ToolRegistry
cargo run --bin language-barrier-runtime -- registry-break
```

## Benefits

1. **Separation of Concerns** - Define what to do separately from how to do it
2. **Composability** - Mix and match middleware as needed
3. **Type Safety** - Use strongly typed tools with associated Input/Output types
4. **Testing** - Operations are just data, making them easy to test
5. **Extensibility** - Add new operations or middleware without changing existing code

## License

Same as Language Barrier Core.
