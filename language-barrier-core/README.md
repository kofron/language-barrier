# Language Barrier

A Rust library that provides abstractions for Large Language Models (LLMs).

## Overview

Language Barrier simplifies working with multiple LLM providers by offering a unified, type-safe API. It allows you to easily switch between different providers like OpenAI, Anthropic, Google, Mistral, and more without changing your application code.

## Features

- Provider-agnostic API for working with LLMs
- Support for chat completions, text generation, and multimodal content
- Type-safe message structures based on the OpenAI standard format
- Built-in tools system with type-safe execution
- History management with automatic token tracking and conversation compaction
- Detailed error handling for common LLM API issues
- Async-first design for efficient network operations
- Comprehensive model and provider management

## Installation

Add the library to your Cargo.toml:

```toml
[dependencies]
language-barrier = "0.1.0"
```

## Quick Start

### Basic Chat Example

This example shows how to create a simple chat with the Anthropic Claude model:

```rust
use language_barrier::{
    Chat, Claude, Message, SingleRequestExecutor,
    provider::anthropic::AnthropicProvider,
};

#[tokio::main]
async fn main() -> language_barrier::Result<()> {
    // Create a provider (automatically uses ANTHROPIC_API_KEY environment variable)
    let provider = AnthropicProvider::new();
    
    // Create an executor with the provider
    let executor = SingleRequestExecutor::new(provider);
    
    // Create a new chat with Claude
    let mut chat = Chat::new(Claude::Haiku3)
        .with_system_prompt("You are a helpful AI assistant that provides concise answers.")
        .with_max_output_tokens(1024);
    
    // Add a user message
    chat.add_message(Message::user("What is the capital of France?"));
    
    // Send the chat and get a response
    let response = executor.send(chat).await?;
    
    // Print the response
    println!("Response: {:?}", response.content);
    
    Ok(())
}
```

### Using Tools

Language Barrier supports tool/function calling with a type-safe interface:

```rust
use language_barrier::{
    Chat, Claude, Message, Tool, ToolDescription, Toolbox, Result,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Define a weather tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherRequest {
    location: String,
    units: Option<String>,
}

impl Tool for WeatherRequest {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get current weather for a location"
    }
}

// Create a toolbox implementation
struct MyToolbox;

impl Toolbox for MyToolbox {
    fn describe(&self) -> Vec<ToolDescription> {
        // Create schema for WeatherRequest
        let weather_schema = schemars::schema_for!(WeatherRequest);
        let weather_schema_value = serde_json::to_value(weather_schema.schema).unwrap();

        vec![
            ToolDescription {
                name: "get_weather".to_string(),
                description: "Get current weather for a location".to_string(),
                parameters: weather_schema_value,
            },
        ]
    }

    fn execute(&self, name: &str, arguments: Value) -> Result<String> {
        match name {
            "get_weather" => {
                let request: WeatherRequest = serde_json::from_value(arguments)?;
                let units = request.units.unwrap_or_else(|| "celsius".to_string());
                Ok(format!(
                    "Weather in {}: 22 degrees {}, partly cloudy",
                    request.location, units
                ))
            }
            _ => Err(language_barrier::Error::ToolNotFound(name.to_string())),
        }
    }
}

fn main() -> Result<()> {
    // Create a chat with tools
    let mut chat = Chat::new(Claude::Haiku3)
        .with_system_prompt("You are a helpful assistant with access to tools.")
        .with_toolbox(MyToolbox);
    
    // In a real application, you would send this chat to the provider
    // and the model would decide when to use tools
    
    println!("Tool descriptions for LLM:");
    for desc in chat.tool_descriptions() {
        println!("- {}: {}", desc.name, desc.description);
    }
    
    Ok(())
}
```

### Multimodal Content

Language Barrier supports multimodal content like images:

```rust
use language_barrier::{
    Chat, Message, ContentPart, ImageUrl,
    Claude, provider::anthropic::AnthropicProvider, 
    SingleRequestExecutor,
};

#[tokio::main]
async fn main() -> language_barrier::Result<()> {
    // Create a provider that supports vision models
    let provider = AnthropicProvider::new();
    let executor = SingleRequestExecutor::new(provider);
    
    // Create a chat with a vision-capable model
    let mut chat = Chat::new(Claude::Sonnet35 { version: language_barrier::model::Sonnet35Version::V2 })
        .with_system_prompt("You are a helpful assistant that can analyze images.");
    
    // Create a message with text and an image
    let parts = vec![
        ContentPart::text("What can you see in this image?"),
        ContentPart::image_url(ImageUrl::new("https://example.com/image.jpg")),
    ];
    
    // Add the multimodal message
    let message = Message::user("").with_content_parts(parts);
    chat.add_message(message);
    
    // Send the chat to get a response (this would require an actual API call)
    // let response = executor.send(chat).await?;
    // println!("Response: {:?}", response.content);
    
    Ok(())
}
```

## Working with Different Providers

Language Barrier makes it easy to switch between providers:

```rust
// Using Anthropic's Claude
let provider = AnthropicProvider::new();
let chat = Chat::new(Claude::Opus3);

// Using Google's Gemini
let provider = GeminiProvider::new();
let chat = Chat::new(Gemini::Flash20);

// Using OpenAI's GPT
let provider = OpenAIProvider::new();
let chat = Chat::new(GPT::GPT4o);

// Using Mistral AI
let provider = MistralProvider::new();
let chat = Chat::new(Mistral::Small);
```

## Token Management and History Compaction

Language Barrier automatically manages conversation history and token usage:

```rust
use language_barrier::{Chat, Claude, DropOldestCompactor};

// Create a chat with custom compaction strategy
let mut chat = Chat::new(Claude::Haiku3)
    .with_compactor(DropOldestCompactor::new());

// The library automatically tracks token usage and compacts history
// when it exceeds the model's context window
```

## Documentation

For more details, see the [API documentation](https://docs.rs/language-barrier).

## Design

For information about the design and architecture of the library, see the [DESIGN.md](DESIGN.md) document.

## License

This project is licensed under the MIT License - see the LICENSE file for details.