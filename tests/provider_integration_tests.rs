use dotenv::dotenv;
use language_barrier::SingleRequestExecutor;
use language_barrier::model::{Claude, Gemini, GPT, Sonnet35Version};
use language_barrier::provider::HTTPProvider;
use language_barrier::provider::anthropic::{AnthropicConfig, AnthropicProvider};
use language_barrier::provider::gemini::{GeminiConfig, GeminiProvider};
use language_barrier::provider::openai::{OpenAIConfig, OpenAIProvider};
use language_barrier::{Chat, Message, Tool, ToolDescription, Toolbox};
use language_barrier::message::{Content, ContentPart, ToolCall, Function};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use tracing::{debug, info, warn, error, Level};
use tracing_subscriber::{fmt, prelude::*, registry, EnvFilter};

#[tokio::test]
async fn test_anthropic_request_creation() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_anthropic_request_creation");
    
    // Create an Anthropic provider with default configuration
    let provider = AnthropicProvider::new();

    // Create a chat with a simple message
    let model = Claude::Sonnet35 {
        version: Sonnet35Version::V2,
    };
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant.")
        .with_max_output_tokens(1000);

    // Add a user message
    chat.add_message(Message::user("What is the capital of France?"));

    // Create a request using the provider
    let request = provider.accept(chat).unwrap();

    // Verify request properties
    assert_eq!(request.method(), "POST");
    assert_eq!(
        request.url().as_str(),
        "https://api.anthropic.com/v1/messages"
    );
    assert!(request.headers().contains_key("x-api-key"));
    assert!(request.headers().contains_key("anthropic-version"));
    assert_eq!(
        request.headers().get("Content-Type").unwrap(),
        "application/json"
    );
}

#[tokio::test]
async fn test_anthropic_integration_with_executor() {
    // Initialize tracing for this test with detailed output
    let subscriber = registry()
        .with(fmt::layer()
            .with_test_writer()
            .with_ansi(false) // Better for CI logs
            .with_file(true)  // Include source code location
            .with_line_number(true))
        .with(EnvFilter::from_default_env()
            .add_directive(Level::TRACE.into())  // Maximum verbosity
            .add_directive("reqwest=info".parse().unwrap())); // Lower verbosity for reqwest
    
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_anthropic_integration_with_executor");

    // Load environment variables from .env file if available
    info!("Loading environment variables from .env");
    dotenv().ok();

    // Skip test if no API key is available
    let api_key = match env::var("ANTHROPIC_API_KEY") {
        Ok(key) if !key.is_empty() => {
            info!("API key found in environment");
            debug!("API key length: {}", key.len());
            key
        },
        _ => {
            warn!("Skipping test: No ANTHROPIC_API_KEY found");
            return;
        }
    };

    // Create a provider with the API key
    info!("Creating Anthropic provider with custom config");
    let config = AnthropicConfig {
        api_key,
        base_url: "https://api.anthropic.com/v1".to_string(),
        api_version: "2023-06-01".to_string(),
    };
    let provider = AnthropicProvider::with_config(config);

    // Create an executor with our provider
    info!("Creating SingleRequestExecutor");
    let executor = SingleRequestExecutor::new(provider);

    // Create a chat with a simple prompt
    info!("Creating Chat with Sonnet 3.5");
    let model = Claude::Sonnet35 {
        version: Sonnet35Version::V2,
    };
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant that provides very short answers.")
        .with_max_output_tokens(100);

    // Add a user message
    info!("Adding user message to chat");
    chat.add_message(Message::user("What is the capital of France?"));
    debug!("Message count: {}", chat.history.len());

    // Send the chat and get the response
    info!("Sending chat request to Anthropic API");
    let response = match executor.send(chat).await {
        Ok(resp) => {
            info!("Successfully received response from Anthropic API");
            resp
        },
        Err(e) => {
            error!("Failed to get response from Anthropic API: {}", e);
            panic!("Test failed: {}", e);
        }
    };

    // Verify we got a response from the assistant
    info!("Verifying response");
    debug!("Response role: {:?}", response.role_str());
    assert!(matches!(response, Message::Assistant { .. }));
    
    // Get content from the response
    let content = match &response {
        Message::Assistant { content, .. } => content,
        _ => panic!("Expected assistant message"),
    };
    assert!(content.is_some());

    // Print the response for manual verification
    info!("Test completed successfully");
    debug!("Response content: {:?}", content);
    println!("Response: {:?}", content);

    // Verify token usage metadata is present
    match &response {
        Message::Assistant { metadata, .. } => {
            debug!("Token usage - input: {:?}, output: {:?}", 
                metadata.get("input_tokens"),
                metadata.get("output_tokens"));
            assert!(metadata.contains_key("input_tokens"));
            assert!(metadata.contains_key("output_tokens"));
        },
        _ => panic!("Expected assistant message"),
    };
}

// Define a simple test tool for weather
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct TestWeatherTool {
    location: String,
}

impl Tool for TestWeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get weather information for a location"
    }
}

// Simple toolbox implementation for testing
struct TestToolbox;

impl Toolbox for TestToolbox {
    fn describe(&self) -> Vec<ToolDescription> {
        // Create schema for TestWeatherTool
        let weather_schema = schemars::schema_for!(TestWeatherTool);
        let weather_schema_value = serde_json::to_value(weather_schema.schema).unwrap();

        vec![
            ToolDescription {
                name: "get_weather".to_string(),
                description: "Get weather information for a location".to_string(),
                parameters: weather_schema_value,
            },
        ]
    }

    fn execute(&self, name: &str, arguments: Value) -> language_barrier::Result<String> {
        match name {
            "get_weather" => {
                let request: TestWeatherTool = serde_json::from_value(arguments)?;
                Ok(format!("Weather in {}: Sunny, 72°F", request.location))
            }
            _ => Err(language_barrier::Error::ToolNotFound(name.to_string())),
        }
    }
}

#[tokio::test]
async fn test_anthropic_tool_response_parsing() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_anthropic_tool_response_parsing");
    
    // Create an Anthropic provider with default configuration
    let provider = AnthropicProvider::new();

    // Create a mock Anthropic response JSON with a tool call
    let response_json = r#"{
        "id": "msg_123abc",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "text",
                "text": "I'll check the weather for you."
            },
            {
                "type": "tool_use",
                "id": "tool_call_123",
                "name": "get_weather",
                "input": {
                    "location": "San Francisco"
                }
            }
        ],
        "usage": {
            "input_tokens": 25,
            "output_tokens": 42
        }
    }"#;

    // Parse the response using the provider
    let message = provider.parse(response_json.to_string()).unwrap();
    
    // Verify it parsed correctly
    assert!(matches!(message, Message::Assistant { .. }));
    
    // Check for text content and tool calls
    match &message {
        Message::Assistant { content, tool_calls, metadata } => {
            // Check content
            match content {
                Some(Content::Text(text)) => {
                    assert_eq!(text, "I'll check the weather for you.");
                },
                Some(Content::Parts(parts)) => {
                    assert_eq!(parts.len(), 1);
                    match &parts[0] {
                        ContentPart::Text { text } => {
                            assert_eq!(text, "I'll check the weather for you.");
                        },
                        _ => panic!("Expected text content part"),
                    }
                },
                None => panic!("Expected content to be present"),
            }
            
            // Verify tool calls are present
            assert!(!tool_calls.is_empty());
            assert_eq!(tool_calls.len(), 1);
            
            // Check first tool call
            let tool_call = &tool_calls[0];
            assert_eq!(tool_call.id, "tool_call_123");
            assert_eq!(tool_call.function.name, "get_weather");
            
            // Parse arguments to verify
            let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments).unwrap();
            assert_eq!(args["location"], "San Francisco");
            
            // Verify token usage metadata is present
            assert_eq!(metadata["input_tokens"], 25);
            assert_eq!(metadata["output_tokens"], 42);
        },
        _ => panic!("Expected assistant message"),
    }
}

#[tokio::test]
async fn test_anthropic_tool_result_conversion() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_anthropic_tool_result_conversion");
    
    // Create a provider to use
    let provider = AnthropicProvider::new();
    
    // Create a sequence with a tool message
    let mut chat = Chat::new(Claude::Haiku3)
        .with_system_prompt("You are a helpful assistant.");
    
    // Create a tool call
    let tool_call = ToolCall {
        id: "call_123".to_string(),
        tool_type: "function".to_string(),
        function: Function {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"San Francisco"}"#.to_string(),
        },
    };
    
    // Add an assistant message with a tool call
    let assistant_message = Message::assistant_with_tool_calls(vec![tool_call]);
    chat.add_message(assistant_message);
    
    // Add a tool response
    let tool_message = Message::tool("call_123", "Weather in San Francisco: Sunny, 72°F");
    chat.add_message(tool_message);
    
    // Create a request using the provider
    let request = provider.accept(chat).unwrap();
    
    // Get the request body as a string
    let body_bytes = request.body().unwrap().as_bytes().unwrap();
    let body_str = std::str::from_utf8(body_bytes).unwrap();
    
    // Verify the request contains tool_result with the right content
    assert!(body_str.contains("\"tool_result\""));
    
    // Anthropic API uses tool_use_id instead of tool_call_id
    assert!(body_str.contains("\"tool_use_id\":\"call_123\""));
    
    assert!(body_str.contains("Weather in San Francisco: Sunny, 72°F"));
}

#[tokio::test]
async fn test_anthropic_tools_request_creation() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_anthropic_tools_request_creation");
    
    // Create an Anthropic provider with default configuration
    let provider = AnthropicProvider::new();

    // Create a chat with a simple message and tools
    let model = Claude::Sonnet35 {
        version: Sonnet35Version::V2,
    };
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant.")
        .with_max_output_tokens(1000)
        .with_toolbox(TestToolbox);

    // Add a user message
    chat.add_message(Message::user("What's the weather in San Francisco?"));

    // Create a request using the provider
    let request = provider.accept(chat).unwrap();

    // Get the request body as a string
    let body_bytes = request.body().unwrap().as_bytes().unwrap();
    let body_str = std::str::from_utf8(body_bytes).unwrap();
    
    // Check that the request includes tools
    assert!(body_str.contains("\"tools\""));
    assert!(body_str.contains("\"get_weather\""));
    assert!(body_str.contains("\"location\""));
    
    // Verify request properties
    assert_eq!(request.method(), "POST");
    assert_eq!(
        request.url().as_str(),
        "https://api.anthropic.com/v1/messages"
    );
    assert!(request.headers().contains_key("x-api-key"));
    assert!(request.headers().contains_key("anthropic-version"));
    assert_eq!(
        request.headers().get("Content-Type").unwrap(),
        "application/json"
    );
}

#[tokio::test]
async fn test_chat_process_tool_calls() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_chat_process_tool_calls");
    
    // Create a chat with toolbox
    let mut chat = Chat::new(Claude::Haiku3)
        .with_system_prompt("You are a helpful assistant.")
        .with_toolbox(TestToolbox);
    
    // Add a user message
    chat.add_message(Message::user("What's the weather in San Francisco?"));
    
    // Create a tool call
    let tool_call = ToolCall {
        id: "call_123".to_string(),
        tool_type: "function".to_string(),
        function: Function {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"San Francisco"}"#.to_string(),
        },
    };
    
    // Create an assistant message with a tool call
    let assistant_message = Message::assistant_with_tool_calls(vec![tool_call]);
    
    // Add the assistant message to chat
    chat.add_message(assistant_message.clone());
    
    // Process the tool calls
    chat.process_tool_calls(&assistant_message).unwrap();
    
    // Verify that a tool message was added
    assert_eq!(chat.history.len(), 3); // User, Assistant, Tool
    
    // Check the tool message
    let tool_message = &chat.history[2];
    assert!(matches!(tool_message, Message::Tool { .. }));
    
    // Extract and verify the tool message details
    match tool_message {
        Message::Tool { tool_call_id, content, .. } => {
            assert_eq!(tool_call_id, "call_123");
            assert!(content.contains("Weather in San Francisco"));
            assert!(content.contains("Sunny"));
        }
        _ => panic!("Expected tool message"),
    }
}

#[tokio::test]
async fn test_gemini_request_creation() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_gemini_request_creation");
    
    // Create a Gemini provider with default configuration
    let provider = GeminiProvider::new();

    // Create a chat with a simple message
    let model = Gemini::Flash20;
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant.")
        .with_max_output_tokens(1000);

    // Add a user message
    chat.add_message(Message::user("What is the capital of France?"));

    // Create a request using the provider
    let request = provider.accept(chat).unwrap();

    // Verify request properties
    assert_eq!(request.method(), "POST");
    assert!(request.url().as_str().contains("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"));
    assert!(request.url().as_str().contains("key="));
    assert_eq!(
        request.headers().get("Content-Type").unwrap(),
        "application/json"
    );
}

#[tokio::test]
async fn test_gemini_integration_with_executor() {
    // Initialize tracing for this test with detailed output
    let subscriber = registry()
        .with(fmt::layer()
            .with_test_writer()
            .with_ansi(false) // Better for CI logs
            .with_file(true)  // Include source code location
            .with_line_number(true))
        .with(EnvFilter::from_default_env()
            .add_directive(Level::TRACE.into())  // Maximum verbosity
            .add_directive("reqwest=info".parse().unwrap())); // Lower verbosity for reqwest
    
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_gemini_integration_with_executor");

    // Load environment variables from .env file if available
    info!("Loading environment variables from .env");
    dotenv().ok();

    // Skip test if no API key is available
    let api_key = match env::var("GEMINI_API_KEY") {
        Ok(key) if !key.is_empty() => {
            info!("API key found in environment");
            debug!("API key length: {}", key.len());
            key
        },
        _ => {
            warn!("Skipping test: No GEMINI_API_KEY found");
            return;
        }
    };

    // Create a provider with the API key
    info!("Creating Gemini provider with custom config");
    let config = GeminiConfig {
        api_key,
        base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
    };
    let provider = GeminiProvider::with_config(config);

    // Create an executor with our provider
    info!("Creating SingleRequestExecutor");
    let executor = SingleRequestExecutor::new(provider);

    // Create a chat with a simple prompt
    info!("Creating Chat with Gemini Flash 2.0");
    let model = Gemini::Flash20;
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant that provides very short answers.")
        .with_max_output_tokens(100);

    // Add a user message
    info!("Adding user message to chat");
    chat.add_message(Message::user("What is the capital of France?"));
    debug!("Message count: {}", chat.history.len());

    // Send the chat and get the response
    info!("Sending chat request to Gemini API");
    let response = match executor.send(chat).await {
        Ok(resp) => {
            info!("Successfully received response from Gemini API");
            resp
        },
        Err(e) => {
            error!("Failed to get response from Gemini API: {}", e);
            panic!("Test failed: {}", e);
        }
    };

    // Verify we got a response from the assistant
    info!("Verifying response");
    debug!("Response role: {:?}", response.role_str());
    assert!(matches!(response, Message::Assistant { .. }));
    
    // Get content from the response
    let content = match &response {
        Message::Assistant { content, .. } => content,
        _ => panic!("Expected assistant message"),
    };
    assert!(content.is_some());

    // Print the response for manual verification
    info!("Test completed successfully");
    debug!("Response content: {:?}", content);
    println!("Response: {:?}", content);

    // Verify token usage metadata is present (field names might be different from Anthropic)
    match &response {
        Message::Assistant { metadata, .. } => {
            debug!("Token usage metadata: {:?}", metadata);
            assert!(metadata.contains_key("prompt_tokens") || 
                   metadata.contains_key("total_tokens"));
        },
        _ => panic!("Expected assistant message"),
    };
}

#[tokio::test]
async fn test_openai_request_creation() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_openai_request_creation");
    
    // Create an OpenAI provider with default configuration
    let provider = OpenAIProvider::new();

    // Create a chat with a simple message
    let model = GPT::GPT4o;
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant.")
        .with_max_output_tokens(1000);

    // Add a user message
    chat.add_message(Message::user("What is the capital of France?"));

    // Create a request using the provider
    let request = provider.accept(chat).unwrap();

    // Verify request properties
    assert_eq!(request.method(), "POST");
    assert_eq!(
        request.url().as_str(),
        "https://api.openai.com/v1/chat/completions"
    );
    assert!(request.headers().contains_key("Authorization"));
    assert_eq!(
        request.headers().get("Content-Type").unwrap(),
        "application/json"
    );
}

#[tokio::test]
async fn test_openai_integration_with_executor() {
    // Initialize tracing for this test with detailed output
    let subscriber = registry()
        .with(fmt::layer()
            .with_test_writer()
            .with_ansi(false) // Better for CI logs
            .with_file(true)  // Include source code location
            .with_line_number(true))
        .with(EnvFilter::from_default_env()
            .add_directive(Level::TRACE.into())  // Maximum verbosity
            .add_directive("reqwest=info".parse().unwrap())); // Lower verbosity for reqwest
    
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_openai_integration_with_executor");

    // Load environment variables from .env file if available
    info!("Loading environment variables from .env");
    dotenv().ok();

    // Skip test if no API key is available
    let api_key = match env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() => {
            info!("API key found in environment");
            debug!("API key length: {}", key.len());
            key
        },
        _ => {
            warn!("Skipping test: No OPENAI_API_KEY found");
            return;
        }
    };

    // Create a provider with the API key
    info!("Creating OpenAI provider with custom config");
    let config = OpenAIConfig {
        api_key,
        base_url: "https://api.openai.com/v1".to_string(),
        organization: None,
    };
    let provider = OpenAIProvider::with_config(config);

    // Create an executor with our provider
    info!("Creating SingleRequestExecutor");
    let executor = SingleRequestExecutor::new(provider);

    // Create a chat with a simple prompt
    info!("Creating Chat with GPT-4o");
    let model = GPT::GPT4o;
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant that provides very short answers.")
        .with_max_output_tokens(100);

    // Add a user message
    info!("Adding user message to chat");
    chat.add_message(Message::user("What is the capital of France?"));
    debug!("Message count: {}", chat.history.len());

    // Send the chat and get the response
    info!("Sending chat request to OpenAI API");
    let response = match executor.send(chat).await {
        Ok(resp) => {
            info!("Successfully received response from OpenAI API");
            resp
        },
        Err(e) => {
            error!("Failed to get response from OpenAI API: {}", e);
            panic!("Test failed: {}", e);
        }
    };

    // Verify we got a response from the assistant
    info!("Verifying response");
    debug!("Response role: {:?}", response.role_str());
    assert!(matches!(response, Message::Assistant { .. }));
    
    // Get content from the response
    let content = match &response {
        Message::Assistant { content, .. } => content,
        _ => panic!("Expected assistant message"),
    };
    assert!(content.is_some());

    // Print the response for manual verification
    info!("Test completed successfully");
    debug!("Response content: {:?}", content);
    println!("Response: {:?}", content);

    // Verify token usage metadata is present
    match &response {
        Message::Assistant { metadata, .. } => {
            debug!("Token usage metadata: {:?}", metadata);
            assert!(metadata.contains_key("prompt_tokens") || 
                   metadata.contains_key("total_tokens"));
        },
        _ => panic!("Expected assistant message"),
    };
}
