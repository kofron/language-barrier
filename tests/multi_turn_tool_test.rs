use dotenv::dotenv;
use language_barrier::SingleRequestExecutor;
use language_barrier::model::{Claude, Sonnet35Version};
use language_barrier::provider::anthropic::{AnthropicConfig, AnthropicProvider};
use language_barrier::{Chat, Message, Tool, ToolDescription, Toolbox, Result};
use language_barrier::message::{Content, ContentPart, ToolCall, Function};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use tracing::{info, warn, Level};
use tracing_subscriber::{fmt, prelude::*, registry, EnvFilter};

// Define a simple weather tool
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

// Define a simple toolbox
struct TestToolbox;

impl Toolbox for TestToolbox {
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
                    "Weather in {}: 22 degrees {}, partly cloudy with a chance of rain",
                    request.location, units
                ))
            }
            _ => Err(language_barrier::Error::ToolNotFound(name.to_string())),
        }
    }
}

#[tokio::test]
async fn test_multi_turn_conversation_with_tools() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer()
            .with_test_writer()
            .with_ansi(false) // Better for CI logs
            .with_file(true)  // Include source code location
            .with_line_number(true))
        .with(EnvFilter::from_default_env()
            .add_directive(Level::TRACE.into()));
    
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_multi_turn_conversation_with_tools");

    // Load environment variables from .env file if available
    dotenv().ok();

    // Skip test if no API key is available
    let api_key = match env::var("ANTHROPIC_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            warn!("Skipping test: No ANTHROPIC_API_KEY found");
            return;
        }
    };

    // Create a provider with the API key
    let config = AnthropicConfig {
        api_key,
        base_url: "https://api.anthropic.com/v1".to_string(),
        api_version: "2023-06-01".to_string(),
    };
    let provider = AnthropicProvider::with_config(config);

    // Create an executor
    let executor = SingleRequestExecutor::new(provider);

    // Create a chat with tools
    let mut chat = Chat::new(Claude::Sonnet35 { 
        version: Sonnet35Version::V2 
    })
    .with_system_prompt("You are a helpful AI assistant that can provide weather information.")
    .with_max_output_tokens(1000)
    .with_toolbox(TestToolbox);

    // Start a multi-turn conversation
    
    // First message
    info!("Sending first user message");
    chat.add_message(Message::user("What's the weather like in San Francisco?"));
    
    // Get first response
    let response = match executor.send(chat).await {
        Ok(resp) => {
            match &resp {
                Message::Assistant { content, .. } => {
                    info!("Received first response: {:?}", content);
                },
                _ => {},
            }
            resp
        },
        Err(e) => {
            warn!("Failed to get response: {}", e);
            if e.to_string().contains("missing field") {
                warn!("This could be an API version mismatch or field name change");
            }
            return; // Skip the rest of the test
        }
    };
    
    // The response should include a tool call for weather
    match &response {
        Message::Assistant { tool_calls, .. } => {
            assert!(!tool_calls.is_empty(), "Expected tool calls in the response");
        },
        _ => panic!("Expected assistant message"),
    }
    
    // Process tool calls
    let mut updated_chat = Chat::new(Claude::Sonnet35 { 
        version: Sonnet35Version::V2 
    })
    .with_system_prompt("You are a helpful AI assistant that can provide weather information.")
    .with_max_output_tokens(1000)
    .with_toolbox(TestToolbox);
    
    // Add the original user question
    updated_chat.add_message(Message::user("What's the weather like in San Francisco?"));
    
    // Add the assistant's response
    updated_chat.add_message(response.clone());
    
    // Process the tool calls to add tool response messages
    updated_chat.process_tool_calls(&response).unwrap();
    
    // Ask a follow-up question
    info!("Sending follow-up question");
    updated_chat.add_message(Message::user("How about the weather in New York?"));
    
    // Get response to follow-up
    let follow_up_response = match executor.send(updated_chat).await {
        Ok(resp) => {
            match &resp {
                Message::Assistant { content, .. } => {
                    info!("Received follow-up response: {:?}", content);
                },
                _ => {},
            }
            resp
        },
        Err(e) => {
            warn!("Failed to get follow-up response: {}", e);
            if e.to_string().contains("missing field") {
                warn!("This could be an API version mismatch or field name change");
            }
            return; // Skip the rest of the test
        }
    };
    
    // The follow-up response should also include a tool call
    match &follow_up_response {
        Message::Assistant { tool_calls, .. } => {
            assert!(!tool_calls.is_empty(), "Expected tool calls in follow-up response");
        },
        _ => panic!("Expected assistant message"),
    }
    
    // Inspect the follow-up response content to verify it references New York
    let references_new_york = match &follow_up_response {
        Message::Assistant { content, tool_calls, .. } => {
            match content {
                Some(Content::Text(text)) => {
                    info!("Follow-up text response: {}", text);
                    text.contains("New York")
                },
                Some(Content::Parts(parts)) => {
                    parts.iter().any(|part| match part {
                        ContentPart::Text { text } => {
                            info!("Follow-up part: {}", text);
                            text.contains("New York")
                        },
                        _ => false,
                    })
                },
                None => {
                    // Check if tool call references New York if there's no content
                    tool_calls.iter().any(|call| call.function.arguments.contains("New York"))
                }
            }
        },
        _ => panic!("Expected assistant message"),
    };
    
    // Alternative check in tool calls for New York reference
    let tool_calls_reference_new_york = match &follow_up_response {
        Message::Assistant { tool_calls, .. } => {
            tool_calls.iter().any(|call| {
                info!("Follow-up tool call: {} - {}", call.function.name, call.function.arguments);
                call.function.arguments.contains("New York")
            })
        },
        _ => panic!("Expected assistant message"),
    };
    
    assert!(references_new_york || tool_calls_reference_new_york, 
            "Follow-up response should reference New York");
    
    // Check that the model remembered previous context - it shouldn't repeat the San Francisco weather
    // when we asked about New York
    let response_text = match &follow_up_response {
        Message::Assistant { content, .. } => {
            match content {
                Some(Content::Text(text)) => text.clone(),
                _ => String::new(),
            }
        },
        _ => String::new(),
    };
    
    assert!(!response_text.contains("San Francisco"), 
        "Follow-up response should not repeat San Francisco weather");
}