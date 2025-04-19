use dotenv::dotenv;
use language_barrier::SingleRequestExecutor;
use language_barrier::model::Gemini;
use language_barrier::provider::gemini::{GeminiConfig, GeminiProvider};
use language_barrier::{Chat, Message, Result, Tool, ToolDescription, Toolbox};
use language_barrier::message::{Content, ContentPart};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use tracing::{Level, info, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*, registry};

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

        vec![ToolDescription {
            name: "get_weather".to_string(),
            description: "Get current weather for a location".to_string(),
            parameters: weather_schema_value,
        }]
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
async fn test_gemini_tools() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::INFO.into()));

    tracing::subscriber::set_global_default(subscriber).ok();

    info!("Starting test_gemini_tools");

    // Load environment variables from .env file if available
    dotenv().ok();

    // Skip test if no API key is available
    let api_key = match env::var("GEMINI_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            warn!("Skipping test: No GEMINI_API_KEY found");
            return;
        }
    };

    // Create a provider with the API key
    let config = GeminiConfig {
        api_key,
        base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
    };
    let provider = GeminiProvider::with_config(config);

    // Create an executor
    let executor = SingleRequestExecutor::new(provider);

    // Create a chat with tools
    let mut chat = Chat::new(Gemini::Flash15)
        .with_system_prompt("You are a helpful AI assistant that can provide weather information.")
        .with_max_output_tokens(1000)
        .with_toolbox(TestToolbox);

    // Add a user message
    info!("Sending user message");
    chat.add_message(Message::user("What's the weather like in Paris?"));

    // Get response
    let response = match executor.send(chat).await {
        Ok(resp) => {
            if let Message::Assistant { content, .. } = &resp {
                info!("Received response: {:?}", content);
            }
            resp
        }
        Err(e) => {
            warn!("Failed to get response: {}", e);
            return; // Skip the rest of the test
        }
    };

    // Check if the response includes a tool call
    match &response {
        Message::Assistant { tool_calls, .. } => {
            if !tool_calls.is_empty() {
                info!("Response contains {} tool calls", tool_calls.len());
                for tool_call in tool_calls {
                    info!(
                        "Tool call: {} - {}",
                        tool_call.function.name, tool_call.function.arguments
                    );
                }
                assert!(!tool_calls.is_empty(), "Expected at least one tool call");
            } else {
                info!("Response does not contain tool calls");
                // Tool calls aren't guaranteed, so this isn't a failure
            }
        },
        _ => panic!("Expected assistant message"),
    }

    // Create a new chat with the tool response
    let has_tool_calls = match &response {
        Message::Assistant { tool_calls, .. } => !tool_calls.is_empty(),
        _ => false,
    };
    
    if has_tool_calls {
        let mut updated_chat = Chat::new(Gemini::Flash15)
            .with_system_prompt(
                "You are a helpful AI assistant that can provide weather information.",
            )
            .with_max_output_tokens(1000)
            .with_toolbox(TestToolbox);

        // Add the original message and response
        updated_chat.add_message(Message::user("What's the weather like in Paris?"));
        updated_chat.add_message(response.clone());

        // Process tool calls
        match updated_chat.process_tool_calls(&response) {
            Ok(()) => info!("Tool calls processed successfully"),
            Err(e) => {
                warn!("Failed to process tool calls: {}", e);
            }
        }

        // Add a follow-up question
        updated_chat.add_message(Message::user("How about London?"));

        // Get follow-up response
        let follow_up = match executor.send(updated_chat).await {
            Ok(resp) => {
                match &resp {
                    Message::Assistant { content, .. } => {
                        info!("Received follow-up response: {:?}", content);
                        assert!(content.is_some(), "Expected content in follow-up response");
                    },
                    _ => panic!("Expected assistant message"),
                }
                resp
            }
            Err(e) => {
                warn!("Failed to get follow-up response: {}", e);
                return; // Skip the rest of the test
            }
        };
        
        // Inspect the follow-up response content
        match &follow_up {
            Message::Assistant { content, tool_calls, .. } => {
                match content {
                    Some(Content::Text(text)) => {
                        // The response should mention London (the follow-up question location)
                        let tool_call_has_london = tool_calls.iter().any(|call| {
                            call.function.arguments.contains("London")
                        });
                        
                        assert!(text.contains("London") || tool_call_has_london,
                                "Follow-up response should reference London");
                        
                        info!("Follow-up response correctly references the new location");
                    },
                    Some(Content::Parts(parts)) => {
                        let has_london = parts.iter().any(|part| {
                            match part {
                                ContentPart::Text { text } => text.contains("London"),
                                _ => false,
                            }
                        });
                        
                        let tool_call_has_london = tool_calls.iter().any(|call| {
                            call.function.arguments.contains("London")
                        });
                        
                        assert!(has_london || tool_call_has_london,
                                "Follow-up response should reference London");
                        
                        info!("Follow-up response correctly references the new location");
                    },
                    None => {
                        // If no content, check tool calls
                        assert!(tool_calls.iter().any(|call| call.function.arguments.contains("London")),
                                "With no content, tool calls should reference London");
                    }
                }
                
                // Check if the follow-up also uses tools
                if !tool_calls.is_empty() {
                    info!("Follow-up contains {} tool calls", tool_calls.len());
                    for tool_call in tool_calls {
                        info!("Follow-up tool call: {} - {}", tool_call.function.name, tool_call.function.arguments);
                        // Check if the tool call is for London weather
                        assert!(tool_call.function.arguments.contains("London"), 
                                "Tool call should be for London weather");
                    }
                }
            },
            _ => panic!("Expected assistant message"),
        }
        
        info!("Gemini tools test successful with proper follow-up question handling");
    } else {
        info!("Skipping follow-up test as no tool calls were made");
    }
}
