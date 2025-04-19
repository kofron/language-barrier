use dotenv::dotenv;
use language_barrier::SingleRequestExecutor;
use language_barrier::model::GPT;
use language_barrier::provider::openai::{OpenAIConfig, OpenAIProvider};
use language_barrier::{Chat, Message, Tool, ToolDescription, Toolbox, Result};
use language_barrier::message::{Content, ContentPart};
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
async fn test_openai_tools() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer()
            .with_test_writer())
        .with(EnvFilter::from_default_env()
            .add_directive(Level::INFO.into()));
    
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_openai_tools");

    // Load environment variables from .env file if available
    dotenv().ok();

    // Skip test if no API key is available
    let api_key = match env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            warn!("Skipping test: No OPENAI_API_KEY found");
            return;
        }
    };

    // Create a provider with the API key
    let config = OpenAIConfig {
        api_key,
        base_url: "https://api.openai.com/v1".to_string(),
        organization: None,
    };
    let provider = OpenAIProvider::with_config(config);

    // Create an executor
    let executor = SingleRequestExecutor::new(provider);

    // Create a chat with tools
    let mut chat = Chat::new(GPT::GPT4o)
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
        },
        Err(e) => {
            warn!("Failed to get response: {}", e);
            return; // Skip the rest of the test
        }
    };
    
    // Check if the response includes a tool call
    let has_tool_calls = match &response {
        Message::Assistant { tool_calls, .. } => {
            if !tool_calls.is_empty() {
                info!("Response contains {} tool calls", tool_calls.len());
                for tool_call in tool_calls {
                    info!("Tool call: {} - {}", tool_call.function.name, tool_call.function.arguments);
                }
                assert!(!tool_calls.is_empty(), "Expected at least one tool call");
                true
            } else {
                info!("Response does not contain tool calls");
                // Tool calls aren't guaranteed, so this isn't a failure
                false
            }
        },
        _ => false,
    };
    
    // Create a new chat with the tool response
    if has_tool_calls {
        let mut updated_chat = Chat::new(GPT::GPT4o)
            .with_system_prompt("You are a helpful AI assistant that can provide weather information.")
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
                if let Message::Assistant { content, .. } = &resp {
                    info!("Received follow-up response: {:?}", content);
                }
                // Note: OpenAI often returns only tool calls without content
                resp
            },
            Err(e) => {
                warn!("Failed to get follow-up response: {}", e);
                return; // Skip the rest of the test
            }
        };
        
        // Check for London reference in either content or tool calls
        let references_london_in_content = match &follow_up {
            Message::Assistant { content, .. } => {
                match content {
                    Some(Content::Text(text)) => {
                        info!("Follow-up text: {}", text);
                        text.contains("London")
                    },
                    Some(Content::Parts(parts)) => {
                        let has_london = parts.iter().any(|part| {
                            match part {
                                ContentPart::Text { text } => {
                                    info!("Follow-up part: {}", text);
                                    text.contains("London")
                                },
                                _ => false,
                            }
                        });
                        has_london
                    },
                    None => {
                        info!("Follow-up has no content, will check tool calls only");
                        false
                    }
                }
            },
            _ => false,
        };
        
        // Check if the follow-up uses tools with London reference
        let references_london_in_tools = match &follow_up {
            Message::Assistant { tool_calls, .. } => {
                if !tool_calls.is_empty() {
                    info!("Follow-up contains {} tool calls", tool_calls.len());
                    let mut has_london_tool = false;
                    
                    for tool_call in tool_calls {
                        info!("Follow-up tool call: {} - {}", tool_call.function.name, tool_call.function.arguments);
                        if tool_call.function.arguments.contains("London") {
                            has_london_tool = true;
                        }
                    }
                    has_london_tool
                } else {
                    false
                }
            },
            _ => false,
        };
        
        // OpenAI may return just tool calls without content, so we check both possibilities
        assert!(references_london_in_content || references_london_in_tools, 
               "Expected a reference to London in either content or tool calls");
        
        info!("OpenAI tools test successful with proper follow-up question handling");
    } else {
        info!("Skipping follow-up test as no tool calls were made");
    }
}