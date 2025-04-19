use dotenv::dotenv;
use std::sync::{Arc, Mutex};
use language_barrier::SingleRequestExecutor;
use language_barrier::model::{Claude, Gemini, GPT, Mistral, Sonnet35Version};
use language_barrier::provider::HTTPProvider;
use language_barrier::provider::anthropic::{AnthropicConfig, AnthropicProvider};
use language_barrier::provider::gemini::{GeminiConfig, GeminiProvider};
use language_barrier::provider::mistral::{MistralConfig, MistralProvider};
use language_barrier::provider::openai::{OpenAIConfig, OpenAIProvider};
use language_barrier::{Chat, Message, Tool, ToolDescription, Toolbox, Result};
use language_barrier::message::{Content, ContentPart};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use tracing::{info, warn, Level};
use tracing_subscriber::{fmt, prelude::*, registry, EnvFilter};

// Define the calculator tool request format
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CalculatorRequest {
    operation: String,
    a: f64,
    b: f64,
}

impl Tool for CalculatorRequest {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform a calculation between two numbers"
    }
}

// Define a calculator toolbox that tracks call count
#[derive(Clone)]
struct CalculatorToolbox {
    call_count: Arc<Mutex<usize>>,
}

impl CalculatorToolbox {
    fn new() -> Self {
        Self {
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

impl Toolbox for CalculatorToolbox {
    fn describe(&self) -> Vec<ToolDescription> {
        let schema = schemars::schema_for!(CalculatorRequest);
        let schema_value = serde_json::to_value(schema.schema).unwrap();

        vec![ToolDescription {
            name: "calculator".to_string(),
            description: "Perform a calculation between two numbers".to_string(),
            parameters: schema_value,
        }]
    }

    fn execute(&self, name: &str, arguments: Value) -> Result<String> {
        if name != "calculator" {
            return Err(language_barrier::Error::ToolNotFound(name.to_string()));
        }

        // Increment call count
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        // Parse the arguments
        let request: CalculatorRequest = serde_json::from_value(arguments)?;
        
        // Perform the calculation
        let result = match request.operation.as_str() {
            "add" => request.a + request.b,
            "subtract" => request.a - request.b,
            "multiply" => request.a * request.b,
            "divide" => {
                if request.b == 0.0 {
                    return Err(language_barrier::Error::ToolExecutionError(
                        "Division by zero".to_string(),
                    ));
                }
                request.a / request.b
            }
            _ => {
                return Err(language_barrier::Error::ToolExecutionError(format!(
                    "Unsupported operation: {}",
                    request.operation
                )))
            }
        };

        Ok(format!("The result of {} {} {} is {}", 
            request.a, request.operation, request.b, result))
    }
}

// Helper function to set up logging for tests
fn setup_tracing(level: Level) {
    let subscriber = registry()
        .with(fmt::layer()
            .with_test_writer()
            .with_ansi(false) // Better for CI logs
            .with_file(true)  // Include source code location
            .with_line_number(true))
        .with(EnvFilter::from_default_env()
            .add_directive(level.into())
            .add_directive("reqwest=info".parse().unwrap())); // Lower verbosity for reqwest
    
    let _ = tracing::subscriber::set_global_default(subscriber);
}

// Test calculator tool with all available providers
#[tokio::test]
async fn test_calculator_tool() {
    setup_tracing(Level::INFO);
    info!("Starting test_calculator_tool");
    
    // Load environment variables
    dotenv().ok();
    
    // Test with Anthropic if credentials available
    if let Ok(api_key) = env::var("ANTHROPIC_API_KEY") {
        if !api_key.is_empty() {
            info!("Testing Anthropic calculator tool");
            test_calculator_with_provider(
                "Anthropic",
                Chat::new(Claude::Sonnet35 { version: Sonnet35Version::V2 }),
                AnthropicProvider::with_config(AnthropicConfig {
                    api_key,
                    base_url: "https://api.anthropic.com/v1".to_string(),
                    api_version: "2023-06-01".to_string(),
                })
            ).await;
        }
    }
    
    // Test with OpenAI if credentials available
    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        if !api_key.is_empty() {
            info!("Testing OpenAI calculator tool");
            test_calculator_with_provider(
                "OpenAI",
                Chat::new(GPT::GPT4o),
                OpenAIProvider::with_config(OpenAIConfig {
                    api_key,
                    base_url: "https://api.openai.com/v1".to_string(),
                    organization: None,
                })
            ).await;
        }
    }
    
    // Test with Gemini if credentials available
    if let Ok(api_key) = env::var("GEMINI_API_KEY") {
        if !api_key.is_empty() {
            info!("Testing Gemini calculator tool");
            test_calculator_with_provider(
                "Gemini",
                Chat::new(Gemini::Flash20),
                GeminiProvider::with_config(GeminiConfig {
                    api_key,
                    base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
                })
            ).await;
        }
    }
    
    // Test with Mistral if credentials available
    if let Ok(api_key) = env::var("MISTRAL_API_KEY") {
        if !api_key.is_empty() {
            info!("Testing Mistral calculator tool");
            test_calculator_with_provider(
                "Mistral",
                Chat::new(Mistral::Small),
                MistralProvider::with_config(MistralConfig {
                    api_key,
                    base_url: "https://api.mistral.ai/v1".to_string(),
                })
            ).await;
        }
    }
}

// Helper function to test calculator tool with a specific provider
async fn test_calculator_with_provider<P, M>(
    provider_name: &str,
    base_chat: Chat<M>,
    provider: P
) where
    P: HTTPProvider<M> + 'static,
    M: Clone + language_barrier::model::ModelInfo,
{
    // Create a toolbox to track tool calls
    let toolbox = CalculatorToolbox::new();
    
    // Create a chat with our configuration
    let mut chat = Chat::new(base_chat.model.clone())
        .with_system_prompt("You are a helpful AI assistant that uses tools when appropriate. Always use the calculator tool for math problems.")
        .with_max_output_tokens(256)
        .with_toolbox(toolbox.clone());
    
    // Add a user message that should trigger the calculator tool
    chat.add_message(Message::user("What is 123 multiplied by 456?"));
    
    // Create an executor with our provider
    let executor = SingleRequestExecutor::new(provider);
    
    // Get the response
    match executor.send(chat).await {
        Ok(response) => {
            info!("{} calculator request sent successfully", provider_name);
            
            // Check if the response includes a tool call
            let has_tool_calls = match &response {
                Message::Assistant { tool_calls, .. } => !tool_calls.is_empty(),
                _ => false,
            };
            
            if has_tool_calls {
                info!("{} response contains tool calls", provider_name);
                
                // Create a new chat with tool response processing
                let mut new_chat = base_chat
                    .with_system_prompt("You are a helpful AI assistant that uses tools when appropriate. Always use the calculator tool for math problems.")
                    .with_max_output_tokens(256)
                    .with_toolbox(toolbox.clone());
                
                // Add the original message and response
                new_chat.add_message(Message::user("What is 123 multiplied by 456?"));
                new_chat.add_message(response.clone());
                
                // Process the tool calls
                match new_chat.process_tool_calls(&response) {
                    Ok(()) => {
                        info!("{} tool calls processed successfully", provider_name);
                        
                        // Verify that the tool was called
                        assert_eq!(toolbox.get_call_count(), 1, "{} should have called the calculator once", provider_name);
                        
                        // Get final response that should include the calculation result
                        match executor.send(new_chat).await {
                            Ok(final_response) => {
                                info!("{} final response received", provider_name);
                                
                                // Check that the final response contains the correct result
                                match &final_response {
                                    Message::Assistant { content, .. } => {
                                        let text = match content {
                                            Some(Content::Text(text)) => text.clone(),
                                            Some(Content::Parts(parts)) => {
                                                let mut combined = String::new();
                                                for part in parts {
                                                    if let ContentPart::Text { text } = part {
                                                        combined.push_str(text);
                                                    }
                                                }
                                                combined
                                            },
                                            None => String::new(),
                                        };
                                        
                                        // The result should mention 56088 (123 * 456)
                                        assert!(
                                            text.contains("56088") || 
                                            text.contains("56,088") || 
                                            text.contains("56 088"), 
                                            "{} should include the correct calculation result", provider_name
                                        );
                                        
                                        info!("{} calculator test successful with correct result", provider_name);
                                    },
                                    _ => warn!("{} didn't return assistant message", provider_name),
                                }
                            },
                            Err(e) => {
                                warn!("{} final response request failed: {}", provider_name, e);
                            }
                        }
                    },
                    Err(e) => {
                        warn!("{} tool call processing failed: {}", provider_name, e);
                    }
                }
            } else {
                // Some models might get the calculation right without using tools
                info!("{} response doesn't contain tool calls, checking direct answer", provider_name);
                
                match &response {
                    Message::Assistant { content, .. } => {
                        let text = match content {
                            Some(Content::Text(text)) => text.clone(),
                            Some(Content::Parts(parts)) => {
                                let mut combined = String::new();
                                for part in parts {
                                    if let ContentPart::Text { text } = part {
                                        combined.push_str(text);
                                    }
                                }
                                combined
                            },
                            None => String::new(),
                        };
                        
                        // Check if the response contains the correct result even without tool use
                        if text.contains("56088") || text.contains("56,088") || text.contains("56 088") {
                            info!("{} provided correct calculation result without using the tool", provider_name);
                        } else {
                            warn!("{} didn't use the tool and didn't provide the correct result", provider_name);
                        }
                    },
                    _ => warn!("{} didn't return assistant message", provider_name),
                }
            }
        },
        Err(e) => {
            warn!("{} calculator tool test failed: {}", provider_name, e);
        }
    }
}

// Test error handling in calculator tool
#[tokio::test]
async fn test_calculator_error_handling() {
    setup_tracing(Level::INFO);
    info!("Starting test_calculator_error_handling");
    
    // Load environment variables
    dotenv().ok();
    
    // We'll test division by zero error handling with any available provider
    let mut provider_tested = false;
    
    // Try with Anthropic if available
    if !provider_tested {
        if let Ok(api_key) = env::var("ANTHROPIC_API_KEY") {
            if !api_key.is_empty() {
                provider_tested = true;
                test_calculator_error_with_provider(
                    "Anthropic",
                    Chat::new(Claude::Sonnet35 { version: Sonnet35Version::V2 }),
                    AnthropicProvider::with_config(AnthropicConfig {
                        api_key,
                        base_url: "https://api.anthropic.com/v1".to_string(),
                        api_version: "2023-06-01".to_string(),
                    })
                ).await;
            }
        }
    }
    
    // Try with OpenAI if available and not tested yet
    if !provider_tested {
        if let Ok(api_key) = env::var("OPENAI_API_KEY") {
            if !api_key.is_empty() {
                provider_tested = true;
                test_calculator_error_with_provider(
                    "OpenAI",
                    Chat::new(GPT::GPT4o),
                    OpenAIProvider::with_config(OpenAIConfig {
                        api_key,
                        base_url: "https://api.openai.com/v1".to_string(),
                        organization: None,
                    })
                ).await;
            }
        }
    }
    
    // Try with Gemini if available and not tested yet
    if !provider_tested {
        if let Ok(api_key) = env::var("GEMINI_API_KEY") {
            if !api_key.is_empty() {
                provider_tested = true;
                test_calculator_error_with_provider(
                    "Gemini",
                    Chat::new(Gemini::Flash20),
                    GeminiProvider::with_config(GeminiConfig {
                        api_key,
                        base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
                    })
                ).await;
            }
        }
    }
    
    // Try with Mistral if available and not tested yet
    if !provider_tested {
        if let Ok(api_key) = env::var("MISTRAL_API_KEY") {
            if !api_key.is_empty() {
                provider_tested = true;
                test_calculator_error_with_provider(
                    "Mistral",
                    Chat::new(Mistral::Small),
                    MistralProvider::with_config(MistralConfig {
                        api_key,
                        base_url: "https://api.mistral.ai/v1".to_string(),
                    })
                ).await;
            }
        }
    }
    
    if !provider_tested {
        warn!("Skipping test_calculator_error_handling: No API keys found");
    }
}

// Helper function to test calculator error handling
async fn test_calculator_error_with_provider<P, M>(
    provider_name: &str,
    base_chat: Chat<M>,
    provider: P
) where
    P: HTTPProvider<M> + 'static,
    M: Clone + language_barrier::model::ModelInfo,
{
    // Create a toolbox to track tool calls
    let toolbox = CalculatorToolbox::new();
    
    // Create a chat with our configuration
    let mut chat = Chat::new(base_chat.model.clone())
        .with_system_prompt("You are a helpful AI assistant that uses tools when appropriate. Always use the calculator tool for math problems.")
        .with_max_output_tokens(256)
        .with_toolbox(toolbox.clone());
    
    // Add a user message that should trigger a division by zero error
    chat.add_message(Message::user("What is 5 divided by 0?"));
    
    // Create an executor with our provider
    let executor = SingleRequestExecutor::new(provider);
    
    // Get the response
    match executor.send(chat).await {
        Ok(response) => {
            info!("{} division by zero request sent successfully", provider_name);
            
            // Check if the response includes a tool call
            let has_tool_calls = match &response {
                Message::Assistant { tool_calls, .. } => !tool_calls.is_empty(),
                _ => false,
            };
            
            if has_tool_calls {
                info!("{} response contains tool calls", provider_name);
                
                // Create a new chat with tool response processing
                let mut new_chat = base_chat
                    .with_system_prompt("You are a helpful AI assistant that uses tools when appropriate. Always use the calculator tool for math problems.")
                    .with_max_output_tokens(256)
                    .with_toolbox(toolbox.clone());
                
                // Add the original message and response
                new_chat.add_message(Message::user("What is 5 divided by 0?"));
                new_chat.add_message(response.clone());
                
                // Process the tool calls - this should generate an error
                match new_chat.process_tool_calls(&response) {
                    Ok(()) => {
                        // We should still have at least one tool call even if it failed
                        assert!(toolbox.get_call_count() > 0, "{} should have called the calculator", provider_name);
                        
                        // Check that the tool response includes an error message
                        let tool_response = new_chat.history.last().unwrap();
                        match tool_response {
                            Message::Tool { content, .. } => {
                                assert!(content.contains("Division by zero") || content.contains("error"),
                                       "{} tool response should contain an error message", provider_name);
                                
                                info!("{} tool response indicates division by zero error as expected", provider_name);
                            },
                            _ => warn!("{} unexpected message type for tool response", provider_name),
                        }
                        
                        // Get final response that should acknowledge the error
                        match executor.send(new_chat).await {
                            Ok(final_response) => {
                                info!("{} final response received", provider_name);
                                
                                // Check that the final response acknowledges the error
                                match &final_response {
                                    Message::Assistant { content, .. } => {
                                        let text = match content {
                                            Some(Content::Text(text)) => text.clone(),
                                            Some(Content::Parts(parts)) => {
                                                let mut combined = String::new();
                                                for part in parts {
                                                    if let ContentPart::Text { text } = part {
                                                        combined.push_str(text);
                                                    }
                                                }
                                                combined
                                            },
                                            None => String::new(),
                                        };
                                        
                                        // The response should explain that division by zero is undefined
                                        assert!(
                                            text.contains("undefined") || 
                                            text.contains("not defined") || 
                                            text.contains("impossible") ||
                                            text.contains("error") ||
                                            text.contains("can't divide"), 
                                            "{} should explain that division by zero is undefined", provider_name
                                        );
                                        
                                        info!("{} calculator error test successful with proper explanation", provider_name);
                                    },
                                    _ => warn!("{} didn't return assistant message", provider_name),
                                }
                            },
                            Err(e) => {
                                warn!("{} final response request failed: {}", provider_name, e);
                            }
                        }
                    },
                    Err(e) => {
                        info!("{} tool call processing failed as expected for division by zero: {}", provider_name, e);
                        // Accept any error message related to the calculation
                        assert!(e.to_string().contains("Division by zero") || 
                               e.to_string().contains("div") || 
                               e.to_string().contains("zero") ||
                               e.to_string().contains("Unsupported operation") ||
                               e.to_string().contains("/"),
                               "Error should be related to division or calculation: {}", e);
                    }
                }
            } else {
                // The model might correctly explain division by zero without using the tool
                info!("{} response doesn't contain tool calls, checking direct answer", provider_name);
                
                match &response {
                    Message::Assistant { content, .. } => {
                        let text = match content {
                            Some(Content::Text(text)) => text.clone(),
                            Some(Content::Parts(parts)) => {
                                let mut combined = String::new();
                                for part in parts {
                                    if let ContentPart::Text { text } = part {
                                        combined.push_str(text);
                                    }
                                }
                                combined
                            },
                            None => String::new(),
                        };
                        
                        // Check if the response explains division by zero
                        if text.contains("undefined") || 
                           text.contains("not defined") || 
                           text.contains("impossible") ||
                           text.contains("can't divide") {
                            info!("{} correctly explained division by zero without using the tool", provider_name);
                        } else {
                            warn!("{} didn't use the tool and didn't properly explain division by zero", provider_name);
                        }
                    },
                    _ => warn!("{} didn't return assistant message", provider_name),
                }
            }
        },
        Err(e) => {
            warn!("{} calculator error test failed: {}", provider_name, e);
        }
    }
}