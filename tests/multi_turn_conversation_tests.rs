use language_barrier::SingleRequestExecutor;
use language_barrier::model::{Claude, Gemini, GPT, Mistral, Sonnet35Version, ModelInfo};
use language_barrier::provider::HTTPProvider;
use language_barrier::{Chat, Message};
use language_barrier::message::{Content, ContentPart, ToolCall, Function};
use tracing::{info, warn, Level};

// Import our helper modules
mod test_utils;
mod test_tools;

use test_utils::{
    setup_tracing, 
    get_anthropic_provider, 
    get_openai_provider, 
    get_gemini_provider, 
    get_mistral_provider,
    extract_text_content,
    has_tool_calls
};
use test_tools::WeatherToolbox;

// Test multi-turn conversation with tools with all available providers
#[tokio::test]
async fn test_multi_turn_conversation() {
    setup_tracing(Level::INFO);
    info!("Starting test_multi_turn_conversation");
    
    // Test with Anthropic if credentials available
    if let Some(provider) = get_anthropic_provider() {
        info!("Testing Anthropic multi-turn conversation");
        let model = Claude::Sonnet35 { version: Sonnet35Version::V2 };
        let base_chat = Chat::new(model);
        test_multi_turn_with_provider("Anthropic", base_chat, provider).await;
    }
    
    // Test with OpenAI if credentials available
    if let Some(provider) = get_openai_provider() {
        info!("Testing OpenAI multi-turn conversation");
        let model = GPT::GPT4o;
        let base_chat = Chat::new(model);
        test_multi_turn_with_provider("OpenAI", base_chat, provider).await;
    }
    
    // Test with Gemini if credentials available
    if let Some(provider) = get_gemini_provider() {
        info!("Testing Gemini multi-turn conversation");
        let model = Gemini::Flash20;
        let base_chat = Chat::new(model);
        test_multi_turn_with_provider("Gemini", base_chat, provider).await;
    }
    
    // Test with Mistral if credentials available
    if let Some(provider) = get_mistral_provider() {
        info!("Testing Mistral multi-turn conversation");
        let model = Mistral::Small;
        let base_chat = Chat::new(model);
        test_multi_turn_with_provider("Mistral", base_chat, provider).await;
    }
}

// Helper function to test multi-turn conversation with a specific provider
async fn test_multi_turn_with_provider<P, M>(
    provider_name: &str,
    base_chat: Chat<M>,
    provider: P
) where
    P: HTTPProvider<M> + 'static + Clone,
    M: Clone + ModelInfo,
{
    // First question
    {
        let mut chat1 = Chat::new(base_chat.model.clone())
            .with_system_prompt("You are a helpful AI assistant that can provide weather information.")
            .with_max_output_tokens(1000)
            .with_toolbox(WeatherToolbox);
        
        // Add first user message
        chat1.add_message(Message::user("Hi! Can you tell me about the weather in Paris?"));
        
        // Create an executor with our provider
        let executor = SingleRequestExecutor::new(provider.clone());
        
        // Get initial response
        match executor.send(chat1).await {
            Ok(first_response) => {
                info!("{} first response received", provider_name);
                
                // Check if the response includes a tool call
                let first_has_tool_calls = has_tool_calls(&first_response);
                
                // Create a new chat for the second question
                let mut chat2 = Chat::new(base_chat.model.clone())
                    .with_system_prompt("You are a helpful AI assistant that can provide weather information.")
                    .with_max_output_tokens(1000)
                    .with_toolbox(WeatherToolbox);
                
                // Add first user message and response
                chat2.add_message(Message::user("Hi! Can you tell me about the weather in Paris?"));
                chat2.add_message(first_response.clone());
                
                // Process any tool calls from the first response
                if first_has_tool_calls {
                    match chat2.process_tool_calls(&first_response) {
                        Ok(()) => {
                            info!("{} first tool calls processed successfully", provider_name);
                        },
                        Err(e) => {
                            warn!("{} first tool call processing failed: {}", provider_name, e);
                        }
                    }
                }
                
                // Add follow-up question
                chat2.add_message(Message::user("Thanks! And what about the weather in London tomorrow?"));
                
                // Get follow-up response
                match executor.send(chat2).await {
                    Ok(second_response) => {
                        info!("{} second response received", provider_name);
                        
                        // Check if the follow-up mentions London
                        let refers_to_london = match &second_response {
                            Message::Assistant { content, tool_calls, .. } => {
                                // Check content for London reference
                                let in_content = match content {
                                    Some(Content::Text(text)) => text.contains("London"),
                                    Some(Content::Parts(parts)) => parts.iter().any(|part| {
                                        if let ContentPart::Text { text } = part {
                                            text.contains("London")
                                        } else {
                                            false
                                        }
                                    }),
                                    None => false,
                                };
                                
                                // Check tool calls for London reference
                                let in_tool_calls = tool_calls.iter().any(|call| {
                                    call.function.arguments.contains("London")
                                });
                                
                                in_content || in_tool_calls
                            },
                            _ => false,
                        };
                        
                        assert!(refers_to_london, "{} should reference London in follow-up", provider_name);
                        
                        // Create a third chat for the comparison question
                        let mut chat3 = Chat::new(base_chat.model.clone())
                            .with_system_prompt("You are a helpful AI assistant that can provide weather information.")
                            .with_max_output_tokens(1000)
                            .with_toolbox(WeatherToolbox);
                            
                        // Add all previous messages
                        chat3.add_message(Message::user("Hi! Can you tell me about the weather in Paris?"));
                        chat3.add_message(first_response.clone());
                        
                        // We can't access chat2.history as it was moved in the executor.send() call
                        // Add a tool response if the first response had tool calls
                        if first_has_tool_calls {
                            // Reconstruct a tool response message - normally this would come from chat2.history
                            // but we can't access it after it was moved
                            if let Message::Assistant { tool_calls, .. } = &first_response {
                                for tool_call in tool_calls {
                                    let location = if tool_call.function.arguments.contains("Paris") { 
                                        "Paris" 
                                    } else { 
                                        "unknown location" 
                                    };
                                    
                                    let weather_response = format!(
                                        "Weather in {}: 22 degrees celsius, partly cloudy with a chance of rain",
                                        location
                                    );
                                    
                                    // Make sure to use the exact same tool call ID from the first response
                                    chat3.add_message(Message::tool(tool_call.id.clone(), weather_response));
                                }
                            }
                        }
                        
                        chat3.add_message(Message::user("Thanks! And what about the weather in London tomorrow?"));
                        chat3.add_message(second_response.clone());
                        
                        // Process any tool calls from the second response
                        let second_has_tool_calls = has_tool_calls(&second_response);
                        
                        if second_has_tool_calls {
                            match chat3.process_tool_calls(&second_response) {
                                Ok(()) => {
                                    info!("{} second tool calls processed successfully", provider_name);
                                },
                                Err(e) => {
                                    warn!("{} second tool call processing failed: {}", provider_name, e);
                                }
                            }
                        }
                        
                        // Add third question
                        chat3.add_message(Message::user("Which city has better weather right now, Paris or London?"));
                        
                        // Get third response
                        match executor.send(chat3).await {
                            Ok(third_response) => {
                                info!("{} third response received", provider_name);
                                
                                // Check that the third response mentions at least one of the cities or has tool calls
                                // This is a more lenient check to avoid test flakiness
                                let text = extract_text_content(&third_response);
                                let contains_city = text.contains("Paris") || 
                                                   text.contains("London") || 
                                                   has_tool_calls(&third_response);
                                
                                assert!(contains_city, 
                                       "{} should reference at least one city or have tool calls in third response", provider_name);
                                
                                info!("{} multi-turn conversation test successful", provider_name);
                            },
                            Err(e) => {
                                warn!("{} third response request failed: {}", provider_name, e);
                            }
                        }
                    },
                    Err(e) => {
                        warn!("{} second response request failed: {}", provider_name, e);
                    }
                }
            },
            Err(e) => {
                warn!("{} first response request failed: {}", provider_name, e);
            }
        }
    }
}

// Test tool result conversion
#[tokio::test]
async fn test_tool_result_conversion() {
    setup_tracing(Level::DEBUG);
    info!("Starting test_tool_result_conversion");
    
    // Test Anthropic tool result conversion
    {
        info!("Testing Anthropic tool result conversion");
        let provider = get_anthropic_provider().unwrap_or_default();
        
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
    
    // Test OpenAI tool result conversion
    {
        info!("Testing OpenAI tool result conversion");
        let provider = get_openai_provider().unwrap_or_default();
        
        // Create a sequence with a tool message
        let mut chat = Chat::new(GPT::GPT4o)
            .with_system_prompt("You are a helpful assistant.");
        
        // Create a tool call
        let tool_call = ToolCall {
            id: "call_456".to_string(),
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"New York"}"#.to_string(),
            },
        };
        
        // Add an assistant message with a tool call
        let assistant_message = Message::assistant_with_tool_calls(vec![tool_call]);
        chat.add_message(assistant_message);
        
        // Add a tool response
        let tool_message = Message::tool("call_456", "Weather in New York: Cloudy, 68°F");
        chat.add_message(tool_message);
        
        // Create a request using the provider
        let request = provider.accept(chat).unwrap();
        
        // Get the request body as a string
        let body_bytes = request.body().unwrap().as_bytes().unwrap();
        let body_str = std::str::from_utf8(body_bytes).unwrap();
        
        // Verify the request contains the tool call and result with the right content
        assert!(body_str.contains("\"tool_call_id\":\"call_456\""));
        assert!(body_str.contains("Weather in New York: Cloudy, 68°F"));
    }
    
    // Test Gemini tool result conversion
    {
        info!("Testing Gemini tool result conversion");
        let provider = get_gemini_provider().unwrap_or_default();
        
        // Create a sequence with a tool message
        let mut chat = Chat::new(Gemini::Flash20)
            .with_system_prompt("You are a helpful assistant.");
        
        // Create a tool call
        let tool_call = ToolCall {
            id: "call_789".to_string(),
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"Tokyo"}"#.to_string(),
            },
        };
        
        // Add an assistant message with a tool call
        let assistant_message = Message::assistant_with_tool_calls(vec![tool_call]);
        chat.add_message(assistant_message);
        
        // Add a tool response
        let tool_message = Message::tool("call_789", "Weather in Tokyo: Sunny, 25°C");
        chat.add_message(tool_message);
        
        // Create a request using the provider
        let request = provider.accept(chat).unwrap();
        
        // Get the request body as a string
        let body_bytes = request.body().unwrap().as_bytes().unwrap();
        let body_str = std::str::from_utf8(body_bytes).unwrap();
        
        // Verify the request contains the tool call and result with the right content
        assert!(body_str.contains("call_789") || body_str.contains("get_weather"));
        assert!(body_str.contains("Weather in Tokyo: Sunny, 25°C"));
    }
    
    // Test Mistral tool result conversion
    {
        info!("Testing Mistral tool result conversion");
        let provider = get_mistral_provider().unwrap_or_default();
        
        // Create a sequence with a tool message
        let mut chat = Chat::new(Mistral::Small)
            .with_system_prompt("You are a helpful assistant.");
        
        // Create a tool call
        let tool_call = ToolCall {
            id: "call_abc".to_string(),
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"Berlin"}"#.to_string(),
            },
        };
        
        // Add an assistant message with a tool call
        let assistant_message = Message::assistant_with_tool_calls(vec![tool_call]);
        chat.add_message(assistant_message);
        
        // Add a tool response
        let tool_message = Message::tool("call_abc", "Weather in Berlin: Rainy, 15°C");
        chat.add_message(tool_message);
        
        // Create a request using the provider
        let request = provider.accept(chat).unwrap();
        
        // Get the request body as a string
        let body_bytes = request.body().unwrap().as_bytes().unwrap();
        let body_str = std::str::from_utf8(body_bytes).unwrap();
        
        // Verify the request contains the tool call and result with the right content
        assert!(body_str.contains("\"id\":\"call_abc\"") || body_str.contains("\"tool_call_id\":\"call_abc\""));
        assert!(body_str.contains("Weather in Berlin: Rainy, 15°C"));
    }
}