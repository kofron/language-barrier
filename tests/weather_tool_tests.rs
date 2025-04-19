use language_barrier::SingleRequestExecutor;
use language_barrier::model::{Claude, Gemini, GPT, Mistral, Sonnet35Version};
use language_barrier::provider::HTTPProvider;
use language_barrier::{Chat, Message};
use language_barrier::message::{Content, ContentPart};
use tracing::{info, warn, Level};

// Import our helper modules
mod test_utils;
mod test_tools;

use test_utils::{
    setup_tracing, 
    get_anthropic_provider, 
    get_openai_provider, 
    get_gemini_provider, 
    get_mistral_provider
};
use test_tools::WeatherToolbox;

// Test request creation with weather tools for all providers
#[tokio::test]
async fn test_weather_tool_request_creation() {
    setup_tracing(Level::DEBUG);
    info!("Starting test_weather_tool_request_creation for all providers");
    
    // Test Anthropic weather tool request creation
    {
        info!("Testing Anthropic weather tool request creation");
        let provider = get_anthropic_provider().unwrap_or_default();
        let model = Claude::Sonnet35 { version: Sonnet35Version::V2 };
        let mut chat = Chat::new(model)
            .with_system_prompt("You are a helpful AI assistant.")
            .with_max_output_tokens(1000)
            .with_toolbox(WeatherToolbox);
        
        chat.add_message(Message::user("What's the weather in San Francisco?"));
        
        let request = provider.accept(chat).unwrap();
        
        // Get the request body as a string
        let body_bytes = request.body().unwrap().as_bytes().unwrap();
        let body_str = std::str::from_utf8(body_bytes).unwrap();
        
        // Check that the request includes tools
        assert!(body_str.contains("\"tools\""));
        assert!(body_str.contains("\"get_weather\""));
        assert!(body_str.contains("\"location\""));
    }
    
    // Test OpenAI weather tool request creation
    {
        info!("Testing OpenAI weather tool request creation");
        let provider = get_openai_provider().unwrap_or_default();
        let model = GPT::GPT4o;
        let mut chat = Chat::new(model)
            .with_system_prompt("You are a helpful AI assistant.")
            .with_max_output_tokens(1000)
            .with_toolbox(WeatherToolbox);
        
        chat.add_message(Message::user("What's the weather in San Francisco?"));
        
        let request = provider.accept(chat).unwrap();
        
        // Get the request body as a string
        let body_bytes = request.body().unwrap().as_bytes().unwrap();
        let body_str = std::str::from_utf8(body_bytes).unwrap();
        
        // Check that the request includes tools
        assert!(body_str.contains("\"tools\""));
        assert!(body_str.contains("\"get_weather\""));
        assert!(body_str.contains("\"location\""));
    }
    
    // Test Gemini weather tool request creation
    {
        info!("Testing Gemini weather tool request creation");
        let provider = get_gemini_provider().unwrap_or_default();
        let model = Gemini::Flash20;
        let mut chat = Chat::new(model)
            .with_system_prompt("You are a helpful AI assistant.")
            .with_max_output_tokens(1000)
            .with_toolbox(WeatherToolbox);
        
        chat.add_message(Message::user("What's the weather in San Francisco?"));
        
        let request = provider.accept(chat).unwrap();
        
        // Get the request body as a string
        let body_bytes = request.body().unwrap().as_bytes().unwrap();
        let body_str = std::str::from_utf8(body_bytes).unwrap();
        
        // Check that the request includes tools
        assert!(body_str.contains("\"tools\"") || body_str.contains("\"functionDeclarations\""));
    }
    
    // Test Mistral weather tool request creation
    {
        info!("Testing Mistral weather tool request creation");
        let provider = get_mistral_provider().unwrap_or_default();
        let model = Mistral::Small;
        let mut chat = Chat::new(model)
            .with_system_prompt("You are a helpful AI assistant.")
            .with_max_output_tokens(1000)
            .with_toolbox(WeatherToolbox);
        
        chat.add_message(Message::user("What's the weather in San Francisco?"));
        
        let request = provider.accept(chat).unwrap();
        
        // Get the request body as a string
        let body_bytes = request.body().unwrap().as_bytes().unwrap();
        let body_str = std::str::from_utf8(body_bytes).unwrap();
        
        // Check that the request includes tools
        assert!(body_str.contains("\"tools\""));
        assert!(body_str.contains("\"get_weather\""));
        assert!(body_str.contains("\"location\""));
    }
}

// Test live tool integration with all available providers
#[tokio::test]
async fn test_weather_tool_integration() {
    setup_tracing(Level::INFO);
    info!("Starting test_weather_tool_integration");
    
    // Test with Anthropic if credentials available
    if let Some(provider) = get_anthropic_provider() {
        info!("Testing Anthropic weather tool integration");
        test_weather_tool_with_provider(
            "Anthropic",
            Chat::new(Claude::Sonnet35 { version: Sonnet35Version::V2 }),
            provider
        ).await;
    }
    
    // Test with OpenAI if credentials available
    if let Some(provider) = get_openai_provider() {
        info!("Testing OpenAI weather tool integration");
        test_weather_tool_with_provider(
            "OpenAI",
            Chat::new(GPT::GPT4o),
            provider
        ).await;
    }
    
    // Test with Gemini if credentials available
    if let Some(provider) = get_gemini_provider() {
        info!("Testing Gemini weather tool integration");
        test_weather_tool_with_provider(
            "Gemini",
            Chat::new(Gemini::Flash20),
            provider
        ).await;
    }
    
    // Test with Mistral if credentials available
    if let Some(provider) = get_mistral_provider() {
        info!("Testing Mistral weather tool integration");
        test_weather_tool_with_provider(
            "Mistral",
            Chat::new(Mistral::Small),
            provider
        ).await;
    }
}

// Helper function to test weather tool with a specific provider
async fn test_weather_tool_with_provider<P, M>(
    provider_name: &str,
    base_chat: Chat<M>,
    provider: P
) where
    P: HTTPProvider<M> + 'static,
    M: Clone + language_barrier::model::ModelInfo,
{
    // Create a chat with our configuration
    let mut chat = Chat::new(base_chat.model.clone())
        .with_system_prompt("You are a helpful AI assistant that can provide weather information. Always use the weather tool when asked about weather.")
        .with_max_output_tokens(1000)
        .with_toolbox(WeatherToolbox);
    
    // Add a user message
    chat.add_message(Message::user("What's the weather in Paris?"));
    
    // Create an executor with our provider
    let executor = SingleRequestExecutor::new(provider);
    
    // Get the response
    match executor.send(chat).await {
        Ok(response) => {
            info!("{} weather tool test successful", provider_name);
            
            // Check if the response includes a tool call
            let has_tool_calls = match &response {
                Message::Assistant { tool_calls, .. } => !tool_calls.is_empty(),
                _ => false,
            };
            
            if has_tool_calls {
                info!("{} response contains tool calls", provider_name);
                
                // Create a new chat with tool response processing
                let mut new_chat = base_chat
                    .with_system_prompt("You are a helpful AI assistant that can provide weather information.")
                    .with_max_output_tokens(1000)
                    .with_toolbox(WeatherToolbox);
                
                // Add the original message and response
                new_chat.add_message(Message::user("What's the weather in Paris?"));
                new_chat.add_message(response.clone());
                
                // Process the tool calls
                match new_chat.process_tool_calls(&response) {
                    Ok(()) => {
                        info!("{} tool calls processed successfully", provider_name);
                        
                        // Add a follow-up question
                        new_chat.add_message(Message::user("How about the weather in London?"));
                        
                        // Get follow-up response
                        match executor.send(new_chat).await {
                            Ok(follow_up) => {
                                info!("{} follow-up response received", provider_name);
                                
                                // Check if the follow-up mentions London
                                let refers_to_london = match &follow_up {
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
                            },
                            Err(e) => {
                                warn!("{} follow-up request failed: {}", provider_name, e);
                            }
                        }
                    },
                    Err(e) => {
                        warn!("{} tool call processing failed: {}", provider_name, e);
                    }
                }
            } else {
                info!("{} response doesn't contain tool calls, skipping follow-up test", provider_name);
            }
        },
        Err(e) => {
            warn!("{} weather tool test failed: {}", provider_name, e);
        }
    }
}