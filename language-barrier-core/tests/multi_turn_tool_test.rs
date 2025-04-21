use language_barrier_core::SingleRequestExecutor;
use language_barrier_core::model::{Claude, Sonnet35Version};
use language_barrier_core::{Chat, Message};
use tracing::{Level, info};
use test_tools::WeatherTool;

// Import our helper modules
mod test_tools;
mod test_utils;

use test_utils::{
    get_anthropic_provider, setup_tracing,
};

#[tokio::test]
async fn test_multi_turn_conversation_with_tools() {
    setup_tracing(Level::DEBUG);
    info!("Starting test_multi_turn_conversation_with_tools");

    // Skip test if no API key is available
    let Some(provider) = get_anthropic_provider() else {
        info!("Skipping test_multi_turn_conversation_with_tools: No API key available");
        return;
    };

    // Create a chat with tools for San Francisco weather
    let chat_sf = Chat::new(Claude::Sonnet35 {
        version: Sonnet35Version::V2,
    })
    .with_system_prompt("You are a helpful AI assistant that can provide weather information.")
    .with_max_output_tokens(1000)
    .with_tool(WeatherTool)
    .unwrap()
    .add_message(Message::user("What's the weather like in San Francisco?"));

    // Get the first response (San Francisco)
    let executor = SingleRequestExecutor::new(provider.clone());
    
    if let Ok(Message::Assistant { tool_calls, .. }) = executor.send(chat_sf).await {
        assert!(!tool_calls.is_empty(), "Expected tool calls for San Francisco");
    } else {
        panic!("Expected assistant message for San Francisco");
    }

    // Create a new chat for New York weather
    let chat_ny = Chat::new(Claude::Sonnet35 {
        version: Sonnet35Version::V2,
    })
    .with_system_prompt("You are a helpful AI assistant that can provide weather information.")
    .with_max_output_tokens(1000)
    .with_tool(WeatherTool)
    .unwrap()
    .add_message(Message::user("What's the weather like in New York?"));

    // Get the second response (New York)
    if let Ok(Message::Assistant { tool_calls, .. }) = executor.send(chat_ny).await {
        assert!(!tool_calls.is_empty(), "Expected tool calls for New York");
        
        // Verify there's a reference to New York in the tool calls
        let references_ny = tool_calls.iter().any(|call| call.function.arguments.contains("New York"));
        assert!(references_ny, "Expected tool call to reference New York");
    } else {
        panic!("Expected assistant message for New York");
    }
}