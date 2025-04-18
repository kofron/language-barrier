use dotenv::dotenv;
use language_barrier::SingleRequestExecutor;
use language_barrier::model::{Claude, Sonnet35Version};
use language_barrier::provider::HTTPProvider;
use language_barrier::provider::anthropic::{AnthropicConfig, AnthropicProvider};
use language_barrier::{Chat, Message};
use std::env;

#[tokio::test]
async fn test_anthropic_request_creation() {
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
    // Load environment variables from .env file if available
    dotenv().ok();

    // Skip test if no API key is available
    let api_key = match env::var("ANTHROPIC_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            println!("Skipping test: No ANTHROPIC_API_KEY found");
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

    // Create an executor with our provider
    let executor = SingleRequestExecutor::new(provider);

    // Create a chat with a simple prompt
    let model = Claude::Sonnet35 {
        version: Sonnet35Version::V2,
    };
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant that provides very short answers.")
        .with_max_output_tokens(100);

    // Add a user message
    chat.add_message(Message::user("What is the capital of France?"));

    // Send the chat and get the response
    let response = executor.send(chat).await.unwrap();

    // Verify we got a response from the assistant
    assert_eq!(
        response.role,
        language_barrier::message::MessageRole::Assistant
    );
    assert!(response.content.is_some());

    // Print the response for manual verification
    println!("Response: {:?}", response.content);

    // Verify token usage metadata is present
    assert!(response.metadata.contains_key("input_tokens"));
    assert!(response.metadata.contains_key("output_tokens"));
}
