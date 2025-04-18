use language_barrier::chat::Chat;
use language_barrier::message::Message;
use language_barrier::model::Claude;
use language_barrier::provider::HTTPProvider;
use language_barrier::provider::anthropic::{AnthropicConfig, AnthropicProvider};

#[tokio::test]
async fn test_anthropic_request_creation() {
    // This test may fail due to transitional state in the codebase
    // We'll implement it properly but comment it out for now

    // Create a provider with a test API key
    let config = AnthropicConfig {
        api_key: "test-api-key".to_string(),
        base_url: "https://api.anthropic.com/v1".to_string(),
        api_version: "2023-06-01".to_string(),
    };
    let provider = AnthropicProvider::with_config(config);

    // Create a chat with a model
    let model = Claude::Sonnet37 {
        use_extended_thinking: false,
    };
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful assistant.")
        .with_max_output_tokens(1024);

    // Add some messages
    chat.push_message(Message::user("Hello, how are you?"));
    chat.push_message(Message::assistant("I'm doing well, thank you for asking!"));
    chat.push_message(Message::user("Can you help me with a question?"));

    // Create the request
    let request = provider.accept(chat).await.unwrap();

    // Verify the request
    assert_eq!(request.method(), "POST");
    assert_eq!(
        request.url().as_str(),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(request.headers()["x-api-key"], "test-api-key");
    assert_eq!(request.headers()["anthropic-version"], "2023-06-01");
    assert_eq!(request.headers()["Content-Type"], "application/json");

    // We can't easily check the body in a Request, but we've verified
    // the payload creation in unit tests
}
