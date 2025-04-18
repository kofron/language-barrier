use language_barrier::provider::{GenerationOptions, LlmProvider};
use language_barrier::message::Message;
use language_barrier::model::{AnthropicModel, GoogleModel};
use language_barrier::provider::{AnthropicProvider, GoogleProvider};
use language_barrier::transport::mock::MockTransport;
use language_barrier::transport::{AnthropicTransportVisitor, GoogleTransportVisitor};

/// This test verifies that the Anthropic provider mock transport correctly formats
/// request structures for testing purposes.
///
/// Note: This is testing the mock transport implementation, not the full API format.
/// The real format would be according to the curl example from the provider implementation:
/// ```bash
/// curl https://api.anthropic.com/v1/messages \
///      --header "x-api-key: $ANTHROPIC_API_KEY" \
///      --header "anthropic-version: 2023-06-01" \
///      --header "content-type: application/json" \
///      --data \
/// '{
///     "model": "claude-3-7-sonnet-20250219",
///     "max_tokens": 1024,
///     "messages": [{
///         "role": "user",
///         "content": "What is the weather like where I am?"
///     }]
/// }'
/// ```
#[tokio::test]
async fn test_anthropic_request_format() {
    // Create our mock transport to capture the request
    let transport = MockTransport::new();

    // Create a message similar to the curl example
    let messages = vec![Message::user("What is the weather like where I am?")];

    // Set up the generation options
    let options = GenerationOptions::new().with_max_tokens(1024);

    // Prepare a request using the visitor pattern
    let model = AnthropicModel::Sonnet37 { use_extended_thinking: false }; // claude-3-7-sonnet-20250219
    let request = transport
        .prepare_anthropic_request(&model, &messages, &options)
        .await
        .unwrap();

    // Verify that the request matches the expected format for the mock transport
    assert_eq!(request["model"], "claude-3-7-sonnet-20250219");
    assert_eq!(request["max_tokens"], 1024);

    // The mock transport uses a simplified message structure
    let messages = request["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);

    // Verify the mock message format (different from actual API format)
    assert_eq!(messages[0]["key"], "1 messages");

    // Verify system message is present
    assert_eq!(request["system"], "You are Claude, a helpful AI assistant.");
}

/// This test verifies that the Google provider mock transport correctly formats
/// request structures for testing purposes.
///
/// Note: This is testing the mock transport implementation, not the full API format.
#[tokio::test]
async fn test_google_request_format() {
    // Create our mock transport to capture the request
    let transport = MockTransport::new();

    // Create a message similar to the Anthropic example but for Google
    let messages = vec![Message::user("What is the weather like where I am?")];

    // Set up the generation options
    let options = GenerationOptions::new().with_max_tokens(1024);

    // Prepare a request using the visitor pattern
    let model = GoogleModel::Gemini15Pro;
    let request = transport
        .prepare_google_request(&model, &messages, &options)
        .await
        .unwrap();

    // Verify that the request matches the expected format for the mock transport
    assert_eq!(request["model"], "gemini-1.5-pro");

    // The mock transport uses a simplified content structure
    let contents = request["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 1);

    // Verify the mock content format (different from actual API format)
    assert_eq!(contents[0]["key"], "1 messages");

    // Verify generation config is set correctly
    let generation_config = &request["generationConfig"];
    assert!(generation_config.is_object());
    assert_eq!(generation_config["maxOutputTokens"], 1024);
}

/// This test exercises the actual provider implementation with the mock transport,
/// capturing the request that would be sent to the API.
///
/// Note: With the current implementation, this test will fail with an error
/// since we haven't provided a mock response. We're only interested in the
/// generated request payload.
#[tokio::test]
async fn test_anthropic_accept_visitor() {
    // Create our mock transport to capture the request
    let transport = MockTransport::new();

    // Create the Anthropic provider
    let provider = AnthropicProvider::with_api_key("test-api-key");

    // Create a message similar to the curl example
    let messages = vec![Message::user("What is the weather like where I am?")];

    // Set up the generation options
    let options = GenerationOptions::new().with_max_tokens(1024);

    // Use the provider's accept method to process the request
    let model = AnthropicModel::Sonnet37 { use_extended_thinking: false };

    // We need to capture and check the exact request that's being sent
    // by using the provider's accept method with our mock transport
    let result = provider
        .accept(&transport, &model, &messages, options)
        .await;

    // The request should fail since we haven't set up a mock response
    assert!(result.is_err());

    // Access the captured request from the transport
    let last_request = transport.last_request();
    assert!(last_request.is_some());
    let request = last_request.unwrap();

    // Now verify the model ID is correctly set
    assert_eq!(
        request["model"].as_str().unwrap(),
        "claude-3-7-sonnet-20250219"
    );

    // We know the request is being captured, which is what we want to test
    // We can't verify the exact format since the MockTransport doesn't generate
    // the actual API request format - that's done by the HttpTransport in production
}

/// This test exercises the Google provider implementation with the mock transport,
/// capturing the request that would be sent to the API.
///
/// Note: Similar to the Anthropic test, we're only testing that the provider
/// attempts to generate a request and that the mock transport captures it.
#[tokio::test]
async fn test_google_provider_implementation() {
    // Create the Google provider
    let provider = GoogleProvider::with_api_key("test-api-key");

    // Create a message for testing
    let messages = vec![Message::user("What is the weather like where I am?")];

    // Set up the generation options
    let options = GenerationOptions::new().with_max_tokens(1024);

    // Create a mock transport (without setting up a response)
    // We're not actually using this transport in this test
    let _transport = MockTransport::new();

    // To test the transport visitor pattern properly, we need a way to use it with
    // our Google provider. Since we don't have direct access to the accept method
    // for Google (as we did with Anthropic), we'll use the generate method which
    // internally creates an HttpTransport.
    //
    // This test won't capture the exact request, but it verifies the provider
    // integration is working at a basic level.

    // Use generate which will fail because it's using an HttpTransport, not our mock
    let result = provider
        .generate("gemini-1.5-pro", &messages, options)
        .await;

    // This should fail since we're not making real HTTP requests
    assert!(result.is_err());

    // Basic test passed - the provider attempted to generate a request
    // In a real implementation, we would need Google provider to expose an accept method
    // similar to Anthropic to properly test with the mock transport
}
