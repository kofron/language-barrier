use dotenv::dotenv;
use language_barrier::SingleRequestExecutor;
use language_barrier::model::{Claude, Gemini, Sonnet35Version};
use language_barrier::provider::HTTPProvider;
use language_barrier::provider::anthropic::{AnthropicConfig, AnthropicProvider};
use language_barrier::provider::gemini::{GeminiConfig, GeminiProvider};
use language_barrier::{Chat, Message};
use std::env;
use tracing::{debug, info, warn, error, Level};
use tracing_subscriber::{fmt, prelude::*, registry, EnvFilter};

#[tokio::test]
async fn test_anthropic_request_creation() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_anthropic_request_creation");
    
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
    // Initialize tracing for this test with detailed output
    let subscriber = registry()
        .with(fmt::layer()
            .with_test_writer()
            .with_ansi(false) // Better for CI logs
            .with_file(true)  // Include source code location
            .with_line_number(true))
        .with(EnvFilter::from_default_env()
            .add_directive(Level::TRACE.into())  // Maximum verbosity
            .add_directive("reqwest=info".parse().unwrap())); // Lower verbosity for reqwest
    
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_anthropic_integration_with_executor");

    // Load environment variables from .env file if available
    info!("Loading environment variables from .env");
    dotenv().ok();

    // Skip test if no API key is available
    let api_key = match env::var("ANTHROPIC_API_KEY") {
        Ok(key) if !key.is_empty() => {
            info!("API key found in environment");
            debug!("API key length: {}", key.len());
            key
        },
        _ => {
            warn!("Skipping test: No ANTHROPIC_API_KEY found");
            return;
        }
    };

    // Create a provider with the API key
    info!("Creating Anthropic provider with custom config");
    let config = AnthropicConfig {
        api_key,
        base_url: "https://api.anthropic.com/v1".to_string(),
        api_version: "2023-06-01".to_string(),
    };
    let provider = AnthropicProvider::with_config(config);

    // Create an executor with our provider
    info!("Creating SingleRequestExecutor");
    let executor = SingleRequestExecutor::new(provider);

    // Create a chat with a simple prompt
    info!("Creating Chat with Sonnet 3.5");
    let model = Claude::Sonnet35 {
        version: Sonnet35Version::V2,
    };
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant that provides very short answers.")
        .with_max_output_tokens(100);

    // Add a user message
    info!("Adding user message to chat");
    chat.add_message(Message::user("What is the capital of France?"));
    debug!("Message count: {}", chat.history.len());

    // Send the chat and get the response
    info!("Sending chat request to Anthropic API");
    let response = match executor.send(chat).await {
        Ok(resp) => {
            info!("Successfully received response from Anthropic API");
            resp
        },
        Err(e) => {
            error!("Failed to get response from Anthropic API: {}", e);
            panic!("Test failed: {}", e);
        }
    };

    // Verify we got a response from the assistant
    info!("Verifying response");
    debug!("Response role: {:?}", response.role);
    assert_eq!(
        response.role,
        language_barrier::message::MessageRole::Assistant
    );
    assert!(response.content.is_some());

    // Print the response for manual verification
    info!("Test completed successfully");
    debug!("Response content: {:?}", response.content);
    println!("Response: {:?}", response.content);

    // Verify token usage metadata is present
    debug!("Token usage - input: {:?}, output: {:?}", 
        response.metadata.get("input_tokens"),
        response.metadata.get("output_tokens"));
    assert!(response.metadata.contains_key("input_tokens"));
    assert!(response.metadata.contains_key("output_tokens"));
}

#[tokio::test]
async fn test_gemini_request_creation() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_gemini_request_creation");
    
    // Create a Gemini provider with default configuration
    let provider = GeminiProvider::new();

    // Create a chat with a simple message
    let model = Gemini::Flash20;
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant.")
        .with_max_output_tokens(1000);

    // Add a user message
    chat.add_message(Message::user("What is the capital of France?"));

    // Create a request using the provider
    let request = provider.accept(chat).unwrap();

    // Verify request properties
    assert_eq!(request.method(), "POST");
    assert!(request.url().as_str().contains("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"));
    assert!(request.url().as_str().contains("key="));
    assert_eq!(
        request.headers().get("Content-Type").unwrap(),
        "application/json"
    );
}

#[tokio::test]
async fn test_gemini_integration_with_executor() {
    // Initialize tracing for this test with detailed output
    let subscriber = registry()
        .with(fmt::layer()
            .with_test_writer()
            .with_ansi(false) // Better for CI logs
            .with_file(true)  // Include source code location
            .with_line_number(true))
        .with(EnvFilter::from_default_env()
            .add_directive(Level::TRACE.into())  // Maximum verbosity
            .add_directive("reqwest=info".parse().unwrap())); // Lower verbosity for reqwest
    
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_gemini_integration_with_executor");

    // Load environment variables from .env file if available
    info!("Loading environment variables from .env");
    dotenv().ok();

    // Skip test if no API key is available
    let api_key = match env::var("GEMINI_API_KEY") {
        Ok(key) if !key.is_empty() => {
            info!("API key found in environment");
            debug!("API key length: {}", key.len());
            key
        },
        _ => {
            warn!("Skipping test: No GEMINI_API_KEY found");
            return;
        }
    };

    // Create a provider with the API key
    info!("Creating Gemini provider with custom config");
    let config = GeminiConfig {
        api_key,
        base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
    };
    let provider = GeminiProvider::with_config(config);

    // Create an executor with our provider
    info!("Creating SingleRequestExecutor");
    let executor = SingleRequestExecutor::new(provider);

    // Create a chat with a simple prompt
    info!("Creating Chat with Gemini Flash 2.0");
    let model = Gemini::Flash20;
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant that provides very short answers.")
        .with_max_output_tokens(100);

    // Add a user message
    info!("Adding user message to chat");
    chat.add_message(Message::user("What is the capital of France?"));
    debug!("Message count: {}", chat.history.len());

    // Send the chat and get the response
    info!("Sending chat request to Gemini API");
    let response = match executor.send(chat).await {
        Ok(resp) => {
            info!("Successfully received response from Gemini API");
            resp
        },
        Err(e) => {
            error!("Failed to get response from Gemini API: {}", e);
            panic!("Test failed: {}", e);
        }
    };

    // Verify we got a response from the assistant
    info!("Verifying response");
    debug!("Response role: {:?}", response.role);
    assert_eq!(
        response.role,
        language_barrier::message::MessageRole::Assistant
    );
    assert!(response.content.is_some());

    // Print the response for manual verification
    info!("Test completed successfully");
    debug!("Response content: {:?}", response.content);
    println!("Response: {:?}", response.content);

    // Verify token usage metadata is present (field names might be different from Anthropic)
    debug!("Token usage metadata: {:?}", response.metadata);
    assert!(response.metadata.contains_key("prompt_tokens") || 
           response.metadata.contains_key("total_tokens"));
}
