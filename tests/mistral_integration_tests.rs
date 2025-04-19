use dotenv::dotenv;
use language_barrier::SingleRequestExecutor;
use language_barrier::model::Mistral;
use language_barrier::provider::HTTPProvider;
use language_barrier::provider::mistral::{MistralConfig, MistralProvider};
use language_barrier::{Chat, Message, Tool, ToolDescription, Toolbox};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use tracing::{debug, info, warn, error, Level};
use tracing_subscriber::{fmt, prelude::*, registry, EnvFilter};

#[tokio::test]
async fn test_mistral_request_creation() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_mistral_request_creation");
    
    // Create a Mistral provider with default configuration
    let provider = MistralProvider::new();

    // Create a chat with a simple message
    let model = Mistral::Small;
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
        "https://api.mistral.ai/v1/chat/completions"
    );
    assert!(request.headers().contains_key("Authorization"));
    assert_eq!(
        request.headers().get("Content-Type").unwrap(),
        "application/json"
    );
}

#[tokio::test]
async fn test_mistral_integration_with_executor() {
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
    
    info!("Starting test_mistral_integration_with_executor");

    // Load environment variables from .env file if available
    info!("Loading environment variables from .env");
    dotenv().ok();

    // Skip test if no API key is available
    let api_key = match env::var("MISTRAL_API_KEY") {
        Ok(key) if !key.is_empty() => {
            info!("API key found in environment");
            debug!("API key length: {}", key.len());
            key
        },
        _ => {
            warn!("Skipping test: No MISTRAL_API_KEY found");
            return;
        }
    };

    // Create a provider with the API key
    info!("Creating Mistral provider with custom config");
    let config = MistralConfig {
        api_key,
        base_url: "https://api.mistral.ai/v1".to_string(),
    };
    let provider = MistralProvider::with_config(config);

    // Create an executor with our provider
    info!("Creating SingleRequestExecutor");
    let executor = SingleRequestExecutor::new(provider);

    // Create a chat with a simple prompt
    info!("Creating Chat with Mistral Small");
    let model = Mistral::Small;
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant that provides very short answers.")
        .with_max_output_tokens(100);

    // Add a user message
    info!("Adding user message to chat");
    chat.add_message(Message::user("What is the capital of France?"));
    debug!("Message count: {}", chat.history.len());

    // Send the chat and get the response
    info!("Sending chat request to Mistral API");
    let response = match executor.send(chat).await {
        Ok(resp) => {
            info!("Successfully received response from Mistral API");
            resp
        },
        Err(e) => {
            error!("Failed to get response from Mistral API: {}", e);
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
    debug!("Token usage metadata: {:?}", response.metadata);
    assert!(response.metadata.contains_key("prompt_tokens") || 
           response.metadata.contains_key("total_tokens"));
}

// Define a simple test tool for weather
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct TestWeatherTool {
    location: String,
}

impl Tool for TestWeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get weather information for a location"
    }
}

// Simple toolbox implementation for testing
struct TestToolbox;

impl Toolbox for TestToolbox {
    fn describe(&self) -> Vec<ToolDescription> {
        // Create schema for TestWeatherTool
        let weather_schema = schemars::schema_for!(TestWeatherTool);
        let weather_schema_value = serde_json::to_value(weather_schema.schema).unwrap();

        vec![
            ToolDescription {
                name: "get_weather".to_string(),
                description: "Get weather information for a location".to_string(),
                parameters: weather_schema_value,
            },
        ]
    }

    fn execute(&self, name: &str, arguments: Value) -> language_barrier::Result<String> {
        match name {
            "get_weather" => {
                let request: TestWeatherTool = serde_json::from_value(arguments)?;
                Ok(format!("Weather in {}: Sunny, 72°F", request.location))
            }
            _ => Err(language_barrier::Error::ToolNotFound(name.to_string())),
        }
    }
}

#[tokio::test]
async fn test_mistral_tools_request_creation() {
    // Initialize tracing for this test
    let subscriber = registry()
        .with(fmt::layer().with_test_writer())
        .with(EnvFilter::from_default_env().add_directive(Level::DEBUG.into()));
    tracing::subscriber::set_global_default(subscriber).ok();
    
    info!("Starting test_mistral_tools_request_creation");
    
    // Create a Mistral provider with default configuration
    let provider = MistralProvider::new();

    // Create a chat with a simple message and tools
    let model = Mistral::Small;
    let mut chat = Chat::new(model)
        .with_system_prompt("You are a helpful AI assistant.")
        .with_max_output_tokens(1000)
        .with_toolbox(TestToolbox);

    // Add a user message
    chat.add_message(Message::user("What's the weather in San Francisco?"));

    // Create a request using the provider
    let request = provider.accept(chat).unwrap();

    // Get the request body as a string
    let body_bytes = request.body().unwrap().as_bytes().unwrap();
    let body_str = std::str::from_utf8(body_bytes).unwrap();
    
    // Check that the request includes tools
    assert!(body_str.contains("\"tools\""));
    assert!(body_str.contains("\"get_weather\""));
    assert!(body_str.contains("\"location\""));
    
    // Verify request properties
    assert_eq!(request.method(), "POST");
    assert_eq!(
        request.url().as_str(),
        "https://api.mistral.ai/v1/chat/completions"
    );
    assert!(request.headers().contains_key("Authorization"));
    assert_eq!(
        request.headers().get("Content-Type").unwrap(),
        "application/json"
    );
}