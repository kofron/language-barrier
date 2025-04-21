use language_barrier_core::SingleRequestExecutor;

use language_barrier_core::model::{Claude, GPT, Gemini, Mistral, Sonnet35Version};
use language_barrier_core::provider::HTTPProvider;
use language_barrier_core::{Chat, Message};
use test_tools::WeatherTool;
use tracing::{Level, error, info, warn};

// Import our helper modules
mod test_tools;
mod test_utils;

use test_utils::{
    get_anthropic_provider, get_gemini_provider, get_mistral_provider, get_openai_provider,
    setup_tracing,
};

// Test request creation with weather tools for all providers
#[tokio::test]
async fn test_weather_tool_request_creation() {
    setup_tracing(Level::DEBUG);
    info!("Starting test_weather_tool_request_creation for all providers");

    // Test Anthropic weather tool request creation
    {
        info!("Testing Anthropic weather tool request creation");
        let provider = get_anthropic_provider().unwrap_or_default();
        let model = Claude::Sonnet35 {
            version: Sonnet35Version::V2,
        };
        let mut chat = Chat::new(model)
            .with_system_prompt("You are a helpful AI assistant.")
            .with_max_output_tokens(1000)
            .with_tool(WeatherTool);

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
            .with_tool_registry(registry);

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
            .with_tool_registry(registry);

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
            .with_tool_registry(registry);

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

    // Create a WeatherToolbox with proper defaults
    let registry = test_tools::create_weather_registry();

    // Test with Anthropic if credentials available
    if let Some(provider) = get_anthropic_provider() {
        info!("Testing Anthropic weather tool integration");
        test_weather_tool_with_provider(
            "Anthropic",
            Chat::new(Claude::Sonnet35 {
                version: Sonnet35Version::V2,
            }),
            provider,
            registry,
        )
        .await;
    }

    // Test with OpenAI if credentials available
    if let Some(provider) = get_openai_provider() {
        info!("Testing OpenAI weather tool integration");
        test_weather_tool_with_provider("OpenAI", Chat::new(GPT::GPT4o), provider, registry).await;
    }

    // Test with Gemini if credentials available
    if let Some(provider) = get_gemini_provider() {
        info!("Testing Gemini weather tool integration");
        test_weather_tool_with_provider("Gemini", Chat::new(Gemini::Flash20), provider, registry)
            .await;
    }

    // Test with Mistral if credentials available
    if let Some(provider) = get_mistral_provider() {
        info!("Testing Mistral weather tool integration");
        test_weather_tool_with_provider("Mistral", Chat::new(Mistral::Small), provider, registry)
            .await;
    }
}

// Helper function to test weather tool with a specific provider
async fn test_weather_tool_with_provider<P, M>(
    provider_name: &str,
    base_chat: Chat<M>,
    provider: P,
    weather_registry: ToolRegistry,
) where
    P: HTTPProvider<M> + 'static,
    M: Clone + language_barrier_core::model::ModelInfo,
{
    // Create a chat with our configuration
    let mut chat = Chat::new(base_chat.model.clone())
        .with_system_prompt("You are a helpful AI assistant that can provide weather information. Always use the weather tool when asked about weather.")
        .with_max_output_tokens(1000)
        .with_tool_registry(weather_registry);

    // Add a user message
    chat.add_message(Message::user("What's the weather in Paris?"));

    // Create an executor with our provider
    let executor = SingleRequestExecutor::new(provider);

    // Get the response
    match executor.send(chat).await {
        Ok(Message::Assistant { tool_calls, .. }) if tool_calls.is_empty() => {
            warn!("{} assistant replied, but no tool called", provider_name);
        }
        Ok(Message::Assistant { .. }) => {
            info!("{} weather tool test successful", provider_name);
        }
        Ok(_) => {
            error!("{} messages are out of order")
        }

        Err(e) => {
            warn!("{} weather tool test failed: {}", provider_name, e);
        }
    }
}
