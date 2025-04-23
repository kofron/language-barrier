use std::env;
use std::{io, io::Write, sync::Arc};

use language_barrier_core::provider::anthropic::AnthropicConfig;
use language_barrier_core::tool::ToolChoice;
use language_barrier_core::{Message, ToolDefinition};
use language_barrier_core::{model::Claude, provider::anthropic::AnthropicProvider};
use language_barrier_runtime::{
    middleware::{FinalInterpreter, GenerateNextMessageService, ToolExecutorMiddleware},
    ops,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tower_service::Service;
use tracing::{Level, debug, info};
use tracing_subscriber::FmtSubscriber;

/// A simple interactive chat application with weather tool using the Tower middleware architecture.
/// This demonstrates how to build a chat application with tool support using the language-barrier-runtime.

// Weather Tool Definition
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct WeatherRequest {
    location: String,
    #[schemars(default, description = "Temperature unit: 'celsius' or 'fahrenheit'")]
    units: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct WeatherResponse {
    temperature: i32,
    conditions: String,
    location: String,
    units: String,
}

#[derive(Clone)]
struct WeatherTool;

impl ToolDefinition for WeatherTool {
    type Input = WeatherRequest;
    type Output = WeatherResponse;

    fn name(&self) -> String {
        "get_weather".to_string()
    }

    fn description(&self) -> String {
        "Get current weather for a location".to_string()
    }
}

async fn run_weather_chat() -> language_barrier_core::error::Result<()> {
    // Get API key from the environment
    let _ = dotenvy::dotenv();
    let api_key = env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set as an environment variable");

    println!("=== Language Barrier Weather Chat ===");
    println!("Type your messages and press enter to send.");
    println!("Ask about the weather in any location!");
    println!("Type 'exit' to quit.");
    println!();

    // Initialize the Anthropic provider
    let provider = AnthropicProvider::with_config(AnthropicConfig {
        api_key,
        base_url: "https://api.anthropic.com/v1".to_string(),
        api_version: "2023-06-01".to_string(),
    });

    // Create the model and provider arcs
    let model = Arc::new(Claude::Haiku35);
    let provider = Arc::new(provider);

    // Create weather tool function
    let weather_tool = WeatherTool;
    let weather_fn = Arc::new(move |req: WeatherRequest| -> WeatherResponse {
        // Log the incoming request
        debug!(
            "Weather tool called with location: {} and units: {:?}",
            req.location, req.units
        );

        // Static weather response for demonstration
        WeatherResponse {
            temperature: 45,
            conditions: "Partly Cloudy".to_string(),
            location: req.location,
            units: req.units.unwrap_or_else(|| "fahrenheit".to_string()),
        }
    });

    // First create the base service
    let base_service = GenerateNextMessageService::new(FinalInterpreter::new(), model, provider);

    // Then wrap it with the tool executor middleware using auto-execute mode
    let mut service =
        ToolExecutorMiddleware::with_auto_execute(base_service, weather_tool.clone(), weather_fn.clone());
    
    debug!("Created service with auto-execute mode enabled");

    // Keep track of conversation history
    let mut chat = language_barrier_core::chat::Chat::new()
        .with_system_prompt("You are a helpful assistant with access to weather information.")
        .with_tool(weather_tool.clone())
        .expect("Failed to add weather tool to chat")
        .with_tool_choice(ToolChoice::Auto);

    debug!("Chat has tools configured: {:?}", chat.tools);

    // Main conversation loop
    loop {
        // Print prompt
        print!("> ");
        io::stdout().flush().unwrap();

        // Read user input
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        let input = input.trim();

        // Check for exit command
        if input.to_lowercase() == "exit" {
            println!("Goodbye!");
            break;
        }

        // Create user message
        let user_message = ops::user_message(input);

        // Add message to chat history
        let add_message_program =
            ops::add_message(chat.clone(), user_message.clone()).and_then(|chat_result| {
                // Unwrap the Result<Chat> to get the updated chat
                let updated_chat = chat_result.unwrap();

                // Create a chat program with the updated history
                ops::generate_next_message(updated_chat)
            });

        // Send the message and get response
        let result = service.call(add_message_program).await??;
        println!("{:?}", result.most_recent_message());

        if let Some(Message::Assistant { tool_calls, .. }) = result.most_recent_message() {
            if !tool_calls.is_empty() {
                break;
            }
        }

        // Update the chat history with both messages
        chat = result;
    }

    Ok(())
}

#[tokio::main]
pub async fn main() {
    // Set up tracing
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    info!("Starting Language Barrier Weather Chat");

    // Run the chat application
    if let Err(e) = run_weather_chat().await {
        eprintln!("Error: {}", e);
    }
}
