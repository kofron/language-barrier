use std::env;
use std::{io, io::Write};

use language_barrier_core::provider::anthropic::AnthropicConfig;
use language_barrier_core::{
    model::Claude, provider::anthropic::AnthropicProvider,
};
use language_barrier_runtime::{
    middleware::{GenerateNextMessageService, FinalInterpreter, ServiceBuilder},
    ops,
};
use tower_service::Service;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

/// A simple interactive chat application using the Tower middleware architecture.
/// This demonstrates how to build a complete chat application with the language-barrier-runtime
/// using the free monad operations pattern and Tower middleware.
async fn run_chat() -> language_barrier_core::error::Result<()> {
    // Get API key from the environment
    let _ = dotenvy::dotenv();
    let api_key = env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set as an environment variable");

    println!("=== Language Barrier Chat ===");
    println!("Type your messages and press enter to send.");
    println!("Type 'exit' to quit.");
    println!();

    // Initialize the Anthropic provider
    let provider = AnthropicProvider::with_config(AnthropicConfig {
        api_key,
        base_url: "https://api.anthropic.com/v1".to_string(),
        api_version: "2023-06-01".to_string(),
    });

    // Create the model and provider arcs
    let model = std::sync::Arc::new(Claude::Opus3);
    let provider = std::sync::Arc::new(provider);

    // Configure the middleware stack
    let mut service = ServiceBuilder::new()
        .service(
            GenerateNextMessageService::new(FinalInterpreter::new(), model, provider)
        );

    // Keep track of conversation history
    let mut chat = language_barrier_core::chat::Chat::new().with_system_prompt(
        "You are a helpful assistant. Provide clear and concise answers to the user's questions.",
    );

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
                // Unwrap the Result<Chat<M>> to get the updated chat
                let updated_chat = chat_result.unwrap();

                // Create a chat program with the updated history
                ops::chat(updated_chat.history.clone(), None)
            });

        // Send the message and get response
        let result = service.call(add_message_program).await??;

        // Print the response
        println!("Assistant: {:?}", result);
        println!();

        // Update the chat history with both messages
        chat = chat.add_message(user_message).add_message(result);
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

    info!("Starting Language Barrier Chat");

    // Run the chat application
    if let Err(e) = run_chat().await {
        eprintln!("Error: {}", e);
    }
}
