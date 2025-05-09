use dotenv::dotenv;
use language_barrier_core::message::{Content, ContentPart, Message};
use language_barrier_core::provider::anthropic::{AnthropicConfig, AnthropicProvider};
use language_barrier_core::provider::gemini::{GeminiConfig, GeminiProvider};
use language_barrier_core::provider::mistral::{MistralConfig, MistralProvider};
use language_barrier_core::provider::openai::{OpenAIConfig, OpenAIProvider};
use std::env;
use tracing::Level;
use tracing_subscriber::{EnvFilter, fmt, prelude::*, registry};

/// Helper function to set up tracing for tests
pub fn setup_tracing(level: Level) {
    let subscriber = registry()
        .with(
            fmt::layer()
                .with_test_writer()
                .with_ansi(false) // Better for CI logs
                .with_file(true) // Include source code location
                .with_line_number(true),
        )
        .with(
            EnvFilter::from_default_env()
                .add_directive(level.into())
                .add_directive("reqwest=info".parse().unwrap()),
        ); // Lower verbosity for reqwest

    let _ = tracing::subscriber::set_global_default(subscriber);
}

/// Get an Anthropic provider if API key is available
pub fn get_anthropic_provider() -> Option<AnthropicProvider> {
    // Opt-in to live provider tests via environment variable.
    if env::var("LIVE_PROVIDER_TESTS").unwrap_or_default() != "1" {
        return None;
    }
    dotenv().ok();
    match env::var("ANTHROPIC_API_KEY") {
        Ok(key) if !key.is_empty() => Some(AnthropicProvider::with_config(AnthropicConfig {
            api_key: key,
            base_url: "https://api.anthropic.com/v1".to_string(),
            api_version: "2023-06-01".to_string(),
        })),
        _ => None,
    }
}

/// Get an OpenAI provider if API key is available
pub fn get_openai_provider() -> Option<OpenAIProvider> {
    if env::var("LIVE_PROVIDER_TESTS").unwrap_or_default() != "1" {
        return None;
    }
    dotenv().ok();
    match env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() => Some(OpenAIProvider::with_config(OpenAIConfig {
            api_key: key,
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
        })),
        _ => None,
    }
}

/// Get a Gemini provider if API key is available
pub fn get_google_provider() -> Option<GeminiProvider> {
    if env::var("LIVE_PROVIDER_TESTS").unwrap_or_default() != "1" {
        return None;
    }
    dotenv().ok();
    match env::var("GEMINI_API_KEY") {
        Ok(key) if !key.is_empty() => Some(GeminiProvider::with_config(GeminiConfig {
            api_key: key,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        })),
        _ => None,
    }
}

/// Get a Mistral provider if API key is available
pub fn get_mistral_provider() -> Option<MistralProvider> {
    if env::var("LIVE_PROVIDER_TESTS").unwrap_or_default() != "1" {
        return None;
    }
    dotenv().ok();
    match env::var("MISTRAL_API_KEY") {
        Ok(key) if !key.is_empty() => Some(MistralProvider::with_config(MistralConfig {
            api_key: key,
            base_url: "https://api.mistral.ai/v1".to_string(),
        })),
        _ => None,
    }
}

/// Extract text content from a message
pub fn extract_text_content(message: &Message) -> String {
    match message {
        Message::Assistant { content, .. } => match content {
            Some(Content::Text(text)) => text.clone(),
            Some(Content::Parts(parts)) => {
                let mut combined = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        combined.push_str(text);
                    }
                }
                combined
            }
            None => String::new(),
        },
        Message::User { content, .. } => match content {
            Content::Text(text) => text.clone(),
            Content::Parts(parts) => {
                let mut combined = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        combined.push_str(text);
                    }
                }
                combined
            }
        },
        Message::System { content, .. } => content.clone(),
        Message::Tool { content, .. } => content.clone(),
    }
}

/// Check if a message has tool calls
pub fn has_tool_calls(message: &Message) -> bool {
    match message {
        Message::Assistant { tool_calls, .. } => !tool_calls.is_empty(),
        _ => false,
    }
}
