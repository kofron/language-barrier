use async_trait::async_trait;
use reqwest::Client;
use std::sync::Arc;
use tracing::{debug, error, info, trace};

use crate::{Chat, Message, ModelInfo, Result, provider::HTTPProvider};

/// This is anything that can generate the next message.
///
/// LLMService is responsible for generating the next message in a conversation.
/// It replaces the previous SingleRequestExecutor pattern with a more flexible
/// trait-based approach, allowing for different implementations (HTTP, local, etc).
#[async_trait]
pub trait LLMService<M: ModelInfo> {
    /// Generates the next message in the conversation.
    ///
    /// Takes a Chat instance and returns a Result containing the next message.
    async fn generate_next_message(&self, chat: &Chat) -> Result<Message>;
}

/// An LLM service implementation that sends requests over HTTP.
///
/// This implementation of LLMService uses HTTP to communicate with language model providers.
/// It replaces the previous SingleRequestExecutor, providing the same functionality
/// but with a more flexible trait-based design.
///
/// # Examples
///
/// ```no_run
/// use language_barrier_core::{Chat, Message, model::Claude};
/// use language_barrier_core::llm_service::{HTTPLlmService, LLMService};
/// use language_barrier_core::provider::anthropic::AnthropicProvider;
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> language_barrier_core::Result<()> {
///     // Create a provider
///     let provider = AnthropicProvider::new();
///
///     // Create a service with the model and provider
///     let service = HTTPLlmService::new(
///         Claude::Opus3,
///         Arc::new(provider)
///     );
///
///     // Create a chat and generate a response
///     let chat = Chat::new()
///         .with_system_prompt("You are a helpful assistant.")
///         .add_message(Message::user("Hello, how are you?"));
///
///     let response = service.generate_next_message(chat).await?;
///
///     Ok(())
/// }
/// ```
pub struct HTTPLlmService<M: ModelInfo> {
    model: M,
    provider: Arc<dyn HTTPProvider<M>>,
}

impl<M: ModelInfo> HTTPLlmService<M> {
    pub fn new(model: M, provider: Arc<dyn HTTPProvider<M>>) -> Self {
        HTTPLlmService { model, provider }
    }
}

#[async_trait]
impl<M: ModelInfo> LLMService<M> for HTTPLlmService<M> {
    async fn generate_next_message(&self, chat: &Chat) -> Result<Message> {
        let client = Client::new();

        let request = match self.provider.accept(self.model, chat) {
            Ok(req) => {
                debug!(
                    "Request created successfully: {} {}",
                    req.method(),
                    req.url()
                );
                trace!("Request headers: {:#?}", req.headers());
                req
            }
            Err(e) => {
                error!("Failed to create request: {}", e);
                return Err(e);
            }
        };

        // Send request and get response
        debug!("Sending HTTP request");
        let response = match client.execute(request).await {
            Ok(resp) => {
                info!("Received response with status: {}", resp.status());
                trace!("Response headers: {:#?}", resp.headers());
                resp
            }
            Err(e) => {
                error!("HTTP request failed: {}", e);
                return Err(e.into());
            }
        };

        // Get response text
        debug!("Reading response body");
        let response_text = match response.text().await {
            Ok(text) => {
                trace!("Response body: {}", text);
                text
            }
            Err(e) => {
                error!("Failed to read response body: {}", e);
                return Err(e.into());
            }
        };

        // Parse response using provider
        debug!("Parsing response");
        let message = match self.provider.parse(response_text) {
            Ok(msg) => {
                info!("Successfully parsed response into message");
                debug!("Message role: {}", msg.role_str());
                // Message content is now accessed through pattern matching
                msg
            }
            Err(e) => {
                error!("Failed to parse response: {}", e);
                return Err(e);
            }
        };

        Ok(message)
    }
}
