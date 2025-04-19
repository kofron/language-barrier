use reqwest::Client;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::{Chat, Message, ModelInfo, error::Result, provider::HTTPProvider};

pub struct SingleRequestExecutor<M: ModelInfo> {
    provider: Box<dyn HTTPProvider<M>>,
    client: Client,
}

impl<M: ModelInfo> SingleRequestExecutor<M> {
    #[instrument(skip_all, level = "debug")]
    pub fn new(provider: impl HTTPProvider<M> + 'static) -> Self {
        info!("Creating new SingleRequestExecutor");
        Self {
            provider: Box::new(provider),
            client: Client::new(),
        }
    }

    #[instrument(skip(self, chat), fields(model_type = std::any::type_name::<M>()), level = "debug")]
    pub async fn send(&self, chat: Chat<M>) -> Result<Message> {
        info!("Sending chat request with {} messages", chat.history.len());
        trace!("System prompt: {}", chat.system_prompt);
        
        // Convert chat to request using provider
        debug!("Converting chat to HTTP request");
        let request = match self.provider.accept(chat) {
            Ok(req) => {
                debug!("Request created successfully: {} {}", req.method(), req.url());
                trace!("Request headers: {:#?}", req.headers());
                req
            },
            Err(e) => {
                error!("Failed to create request: {}", e);
                return Err(e);
            }
        };
        
        // Send request and get response
        debug!("Sending HTTP request");
        let response = match self.client.execute(request).await {
            Ok(resp) => {
                info!("Received response with status: {}", resp.status());
                trace!("Response headers: {:#?}", resp.headers());
                resp
            },
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
            },
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
            },
            Err(e) => {
                error!("Failed to parse response: {}", e);
                return Err(e);
            }
        };
        
        Ok(message)
    }
}
