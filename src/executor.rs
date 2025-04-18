use reqwest::{Client, Response};

use crate::{Chat, Message, ModelInfo, error::Result, provider::HTTPProvider};

pub struct SingleRequestExecutor<M: ModelInfo> {
    provider: Box<dyn HTTPProvider<M>>,
    client: Client,
}

impl<M: ModelInfo> SingleRequestExecutor<M> {
    pub fn new(provider: impl HTTPProvider<M> + 'static) -> Self {
        Self {
            provider: Box::new(provider),
            client: Client::new(),
        }
    }

    pub async fn send(&self, chat: Chat<M>) -> Result<Message> {
        // Convert chat to request using provider
        let request = self.provider.accept(chat)?;

        // Send request and get response
        let response = self.client.execute(request).await?;

        // Parse response using provider
        let message = self.provider.parse(response.text().await?)?;

        Ok(message)
    }
}
