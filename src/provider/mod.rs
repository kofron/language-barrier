use crate::error::Result;
use crate::{Chat, ModelInfo};
use async_trait::async_trait;
use reqwest::Request;

// Include the provider-specific modules
pub mod anthropic;

/// An HTTPProvider can take a chat and turn it into an http request.
#[async_trait]
pub trait HTTPProvider {
    async fn accept<M: ModelInfo>(&self, chat: Chat<M>) -> Result<Request>;
}
