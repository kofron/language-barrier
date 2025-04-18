use crate::error::Result;
use crate::{Chat, Message, ModelInfo};
use async_trait::async_trait;
use reqwest::{Request, Response};

// Include the provider-specific modules
pub mod anthropic;

/// An HTTPProvider can take a chat and turn it into an http request.
#[async_trait]
pub trait HTTPProvider<M: ModelInfo> {
    async fn accept(&self, chat: Chat<M>) -> Result<Request>;
    async fn parse(&self, response: Response) -> Result<Message>;
}
