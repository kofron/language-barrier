use crate::error::Result;
use crate::{Chat, Message, ModelInfo};

use reqwest::Request;

// Include the provider-specific modules
pub mod anthropic;

/// An HTTPProvider can take a chat and turn it into an http request.
pub trait HTTPProvider<M: ModelInfo> {
    fn accept(&self, chat: Chat<M>) -> Result<Request>;
    
    fn parse(&self, raw_response_text: String) -> Result<Message>;
}
