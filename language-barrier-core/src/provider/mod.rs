use std::sync::Arc;

use crate::error::Result;
use crate::{Chat, Message, ModelInfo};

use reqwest::Request;

// Include the provider-specific modules
pub mod anthropic;
pub mod gemini;
pub mod mistral;
pub mod openai;

/// An `HTTPProvider` can take a chat and turn it into an http request.
pub trait HTTPProvider<M: ModelInfo>: Send + Sync {
    /// Converts a chat into an HTTP request
    ///
    /// # Errors
    ///
    /// Returns an error if the chat cannot be converted to a request, for example
    /// if the provider configuration is invalid or if serialization fails.
    fn accept(&self, model: M, chat: &Chat) -> Result<Request>;

    /// Parses a raw HTTP response into a message
    ///
    /// # Errors
    ///
    /// Returns an error if the response cannot be parsed, for example if the
    /// response is not valid JSON or if it contains an error status.
    fn parse(&self, raw_response_text: String) -> Result<Message>;
}
