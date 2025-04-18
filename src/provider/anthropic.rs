use crate::Chat;
use crate::error::Result;
use crate::provider::HTTPProvider;
use async_trait::async_trait;
use reqwest::{Method, Request};
use serde::{Deserialize, Serialize};
use std::env;

/// Configuration for the Anthropic provider
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the API
    pub base_url: String,
    /// API version header
    pub api_version: String,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            api_version: "2023-06-01".to_string(),
        }
    }
}

/// Implementation of the Anthropic provider
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    /// Configuration for the provider
    config: AnthropicConfig,
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl HTTPProvider for AnthropicProvider {
    async fn accept<M>(&self, chat: Chat<M>) -> Result<Request> {
        let mut request = Request::new(Method::POST, format!("{}/messages", self.config.base_url));
        request.headers().append("x-api-key", self.config.api_key);
        request.headers().append("Content-Type", "application/json");
        request.headers().insert(
            "anthropic-version".to_string(),
            self.config.api_version.clone(),
        );
        Ok(request)
    }
}

/// Represents a message in the Anthropic API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicMessage {
    /// The role of the message sender (user or assistant)
    pub role: String,
    /// The content of the message
    pub content: Vec<AnthropicContentPart>,
}

/// Represents a content part in an Anthropic message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicContentPart {
    /// Text content
    #[serde(rename = "text")]
    Text {
        /// The text content
        text: String,
    },
    /// Image content
    #[serde(rename = "image")]
    Image {
        /// The source of the image
        source: AnthropicImageSource,
    },
    /// Tool result content (for tool responses)
    #[serde(rename = "tool_result")]
    ToolResult(AnthropicToolResponse),
}

impl AnthropicContentPart {
    /// Create a new text content part
    fn text(text: String) -> Self {
        AnthropicContentPart::Text { text }
    }

    /// Create a new image content part
    fn image(url: String) -> Self {
        AnthropicContentPart::Image {
            source: AnthropicImageSource {
                type_field: "base64".to_string(),
                media_type: "image/jpeg".to_string(),
                data: url,
            },
        }
    }
}

/// Represents the source of an image in an Anthropic message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicImageSource {
    /// The type of the image source (base64 or url)
    #[serde(rename = "type")]
    pub type_field: String,
    /// The media type of the image
    pub media_type: String,
    /// The image data (base64 or url)
    pub data: String,
}

/// Represents a tool response in an Anthropic message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicToolResponse {
    /// The type of the tool response
    #[serde(rename = "type")]
    pub type_field: String,
    /// The ID of the tool call
    pub tool_call_id: String,
    /// The content of the tool response
    pub content: String,
}

/// Represents a tool in the Anthropic API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicTool {
    /// The name of the tool
    pub name: String,
    /// The description of the tool
    pub description: String,
    /// The input schema for the tool
    pub input_schema: serde_json::Value,
}

/// Represents a response from the Anthropic API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicResponse {
    /// The ID of the response
    pub id: String,
    /// The type of the response
    #[serde(rename = "type")]
    pub type_field: String,
    /// The role of the message
    pub role: String,
    /// The model used for generation
    pub model: String,
    /// Whether the response is complete
    pub stop_reason: Option<String>,
    /// The content of the response
    pub content: Vec<AnthropicResponseContent>,
    /// Usage information
    pub usage: AnthropicUsage,
}

/// Represents a content part in an Anthropic response
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicResponseContent {
    /// Text content
    #[serde(rename = "text")]
    Text {
        /// The text content
        text: String,
    },
    /// Tool use content
    #[serde(rename = "tool_use")]
    ToolUse {
        /// The ID of the tool use
        id: String,
        /// The name of the tool
        name: String,
        /// The input to the tool
        input: serde_json::Value,
    },
}

/// Represents usage information in an Anthropic response
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicUsage {
    /// Number of tokens in the input
    pub input_tokens: u32,
    /// Number of tokens in the output
    pub output_tokens: u32,
}

#[cfg(test)]
mod tests {}
