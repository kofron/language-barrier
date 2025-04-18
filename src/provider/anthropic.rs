use crate::error::{Error, Result};
use crate::message::{Content, ContentPart, Message, MessageRole};
use crate::model::Sonnet35Version;
use crate::provider::HTTPProvider;
use crate::{Chat, Claude};
use async_trait::async_trait;
use reqwest::{Method, Request, Response, Url};
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

impl AnthropicProvider {
    /// Creates a new AnthropicProvider with default configuration
    ///
    /// This method will use the ANTHROPIC_API_KEY environment variable for authentication.
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::anthropic::AnthropicProvider;
    ///
    /// let provider = AnthropicProvider::new();
    /// ```
    pub fn new() -> Self {
        Self {
            config: AnthropicConfig::default(),
        }
    }

    /// Creates a new AnthropicProvider with custom configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::anthropic::{AnthropicProvider, AnthropicConfig};
    ///
    /// let config = AnthropicConfig {
    ///     api_key: "your-api-key".to_string(),
    ///     base_url: "https://api.anthropic.com/v1".to_string(),
    ///     api_version: "2023-06-01".to_string(),
    /// };
    ///
    /// let provider = AnthropicProvider::with_config(config);
    /// ```
    pub fn with_config(config: AnthropicConfig) -> Self {
        Self { config }
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl HTTPProvider<Claude> for AnthropicProvider {
    async fn accept(&self, chat: Chat<Claude>) -> Result<Request> {
        let url = Url::parse(format!("{}/messages", self.config.base_url).as_str())?;
        let mut request = Request::new(Method::POST, url);

        // Set headers
        request.headers_mut().insert(
            "x-api-key",
            self.config
                .api_key
                .parse()
                .map_err(|_| Error::Authentication("Invalid API key format".into()))?,
        );
        request.headers_mut().insert(
            "Content-Type",
            "application/json"
                .parse()
                .map_err(|_| Error::Other("Failed to set content type".into()))?,
        );
        request.headers_mut().insert(
            "anthropic-version",
            self.config
                .api_version
                .parse()
                .map_err(|_| Error::Other("Invalid API version format".into()))?,
        );

        // Create the request payload
        let payload = self.create_request_payload(&chat)?;

        // Set the request body
        *request.body_mut() = Some(
            serde_json::to_vec(&payload)
                .map_err(Error::Serialization)?
                .into(),
        );

        Ok(request)
    }

    async fn parse(&self, response: Response) -> Result<Message> {
        // Check if the response was successful
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.map_err(Error::Request)?;

            return match status.as_u16() {
                401 => Err(Error::Authentication(format!(
                    "Anthropic API authentication failed: {}",
                    error_text
                ))),
                429 => Err(Error::RateLimit(format!(
                    "Anthropic API rate limit exceeded: {}",
                    error_text
                ))),
                _ => Err(Error::Other(format!(
                    "Anthropic API error ({}): {}",
                    status, error_text
                ))),
            };
        }

        // Parse the response body
        let response_text = response.text().await.map_err(Error::Request)?;
        let anthropic_response: AnthropicResponse =
            serde_json::from_str(&response_text).map_err(Error::Serialization)?;

        // Convert to our message format using the existing From implementation
        let message = Message::from(&anthropic_response);

        Ok(message)
    }
}

impl AnthropicProvider {
    fn id_for_model(model: Claude) -> &'static str {
        match model {
            Claude::Sonnet37 { .. } => "sonnet-3-7-latest",
            Claude::Sonnet35 {
                version: Sonnet35Version::V1,
            } => "sonnet-3-5-1",
            Claude::Sonnet35 {
                version: Sonnet35Version::V2,
            } => "sonnet-3-5-2",
            Claude::Opus3 => "opus-3",
            Claude::Haiku3 => "haiku-3",
            Claude::Haiku35 => "haiku-3-5",
        }
    }

    /// Creates a request payload from a Chat object
    ///
    /// This method converts the Chat's messages and settings into an Anthropic-specific
    /// format for the API request.
    fn create_request_payload(&self, chat: &Chat<Claude>) -> Result<AnthropicRequest> {
        // Convert system prompt if present
        let system = if !chat.system_prompt.is_empty() {
            Some(chat.system_prompt.clone())
        } else {
            None
        };

        // Convert messages
        let messages: Vec<AnthropicMessage> = chat
            .history
            .iter()
            .filter(|msg| msg.role != MessageRole::System) // Filter out system messages as they go in system field
            .map(AnthropicMessage::from)
            .collect();

        // Create the request
        let request = AnthropicRequest {
            model: Self::id_for_model(chat.model).to_string(),
            messages,
            system,
            max_tokens: Some(chat.max_output_tokens),
            temperature: None,
            top_p: None,
            top_k: None,
            tools: None, // We'll add tools support later
        };

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

/// Represents a request to the Anthropic API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnthropicRequest {
    /// The model to use for generation
    pub model: String,
    /// The messages to send to the model
    pub messages: Vec<AnthropicMessage>,
    /// The system prompt (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// The maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// The temperature (randomness) of the generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// The top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// The top-k sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// The tools available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
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

/// Convert from our Message to Anthropic's message format
impl From<&Message> for AnthropicMessage {
    fn from(msg: &Message) -> Self {
        let role = match msg.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Function => "user", // Map function to user since Anthropic doesn't have function role
            MessageRole::Tool => "user",     // Map tool to user for similar reasons
        }
        .to_string();

        let content = match &msg.content {
            Some(Content::Text(text)) => vec![AnthropicContentPart::text(text.clone())],
            Some(Content::Parts(parts)) => parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => AnthropicContentPart::text(text.clone()),
                    ContentPart::ImageUrl { image_url } => {
                        AnthropicContentPart::image(image_url.url.clone())
                    }
                })
                .collect(),
            None => vec![AnthropicContentPart::text("".to_string())],
        };

        AnthropicMessage { role, content }
    }
}

/// Convert from Anthropic's response content to our content part
impl From<&AnthropicResponseContent> for ContentPart {
    fn from(content: &AnthropicResponseContent) -> Self {
        match content {
            AnthropicResponseContent::Text { text } => ContentPart::text(text.clone()),
            AnthropicResponseContent::ToolUse { id, name, input } => {
                // For tool use, we'll create a text representation
                let text = format!(
                    "{{\"id\":\"{}\",\"name\":\"{}\",\"input\":{}}}",
                    id, name, input
                );
                ContentPart::text(text)
            }
        }
    }
}

/// Convert from Anthropic's response to our message format
impl From<&AnthropicResponse> for Message {
    fn from(response: &AnthropicResponse) -> Self {
        let role = match response.role.as_str() {
            "assistant" => MessageRole::Assistant,
            _ => MessageRole::User, // Default to user for unknown roles
        };

        let content = if response.content.is_empty() {
            None
        } else if response.content.len() == 1 {
            // If there's only one content part and it's text, use simple Text content
            match &response.content[0] {
                AnthropicResponseContent::Text { text } => Some(Content::Text(text.clone())),
                _ => {
                    // Otherwise, convert all parts
                    let parts = response
                        .content
                        .iter()
                        .map(|part| part.into())
                        .collect::<Vec<ContentPart>>();
                    Some(Content::Parts(parts))
                }
            }
        } else {
            // Multiple content parts
            let parts = response
                .content
                .iter()
                .map(|part| part.into())
                .collect::<Vec<ContentPart>>();
            Some(Content::Parts(parts))
        };

        let mut msg = Message {
            role,
            content,
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
            metadata: Default::default(),
        };

        // Add token usage info as metadata
        msg = msg.with_metadata(
            "input_tokens",
            serde_json::Value::Number(response.usage.input_tokens.into()),
        );
        msg = msg.with_metadata(
            "output_tokens",
            serde_json::Value::Number(response.usage.output_tokens.into()),
        );

        msg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ImageUrl;
    use crate::message::{Content, ContentPart, Message, MessageRole};

    #[test]
    fn test_message_to_anthropic_conversion() {
        // Test simple text message
        let msg = Message::user("Hello, world!");
        let anthropic_msg = AnthropicMessage::from(&msg);

        assert_eq!(anthropic_msg.role, "user");
        assert_eq!(anthropic_msg.content.len(), 1);
        match &anthropic_msg.content[0] {
            AnthropicContentPart::Text { text } => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected text content"),
        }

        // Test multipart message
        let parts = vec![ContentPart::text("Hello"), ContentPart::text("world")];
        let msg = Message::user("").with_content_parts(parts);
        let anthropic_msg = AnthropicMessage::from(&msg);

        assert_eq!(anthropic_msg.role, "user");
        assert_eq!(anthropic_msg.content.len(), 2);

        // Test system message
        let msg = Message::system("You are a helpful assistant.");
        let anthropic_msg = AnthropicMessage::from(&msg);

        assert_eq!(anthropic_msg.role, "system");
        assert_eq!(anthropic_msg.content.len(), 1);

        // Test image content
        let image_url = ImageUrl::new("https://example.com/image.jpg");
        let parts = vec![
            ContentPart::text("Look at this image:"),
            ContentPart::image_url(image_url),
        ];
        let msg = Message::user("").with_content_parts(parts);
        let anthropic_msg = AnthropicMessage::from(&msg);

        assert_eq!(anthropic_msg.role, "user");
        assert_eq!(anthropic_msg.content.len(), 2);

        // Verify the image content
        match &anthropic_msg.content[1] {
            AnthropicContentPart::Image { source } => {
                assert_eq!(source.data, "https://example.com/image.jpg");
                assert_eq!(source.type_field, "base64");
                assert_eq!(source.media_type, "image/jpeg");
            }
            _ => panic!("Expected image content"),
        }
    }

    #[test]
    fn test_anthropic_response_to_message() {
        // Test single text content
        let response = AnthropicResponse {
            id: "msg_123".to_string(),
            type_field: "message".to_string(),
            role: "assistant".to_string(),
            model: "claude-3-opus-20240229".to_string(),
            stop_reason: Some("end_turn".to_string()),
            content: vec![AnthropicResponseContent::Text {
                text: "I'm Claude, an AI assistant.".to_string(),
            }],
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 20,
            },
        };

        let msg = Message::from(&response);

        assert_eq!(msg.role, MessageRole::Assistant);
        match &msg.content {
            Some(Content::Text(text)) => assert_eq!(text, "I'm Claude, an AI assistant."),
            _ => panic!("Expected text content"),
        }

        // Check metadata
        assert_eq!(msg.metadata["input_tokens"], 10);
        assert_eq!(msg.metadata["output_tokens"], 20);

        // Test multiple content parts
        let response = AnthropicResponse {
            id: "msg_123".to_string(),
            type_field: "message".to_string(),
            role: "assistant".to_string(),
            model: "claude-3-opus-20240229".to_string(),
            stop_reason: Some("end_turn".to_string()),
            content: vec![
                AnthropicResponseContent::Text {
                    text: "Here's the information:".to_string(),
                },
                AnthropicResponseContent::ToolUse {
                    id: "tool_123".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({
                        "location": "San Francisco",
                    }),
                },
            ],
            usage: AnthropicUsage {
                input_tokens: 15,
                output_tokens: 30,
            },
        };

        let msg = Message::from(&response);

        assert_eq!(msg.role, MessageRole::Assistant);
        match &msg.content {
            Some(Content::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                match &parts[0] {
                    ContentPart::Text { text } => assert_eq!(text, "Here's the information:"),
                    _ => panic!("Expected text content"),
                }
            }
            _ => panic!("Expected parts content"),
        }
    }

    #[test]
    fn test_manual_message_conversion() {
        // Test that we can directly convert between types without relying on Chat

        // Create a message
        let text_msg = Message::user("Hello, world!");
        let anthropic_msg = AnthropicMessage::from(&text_msg);

        // Verify the conversion
        assert_eq!(anthropic_msg.role, "user");
        assert_eq!(anthropic_msg.content.len(), 1);
        match &anthropic_msg.content[0] {
            AnthropicContentPart::Text { text } => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected text content"),
        }

        // Create a manual Anthropic response
        let response = AnthropicResponse {
            id: "msg_123".to_string(),
            type_field: "message".to_string(),
            role: "assistant".to_string(),
            model: "claude-3-opus-20240229".to_string(),
            stop_reason: Some("end_turn".to_string()),
            content: vec![AnthropicResponseContent::Text {
                text: "I'm here to help!".to_string(),
            }],
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 15,
            },
        };

        // Convert to our message type
        let lib_msg = Message::from(&response);

        // Verify the conversion
        assert_eq!(lib_msg.role, MessageRole::Assistant);
        match &lib_msg.content {
            Some(Content::Text(text)) => assert_eq!(text, "I'm here to help!"),
            _ => panic!("Expected text content"),
        }

        // Check metadata
        assert_eq!(lib_msg.metadata["input_tokens"], 5);
        assert_eq!(lib_msg.metadata["output_tokens"], 15);
    }
}
