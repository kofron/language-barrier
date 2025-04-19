use crate::error::{Error, Result};
use crate::message::{Content, ContentPart, Message};
use crate::model::Sonnet35Version;
use crate::provider::HTTPProvider;
use crate::{Chat, Claude};
use reqwest::{Method, Request, Url};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use tracing::{debug, error, info, instrument, trace, warn};

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
    #[instrument(level = "debug")]
    pub fn new() -> Self {
        info!("Creating new AnthropicProvider with default configuration");
        let config = AnthropicConfig::default();
        debug!("API key set: {}", !config.api_key.is_empty());
        debug!("Base URL: {}", config.base_url);
        debug!("API version: {}", config.api_version);

        Self { config }
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
    #[instrument(skip(config), level = "debug")]
    pub fn with_config(config: AnthropicConfig) -> Self {
        info!("Creating new AnthropicProvider with custom configuration");
        debug!("API key set: {}", !config.api_key.is_empty());
        debug!("Base URL: {}", config.base_url);
        debug!("API version: {}", config.api_version);

        Self { config }
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl HTTPProvider<Claude> for AnthropicProvider {
    fn accept(&self, chat: Chat<Claude>) -> Result<Request> {
        info!("Creating request for Claude model: {:?}", chat.model);
        debug!("Messages in chat history: {}", chat.history.len());

        let url_str = format!("{}/messages", self.config.base_url);
        debug!("Parsing URL: {}", url_str);
        let url = match Url::parse(&url_str) {
            Ok(url) => {
                debug!("URL parsed successfully: {}", url);
                url
            }
            Err(e) => {
                error!("Failed to parse URL '{}': {}", url_str, e);
                return Err(e.into());
            }
        };

        let mut request = Request::new(Method::POST, url);
        debug!("Created request: {} {}", request.method(), request.url());

        // Set headers
        debug!("Setting request headers");
        let api_key_header = match self.config.api_key.parse() {
            Ok(header) => header,
            Err(e) => {
                error!("Invalid API key format: {}", e);
                return Err(Error::Authentication("Invalid API key format".into()));
            }
        };

        let content_type_header = match "application/json".parse() {
            Ok(header) => header,
            Err(e) => {
                error!("Failed to set content type: {}", e);
                return Err(Error::Other("Failed to set content type".into()));
            }
        };

        let api_version_header = match self.config.api_version.parse() {
            Ok(header) => header,
            Err(e) => {
                error!("Invalid API version format: {}", e);
                return Err(Error::Other("Invalid API version format".into()));
            }
        };

        request.headers_mut().insert("x-api-key", api_key_header);
        request
            .headers_mut()
            .insert("Content-Type", content_type_header);
        request
            .headers_mut()
            .insert("anthropic-version", api_version_header);

        trace!("Request headers set: {:#?}", request.headers());

        // Create the request payload
        debug!("Creating request payload");
        let payload = match self.create_request_payload(&chat) {
            Ok(payload) => {
                debug!("Request payload created successfully");
                trace!("Model: {}", payload.model);
                trace!("Max tokens: {:?}", payload.max_tokens);
                trace!("System prompt present: {}", payload.system.is_some());
                trace!("Number of messages: {}", payload.messages.len());
                payload
            }
            Err(e) => {
                error!("Failed to create request payload: {}", e);
                return Err(e);
            }
        };

        // Set the request body
        debug!("Serializing request payload");
        let body_bytes = match serde_json::to_vec(&payload) {
            Ok(bytes) => {
                debug!("Payload serialized successfully ({} bytes)", bytes.len());
                bytes
            }
            Err(e) => {
                error!("Failed to serialize payload: {}", e);
                return Err(Error::Serialization(e));
            }
        };

        *request.body_mut() = Some(body_bytes.into());
        info!("Request created successfully");

        Ok(request)
    }

    fn parse(&self, raw_response_text: String) -> Result<Message> {
        info!("Parsing response from Anthropic API");
        trace!("Raw response: {}", raw_response_text);

        // First check if it's an error response
        if raw_response_text.contains("\"error\"") {
            let error_response: serde_json::Value = match serde_json::from_str(&raw_response_text) {
                Ok(err) => err,
                Err(e) => {
                    error!("Failed to parse error response: {}", e);
                    error!("Raw response: {}", raw_response_text);
                    return Err(Error::Serialization(e));
                }
            };
            
            if let Some(error) = error_response.get("error") {
                if let Some(message) = error.get("message") {
                    let error_message = message.as_str().unwrap_or("Unknown error");
                    error!("Anthropic API returned an error: {}", error_message);
                    return Err(Error::ProviderUnavailable(error_message.to_string()));
                }
            }
            
            error!("Unknown error format in response: {}", raw_response_text);
            return Err(Error::ProviderUnavailable("Unknown error from Anthropic API".to_string()));
        }

        debug!("Deserializing response JSON");
        let anthropic_response = match serde_json::from_str::<AnthropicResponse>(&raw_response_text)
        {
            Ok(response) => {
                debug!("Response deserialized successfully");
                debug!("Response ID: {}", response.id);
                debug!("Response model: {}", response.model);
                debug!("Stop reason: {:?}", response.stop_reason);
                debug!("Content parts: {}", response.content.len());
                debug!("Input tokens: {}", response.usage.input_tokens);
                debug!("Output tokens: {}", response.usage.output_tokens);
                response
            }
            Err(e) => {
                error!("Failed to deserialize response: {}", e);
                error!("Raw response: {}", raw_response_text);
                return Err(Error::Serialization(e));
            }
        };

        // Convert to our message format using the existing From implementation
        debug!("Converting Anthropic response to Message");
        let message = Message::from(&anthropic_response);

        info!("Response parsed successfully");
        trace!("Response message processed");

        Ok(message)
    }
}

impl AnthropicProvider {
    #[instrument(level = "debug")]
    fn id_for_model(model: Claude) -> &'static str {
        let model_id = match model {
            Claude::Sonnet37 { .. } => "claude-3-7-sonnet-latest",
            Claude::Sonnet35 {
                version: Sonnet35Version::V1,
            } => "claude-3-5-sonnet-20240620",
            Claude::Sonnet35 {
                version: Sonnet35Version::V2,
            } => "claude-3-5-sonnet-20241022",
            Claude::Opus3 => "claude-3-opus-latest",
            Claude::Haiku3 => "claude-3-haiku-20240307",
            Claude::Haiku35 => "claude-3-5-haiku-latest",
        };

        debug!("Mapped Claude model to Anthropic model ID: {}", model_id);
        model_id
    }

    /// Creates a request payload from a Chat object
    ///
    /// This method converts the Chat's messages and settings into an Anthropic-specific
    /// format for the API request.
    #[instrument(skip(self, chat), level = "debug")]
    fn create_request_payload(&self, chat: &Chat<Claude>) -> Result<AnthropicRequest> {
        info!("Creating request payload for chat with Claude model");
        debug!("System prompt length: {}", chat.system_prompt.len());
        debug!("Messages in history: {}", chat.history.len());
        debug!("Max output tokens: {}", chat.max_output_tokens);

        // Convert system prompt if present
        let system = if chat.system_prompt.is_empty() {
            debug!("No system prompt provided");
            None
        } else {
            debug!("Including system prompt in request");
            trace!("System prompt: {}", chat.system_prompt);
            Some(chat.system_prompt.clone())
        };

        // Convert messages
        debug!("Converting messages to Anthropic format");
        let messages: Vec<AnthropicMessage> = chat
            .history
            .iter()
            .filter(|msg| !matches!(msg, Message::System { .. })) // Filter out system messages as they go in system field
            .map(|msg| {
                trace!("Converting message with role: {}", msg.role_str());
                AnthropicMessage::from(msg)
            })
            .collect();

        debug!("Converted {} messages for the request", messages.len());

        // Get model ID for the chat model
        let model_id = Self::id_for_model(chat.model).to_string();
        debug!("Using model ID: {}", model_id);
        
        // Convert tool descriptions if a toolbox is provided
        let tools = if chat.has_toolbox() {
            let tool_descriptions = chat.tool_descriptions();
            debug!("Converting {} tool descriptions to Anthropic format", tool_descriptions.len());
            
            if tool_descriptions.is_empty() {
                None
            } else {
                Some(
                    tool_descriptions
                        .into_iter()
                        .map(|desc| AnthropicTool {
                            name: desc.name,
                            description: desc.description,
                            input_schema: desc.parameters,
                        })
                        .collect()
                )
            }
        } else {
            debug!("No toolbox provided");
            None
        };

        // Create the request
        debug!("Creating AnthropicRequest");
        let request = AnthropicRequest {
            model: model_id,
            messages,
            system,
            max_tokens: Some(chat.max_output_tokens),
            temperature: None,
            top_p: None,
            top_k: None,
            tools,
        };

        info!("Request payload created successfully");
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
    /// The ID of the tool call (Anthropic API uses `tool_use_id`, but we use `tool_call_id` internally)
    #[serde(rename = "tool_use_id")]
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
        let role = match msg {
            Message::System { .. } => "system",
            Message::User { .. } | Message::Tool { .. } => "user", // API requires tool_result blocks to be in user messages
            Message::Assistant { .. } => "assistant",
        }
        .to_string();

        let content = match msg {
            Message::System { content, .. } => vec![AnthropicContentPart::text(content.clone())],
            Message::User { content, .. } => match content {
                Content::Text(text) => vec![AnthropicContentPart::text(text.clone())],
                Content::Parts(parts) => parts
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text { text } => AnthropicContentPart::text(text.clone()),
                        ContentPart::ImageUrl { image_url } => {
                            AnthropicContentPart::image(image_url.url.clone())
                        }
                    })
                    .collect(),
            },
            Message::Assistant { content, .. } => match content {
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
                None => vec![AnthropicContentPart::text(String::new())],
            },
            Message::Tool { tool_call_id, content, .. } => {
                // For tool messages, add a tool_result part
                vec![AnthropicContentPart::ToolResult(AnthropicToolResponse {
                    type_field: "tool_result".to_string(),
                    tool_call_id: tool_call_id.clone(),
                    content: content.clone(),
                })]
            }
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
                let text = format!("{{\"id\":\"{id}\",\"name\":\"{name}\",\"input\":{input}}}");
                ContentPart::text(text)
            }
        }
    }
}

/// Convert from Anthropic's response to our message format
impl From<&AnthropicResponse> for Message {
    fn from(response: &AnthropicResponse) -> Self {
        // Extract text content and tool calls from response content
        let mut text_content = Vec::new();
        let mut tool_calls = Vec::new();

        for content_part in &response.content {
            match content_part {
                AnthropicResponseContent::Text { text } => {
                    text_content.push(ContentPart::text(text.clone()));
                }
                AnthropicResponseContent::ToolUse { id, name, input } => {
                    // Convert tool use to our ToolCall format
                    let function = crate::message::Function {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    };

                    let tool_call = crate::message::ToolCall {
                        id: id.clone(),
                        tool_type: "function".to_string(),
                        function,
                    };

                    tool_calls.push(tool_call);
                }
            }
        }

        // Create content based on what we found
        let content = if text_content.is_empty() {
            None
        } else if text_content.len() == 1 {
            match &text_content[0] {
                ContentPart::Text { text } => Some(Content::Text(text.clone())),
                ContentPart::ImageUrl { .. } => Some(Content::Parts(text_content)),
            }
        } else {
            Some(Content::Parts(text_content))
        };

        // Create the message based on response.role
        let mut msg = match response.role.as_str() {
            "assistant" => {
                if tool_calls.is_empty() {
                    let content_to_use = content.unwrap_or(Content::Text(String::new()));
                    match content_to_use {
                        Content::Text(text) => Message::assistant(text),
                        Content::Parts(_) => Message::Assistant {
                            content: Some(content_to_use),
                            tool_calls: Vec::new(),
                            metadata: HashMap::default(),
                        },
                    }
                } else {
                    Message::Assistant {
                        content,
                        tool_calls,
                        metadata: HashMap::default(),
                    }
                }
            }
            // Default to user for unknown roles
            _ => match content {
                Some(Content::Text(text)) => Message::user(text),
                Some(Content::Parts(parts)) => Message::User {
                    content: Content::Parts(parts),
                    name: None,
                    metadata: HashMap::default(),
                },
                None => Message::user(""),
            },
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
    
    use crate::message::{Content, ContentPart, Message};

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
        let msg = Message::user_with_parts(parts);
        let anthropic_msg = AnthropicMessage::from(&msg);

        assert_eq!(anthropic_msg.role, "user");
        assert_eq!(anthropic_msg.content.len(), 2);

        // Test system message
        let msg = Message::system("You are a helpful assistant.");
        let anthropic_msg = AnthropicMessage::from(&msg);

        assert_eq!(anthropic_msg.role, "system");
        assert_eq!(anthropic_msg.content.len(), 1);

        // Test image content
        let parts = vec![
            ContentPart::text("Look at this image:"),
            ContentPart::image_url("https://example.com/image.jpg"),
        ];
        let msg = Message::user_with_parts(parts);
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
        
        // Test tool message
        let msg = Message::tool("tool_call_123", "The weather is sunny.");
        let anthropic_msg = AnthropicMessage::from(&msg);
        
        assert_eq!(anthropic_msg.role, "user");
        assert_eq!(anthropic_msg.content.len(), 1);
        
        // Verify tool content
        match &anthropic_msg.content[0] {
            AnthropicContentPart::ToolResult(tool_result) => {
                assert_eq!(tool_result.type_field, "tool_result");
                assert_eq!(tool_result.tool_call_id, "tool_call_123");
                assert_eq!(tool_result.content, "The weather is sunny.");
            }
            _ => panic!("Expected tool result content"),
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

        // Check the message is an Assistant variant
        match &msg {
            Message::Assistant { content, tool_calls, .. } => {
                // Verify content
                match content {
                    Some(Content::Text(text)) => assert_eq!(text, "I'm Claude, an AI assistant."),
                    _ => panic!("Expected text content"),
                }
                // Verify no tool calls
                assert!(tool_calls.is_empty());
            }
            _ => panic!("Expected Assistant variant"),
        }

        // Check metadata
        match &msg {
            Message::Assistant { metadata, .. } => {
                assert_eq!(metadata["input_tokens"], 10);
                assert_eq!(metadata["output_tokens"], 20);
            }
            _ => panic!("Expected Assistant variant"),
        }

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

        // Check the message is an Assistant variant
        match &msg {
            Message::Assistant { content, tool_calls, .. } => {
                // Verify content
                match content {
                    Some(Content::Text(text)) => {
                        assert_eq!(text, "Here's the information:");
                    }
                    Some(Content::Parts(parts)) => {
                        assert_eq!(parts.len(), 1);
                        match &parts[0] {
                            ContentPart::Text { text } => assert_eq!(text, "Here's the information:"),
                            _ => panic!("Expected text content"),
                        }
                    }
                    _ => panic!("Expected content"),
                }
                
                // Verify tool calls
                assert_eq!(tool_calls.len(), 1);
                
                let tool_call = &tool_calls[0];
                assert_eq!(tool_call.id, "tool_123");
                assert_eq!(tool_call.tool_type, "function");
                assert_eq!(tool_call.function.name, "get_weather");
                
                // Parse the arguments JSON to verify it
                let arguments: serde_json::Value = serde_json::from_str(&tool_call.function.arguments).unwrap();
                assert_eq!(arguments["location"], "San Francisco");
            }
            _ => panic!("Expected Assistant variant"),
        }
    }
    
    #[test]
    fn test_anthropic_response_with_tool_use_only() {
        // Test response with only tool use (no text)
        let response = AnthropicResponse {
            id: "msg_123".to_string(),
            type_field: "message".to_string(),
            role: "assistant".to_string(),
            model: "claude-3-opus-20240229".to_string(),
            stop_reason: Some("end_turn".to_string()),
            content: vec![
                AnthropicResponseContent::ToolUse {
                    id: "tool_xyz".to_string(),
                    name: "calculate".to_string(),
                    input: serde_json::json!({
                        "expression": "2+2",
                    }),
                },
            ],
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 10,
            },
        };

        let msg = Message::from(&response);

        // Check the message is an Assistant variant
        match &msg {
            Message::Assistant { content, tool_calls, .. } => {
                // Content should be None since there's no text
                assert!(content.is_none() || 
                       (match content {
                           Some(Content::Parts(parts)) => parts.is_empty(),
                           _ => false,
                       }));
                
                // Verify tool calls
                assert_eq!(tool_calls.len(), 1);
                
                let tool_call = &tool_calls[0];
                assert_eq!(tool_call.id, "tool_xyz");
                assert_eq!(tool_call.function.name, "calculate");
                
                // Parse the arguments JSON to verify it
                let arguments: serde_json::Value = serde_json::from_str(&tool_call.function.arguments).unwrap();
                assert_eq!(arguments["expression"], "2+2");
            }
            _ => panic!("Expected Assistant variant"),
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
        match &lib_msg {
            Message::Assistant { content, tool_calls, metadata } => {
                // Verify content
                match content {
                    Some(Content::Text(text)) => assert_eq!(text, "I'm here to help!"),
                    _ => panic!("Expected text content"),
                }
                
                // Verify no tool calls
                assert!(tool_calls.is_empty());
                
                // Check metadata
                assert_eq!(metadata["input_tokens"], 5);
                assert_eq!(metadata["output_tokens"], 15);
            }
            _ => panic!("Expected Assistant variant"),
        }
    }

    #[test]
    fn test_headers() {
        // This test may fail due to transitional state in the codebase
        // We'll implement it properly but comment it out for now

        // Create a provider with a test API key
        let config = AnthropicConfig {
            api_key: "test-api-key".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            api_version: "2023-06-01".to_string(),
        };
        let provider = AnthropicProvider::with_config(config);

        // Create a chat with a model
        let model = Claude::Sonnet37 {
            use_extended_thinking: false,
        };
        let mut chat = Chat::new(model)
            .with_system_prompt("You are a helpful assistant.")
            .with_max_output_tokens(1024);

        // Add some messages
        chat.push_message(Message::user("Hello, how are you?"));
        chat.push_message(Message::assistant("I'm doing well, thank you for asking!"));
        chat.push_message(Message::user("Can you help me with a question?"));

        // Create the request
        let request = provider.accept(chat).unwrap();

        // Verify the request
        assert_eq!(request.method(), "POST");
        assert_eq!(
            request.url().as_str(),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(request.headers()["x-api-key"], "test-api-key");
        assert_eq!(request.headers()["anthropic-version"], "2023-06-01");
        assert_eq!(request.headers()["Content-Type"], "application/json");
    }
}
