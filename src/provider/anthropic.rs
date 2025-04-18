// curl https://api.anthropic.com/v1/messages \
//      --header "x-api-key: $ANTHROPIC_API_KEY" \
//      --header "anthropic-version: 2023-06-01" \
//      --header "content-type: application/json" \
//      --data \
// '{
//     "model": "claude-3-7-sonnet-20250219",
//     "max_tokens": 1024,
//     "tools": [
//         {
//             "name": "get_location",
//             "description": "Get the current user location based on their IP address. This tool has no parameters or arguments.",
//             "input_schema": {
//                 "type": "object",
//                 "properties": {}
//             }
//         },
//         {
//             "name": "get_weather",
//             "description": "Get the current weather in a given location",
//             "input_schema": {
//                 "type": "object",
//                 "properties": {
//                     "location": {
//                         "type": "string",
//                         "description": "The city and state, e.g. San Francisco, CA"
//                     },
//                     "unit": {
//                         "type": "string",
//                         "enum": ["celsius", "fahrenheit"],
//                         "description": "The unit of temperature, either 'celsius' or 'fahrenheit'"
//                     }
//                 },
//                 "required": ["location"]
//             }
//         }
//     ],
//     "messages": [{
//         "role": "user",
//         "content": "What is the weather like where I am?"
//     }]
// }'

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex};

use crate::client::{
    GenerationMetadata, GenerationOptions, GenerationResult, LlmProvider, ToolChoice, UsageInfo,
};
use crate::error::{Error, Result};
use crate::message::{Content, Message, MessageRole, ToolCall};
use crate::model::{AnthropicModel, Model, ModelCapability, ModelFamily, ModelInfo};

/// Configuration for the Anthropic provider
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the API
    pub base_url: String,
    /// API version header
    pub api_version: String,
    /// Default model to use
    pub default_model: String,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            api_version: "2023-06-01".to_string(),
            default_model: "claude-3-opus-20240229".to_string(),
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

impl AnthropicProvider {
    /// Creates a new Anthropic provider with the default configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::AnthropicProvider;
    ///
    /// let provider = AnthropicProvider::new();
    /// ```
    pub fn new() -> Self {
        Self::with_config(AnthropicConfig::default())
    }

    /// Creates a new Anthropic provider with a custom configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::{AnthropicProvider, AnthropicConfig};
    ///
    /// let config = AnthropicConfig {
    ///     api_key: "your_api_key".to_string(),
    ///     base_url: "https://api.anthropic.com/v1".to_string(),
    ///     api_version: "2023-06-01".to_string(),
    ///     default_model: "claude-3-opus-20240229".to_string(),
    /// };
    ///
    /// let provider = AnthropicProvider::with_config(config);
    /// ```
    pub fn with_config(config: AnthropicConfig) -> Self {
        Self {
            config,
        }
    }

    /// Creates a new Anthropic provider with an API key
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::AnthropicProvider;
    ///
    /// let provider = AnthropicProvider::with_api_key("your_api_key");
    /// ```
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let config = AnthropicConfig {
            api_key,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Returns the API key for this provider
    pub fn api_key(&self) -> &str {
        &self.config.api_key
    }
    
    /// Returns the base URL for this provider
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }
    
    /// Returns the API version for this provider
    pub fn api_version(&self) -> &str {
        &self.config.api_version
    }
    
    /// Helper method to convert our message format to Anthropic's message format
    pub fn convert_messages(&self, messages: &[Message]) -> Vec<AnthropicMessage> {
        let mut anthropic_messages = Vec::new();

        // Process messages in order
        for message in messages {
            match message.role {
                MessageRole::System => {
                    // System messages in Anthropic are not part of the messages array
                    // Instead, they are passed as a separate 'system' parameter
                    // We handle those separately in extract_system_message
                }
                MessageRole::User => {
                    let anthropic_content = if let Some(content) = &message.content {
                        match content {
                            Content::Text(text) => vec![AnthropicContentPart::text(text.clone())],
                            Content::Parts(parts) => {
                                // Convert our content parts to Anthropic's format
                                parts
                                    .iter()
                                    .map(|part| match part {
                                        crate::message::ContentPart::Text { text } => {
                                            AnthropicContentPart::text(text.clone())
                                        }
                                        crate::message::ContentPart::ImageUrl { image_url } => {
                                            AnthropicContentPart::image(image_url.url.clone())
                                        }
                                    })
                                    .collect()
                            }
                        }
                    } else {
                        // Empty content
                        vec![AnthropicContentPart::text("".to_string())]
                    };

                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: anthropic_content,
                    });
                }
                MessageRole::Assistant => {
                    // For assistant messages with text content
                    if let Some(Content::Text(text)) = &message.content {
                        anthropic_messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: vec![AnthropicContentPart::text(text.clone())],
                        });
                    }
                    // For assistant messages with tool calls
                    else if let Some(tool_calls) = &message.tool_calls {
                        // For Anthropic, tool calls are part of the message content
                        // Serialize tool calls to a text representation
                        let text = format!("Tool calls: {:?}", tool_calls);
                        anthropic_messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: vec![AnthropicContentPart::text(text)],
                        });
                    }
                }
                MessageRole::Tool => {
                    // Anthropic expects tool responses to be in the format of a user message
                    // with a specific tool response format
                    if let Some(Content::Text(text)) = &message.content {
                        if let Some(tool_call_id) = &message.tool_call_id {
                            let tool_response = AnthropicToolResponse {
                                type_field: "tool_result".to_string(),
                                tool_call_id: tool_call_id.clone(),
                                content: text.clone(),
                            };

                            anthropic_messages.push(AnthropicMessage {
                                role: "user".to_string(),
                                content: vec![AnthropicContentPart::ToolResult(tool_response)],
                            });
                        }
                    }
                }
                // Anthropic doesn't have a direct equivalent for function messages
                // We'll convert them to tool messages for compatibility
                MessageRole::Function => {
                    if let Some(Content::Text(text)) = &message.content {
                        if let Some(name) = &message.name {
                            let tool_response = AnthropicToolResponse {
                                type_field: "tool_result".to_string(),
                                tool_call_id: format!("function-{}", name), // Create a synthetic ID
                                content: text.clone(),
                            };

                            anthropic_messages.push(AnthropicMessage {
                                role: "user".to_string(),
                                content: vec![AnthropicContentPart::ToolResult(tool_response)],
                            });
                        }
                    }
                }
            }
        }

        // If there are no messages after conversion, add a default user message
        if anthropic_messages.is_empty() {
            anthropic_messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: vec![AnthropicContentPart::text("Hello".to_string())],
            });
        }

        // Ensure the last message is from the user for Anthropic's requirements
        if let Some(last) = anthropic_messages.last() {
            if last.role != "user" {
                anthropic_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![AnthropicContentPart::text("Please continue".to_string())],
                });
            }
        }

        anthropic_messages
    }

    /// Helper method to extract system message from the messages array
    pub fn extract_system_message(&self, messages: &[Message]) -> Option<String> {
        for message in messages {
            if message.role == MessageRole::System {
                if let Some(Content::Text(text)) = &message.content {
                    return Some(text.clone());
                }
            }
        }
        None
    }

    /// Convert ToolChoice to Anthropic's tool_choice format
    pub fn convert_tool_choice(&self, tool_choice: &ToolChoice) -> serde_json::Value {
        match tool_choice {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::None => serde_json::json!("none"),
            ToolChoice::Tool(name) => serde_json::json!({
                "type": "tool",
                "name": name
            }),
            ToolChoice::Any => serde_json::json!("any"),
        }
    }
    
    /// Convert generated response JSON to our internal format
    pub fn parse_response(&self, response_json: serde_json::Value) -> Result<GenerationResult> {
        let response: AnthropicResponse = serde_json::from_value(response_json)
            .map_err(Error::Serialization)?;
        
        // Create our message from the response
        let mut message = Message {
            role: MessageRole::Assistant,
            content: None,
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
            metadata: HashMap::new(),
        };

        // Process content from the response
        if let Some(content) = response.content.first() {
            match content {
                AnthropicResponseContent::Text { text, .. } => {
                    message.content = Some(Content::Text(text.clone()));
                }
                AnthropicResponseContent::ToolUse {
                    name, input, id, ..
                } => {
                    // Create a ToolCall for our format
                    let tool_call = ToolCall {
                        id: id.clone(),
                        tool_type: "function".to_string(),
                        function: crate::message::FunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        },
                    };

                    message.tool_calls = Some(vec![tool_call]);
                    // For tool calls, content is typically not included
                    message.content = None;
                }
            }
        }

        // Create generation metadata
        let metadata = GenerationMetadata {
            id: response.id,
            model: response.model,
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            usage: Some(UsageInfo {
                prompt_tokens: response.usage.input_tokens,
                completion_tokens: response.usage.output_tokens,
                total_tokens: response.usage.input_tokens
                    + response.usage.output_tokens,
            }),
            extra: HashMap::new(),
        };

        Ok(GenerationResult { message, metadata })
    }
    
    /// Accept a visitor to perform transport operations
    pub async fn accept<V: crate::transport::AnthropicTransportVisitor>(
        &self,
        visitor: &V,
        model: &impl ModelInfo,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        // First check if the model is supported
        if model.family() != ModelFamily::Claude {
            return Err(Error::UnsupportedModel(format!(
                "Model family '{}' not supported by Anthropic provider",
                model.family()
            )));
        }

        // Prepare request with the visitor
        let request = visitor.prepare_anthropic_request(model, messages, &options).await?;
        
        // Prepare headers
        let mut headers = HashMap::new();
        headers.insert("x-api-key".to_string(), self.config.api_key.clone());
        headers.insert("anthropic-version".to_string(), self.config.api_version.clone());
        headers.insert("content-type".to_string(), "application/json".to_string());

        // Process the request using the visitor
        let endpoint = format!("{}/messages", self.config.base_url);
        let response = visitor.process_request(request, &endpoint, headers).await?;
        
        // Process the response with the visitor
        let processed_response = visitor.process_anthropic_response(response).await?;
        
        // Parse the response
        self.parse_response(processed_response)
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn generate(
        &self,
        model: &str,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        // Create an HTTP transport
        let transport = crate::transport::http::HttpTransport::new();
        
        // Find the matching AnthropicModel enum or use a generic approach
        let model_result = match model {
            "claude-3-opus-20240229" => self.accept(&transport, &AnthropicModel::Opus3, messages, options).await,
            "claude-3-sonnet-20240229" => self.accept(&transport, &AnthropicModel::Sonnet3, messages, options).await,
            "claude-3.5-sonnet-20240620" => self.accept(&transport, &AnthropicModel::Sonnet35, messages, options).await,
            "claude-3-7-sonnet-20250219" => self.accept(&transport, &AnthropicModel::Sonnet37, messages, options).await,
            "claude-3-haiku-20240307" => self.accept(&transport, &AnthropicModel::Haiku3, messages, options).await,
            "claude-3.5-haiku-20240307" => self.accept(&transport, &AnthropicModel::Haiku35, messages, options).await,
            "claude-2.1" => self.accept(&transport, &AnthropicModel::Claude21, messages, options).await,
            "claude-2.0" => self.accept(&transport, &AnthropicModel::Claude20, messages, options).await,
            "claude-instant-1.2" => self.accept(&transport, &AnthropicModel::ClaudeInstant12, messages, options).await,
            _ => {
                // Generic fallback for any other model ID
                // Create a temporary ModelInfo implementation for the string model ID
                #[derive(Debug)]
                struct StringModel(String);
                impl ModelInfo for StringModel {
                    fn model_id(&self) -> String { self.0.clone() }
                    fn name(&self) -> String { self.0.clone() }
                    fn family(&self) -> ModelFamily { ModelFamily::Claude }
                    fn capabilities(&self) -> Vec<ModelCapability> { 
                        if self.0.starts_with("claude-3") {
                            vec![
                                ModelCapability::ChatCompletion,
                                ModelCapability::TextGeneration,
                                ModelCapability::Vision,
                                ModelCapability::ToolCalling,
                            ]
                        } else {
                            vec![
                                ModelCapability::ChatCompletion,
                                ModelCapability::TextGeneration,
                            ]
                        }
                    }
                    fn context_window(&self) -> usize { 
                        if self.0.contains("instant") || self.0 == "claude-2.0" {
                            100_000
                        } else {
                            200_000
                        }
                    }
                }
                
                self.accept(&transport, &StringModel(model.to_string()), messages, options).await
            }
        };
        
        model_result
    }

    async fn supports_tool_calling(&self, model: &str) -> Result<bool> {
        // Claude 3 models support tool calling
        Ok(model.starts_with("claude-3"))
    }

    async fn has_capability(&self, model: &str, capability: ModelCapability) -> Result<bool> {
        // Convert the string model ID to a ModelInfo instance
        let model_info = match model {
            "claude-3-opus-20240229" => AnthropicModel::Opus3,
            "claude-3-sonnet-20240229" => AnthropicModel::Sonnet3,
            "claude-3.5-sonnet-20240620" => AnthropicModel::Sonnet35,
            "claude-3-7-sonnet-20250219" => AnthropicModel::Sonnet37,
            "claude-3-haiku-20240307" => AnthropicModel::Haiku3,
            "claude-3.5-haiku-20240307" => AnthropicModel::Haiku35,
            "claude-2.1" => AnthropicModel::Claude21,
            "claude-2.0" => AnthropicModel::Claude20,
            "claude-instant-1.2" => AnthropicModel::ClaudeInstant12,
            // For unknown models, infer capabilities based on the model name
            _ if model.starts_with("claude-3") => {
                return Ok(match capability {
                    ModelCapability::ChatCompletion | 
                    ModelCapability::TextGeneration | 
                    ModelCapability::Vision |
                    ModelCapability::ToolCalling => true,
                    _ => false,
                })
            }
            _ => {
                return Ok(match capability {
                    ModelCapability::ChatCompletion | 
                    ModelCapability::TextGeneration => true,
                    _ => false,
                })
            }
        };
        
        Ok(model_info.has_capability(capability))
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
mod tests {
    use super::*;
    use crate::message::MessageRole;
    use serde_json::json;

    // We don't need the mock response function for our current tests
    // Let's remove it to simplify things

    #[tokio::test]
    async fn test_model_capabilities() {
        let provider = AnthropicProvider::new();
        
        // Check models have the right capabilities
        assert!(provider.has_capability("claude-3-opus-20240229", ModelCapability::ChatCompletion).await.unwrap());
        assert!(provider.has_capability("claude-3-opus-20240229", ModelCapability::Vision).await.unwrap());
        assert!(provider.has_capability("claude-3-opus-20240229", ModelCapability::ToolCalling).await.unwrap());

        // Make sure Claude 2 doesn't have tool calling
        assert!(!provider.has_capability("claude-2.1", ModelCapability::ToolCalling).await.unwrap());
        assert!(provider.has_capability("claude-2.1", ModelCapability::ChatCompletion).await.unwrap());
    }

    #[tokio::test]
    async fn test_convert_messages() {
        let provider = AnthropicProvider::new();

        // Test basic message conversion
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello, who are you?"),
        ];

        let anthropic_messages = provider.convert_messages(&messages);

        // Should have one user message (system is handled separately)
        assert_eq!(anthropic_messages.len(), 1);
        assert_eq!(anthropic_messages[0].role, "user");

        if let AnthropicContentPart::Text { text } = &anthropic_messages[0].content[0] {
            assert_eq!(text, "Hello, who are you?");
        } else {
            panic!("Expected text content");
        }

        // Test conversion with assistant message
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello, who are you?"),
            Message::assistant("I am Claude, an AI assistant."),
        ];

        let anthropic_messages = provider.convert_messages(&messages);

        // Should have user and assistant messages, plus a final "Please continue" user message
        // Anthropic requires the last message to be from the user
        assert_eq!(anthropic_messages.len(), 3);
        assert_eq!(anthropic_messages[0].role, "user");
        assert_eq!(anthropic_messages[1].role, "assistant");
        assert_eq!(anthropic_messages[2].role, "user");

        // Test that tool messages are properly converted
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            tool_type: "function".to_string(),
            function: crate::message::FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"New York"}"#.to_string(),
            },
        };

        let messages = vec![
            Message::user("What's the weather in New York?"),
            Message {
                role: MessageRole::Assistant,
                content: None,
                name: None,
                function_call: None,
                tool_calls: Some(vec![tool_call]),
                tool_call_id: None,
                metadata: HashMap::new(),
            },
            Message::tool("call_123", r#"{"temperature": 72, "condition": "sunny"}"#),
        ];

        let anthropic_messages = provider.convert_messages(&messages);

        // Should have user, assistant, and user (tool response) messages
        assert_eq!(anthropic_messages.len(), 3);
        assert_eq!(anthropic_messages[0].role, "user");
        assert_eq!(anthropic_messages[1].role, "assistant");
        assert_eq!(anthropic_messages[2].role, "user");

        // Last message should be a tool result
        match &anthropic_messages[2].content[0] {
            AnthropicContentPart::ToolResult(tool_result) => {
                assert_eq!(tool_result.tool_call_id, "call_123");
                assert_eq!(
                    tool_result.content,
                    r#"{"temperature": 72, "condition": "sunny"}"#
                );
            }
            _ => panic!("Expected tool result content"),
        }
    }

    #[tokio::test]
    async fn test_supports_tool_calling() {
        let provider = AnthropicProvider::new();

        // Claude 3 models support tool calling
        assert!(provider
            .supports_tool_calling("claude-3-opus-20240229")
            .await
            .unwrap());
        assert!(provider
            .supports_tool_calling("claude-3-sonnet-20240229")
            .await
            .unwrap());
        assert!(provider
            .supports_tool_calling("claude-3-haiku-20240307")
            .await
            .unwrap());

        // Older models don't
        assert!(!provider.supports_tool_calling("claude-2.1").await.unwrap());
    }

    // #[tokio::test]
    // async fn test_register_and_execute_tool() {
    //     // Test commented out as tool functionality has been temporarily removed
    // }
}
