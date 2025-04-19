use crate::error::{Error, Result};
use crate::message::{Content, ContentPart, Message};
use crate::provider::HTTPProvider;
use crate::{Chat, ModelInfo};
use reqwest::{Method, Request, Url};
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{debug, error, info, instrument, trace, warn};

/// Configuration for the Mistral provider
#[derive(Debug, Clone)]
pub struct MistralConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the API
    pub base_url: String,
}

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            api_key: env::var("MISTRAL_API_KEY").unwrap_or_default(),
            base_url: "https://api.mistral.ai/v1".to_string(),
        }
    }
}

/// Implementation of the Mistral provider
#[derive(Debug, Clone)]
pub struct MistralProvider {
    /// Configuration for the provider
    config: MistralConfig,
}

impl MistralProvider {
    /// Creates a new MistralProvider with default configuration
    ///
    /// This method will use the MISTRAL_API_KEY environment variable for authentication.
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier_core::provider::mistral::MistralProvider;
    ///
    /// let provider = MistralProvider::new();
    /// ```
    #[instrument(level = "debug")]
    pub fn new() -> Self {
        info!("Creating new MistralProvider with default configuration");
        let config = MistralConfig::default();
        debug!("API key set: {}", !config.api_key.is_empty());
        debug!("Base URL: {}", config.base_url);

        Self { config }
    }

    /// Creates a new MistralProvider with custom configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier_core::provider::mistral::{MistralProvider, MistralConfig};
    ///
    /// let config = MistralConfig {
    ///     api_key: "your-api-key".to_string(),
    ///     base_url: "https://api.mistral.ai/v1".to_string(),
    /// };
    ///
    /// let provider = MistralProvider::with_config(config);
    /// ```
    #[instrument(skip(config), level = "debug")]
    pub fn with_config(config: MistralConfig) -> Self {
        info!("Creating new MistralProvider with custom configuration");
        debug!("API key set: {}", !config.api_key.is_empty());
        debug!("Base URL: {}", config.base_url);

        Self { config }
    }
}

impl Default for MistralProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait to get Mistral-specific model IDs
pub trait MistralModelInfo {
    fn mistral_model_id(&self) -> String;
}

impl<M: ModelInfo + MistralModelInfo> HTTPProvider<M> for MistralProvider {
    fn accept(&self, chat: Chat<M>) -> Result<Request> {
        info!("Creating request for Mistral model: {:?}", chat.model);
        debug!("Messages in chat history: {}", chat.history.len());

        let url_str = format!("{}/chat/completions", self.config.base_url);
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

        // API key as bearer token
        let auth_header = match format!("Bearer {}", self.config.api_key).parse() {
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

        request.headers_mut().insert("Authorization", auth_header);
        request
            .headers_mut()
            .insert("Content-Type", content_type_header);

        trace!("Request headers set: {:#?}", request.headers());

        // Create the request payload
        debug!("Creating request payload");
        let payload = match self.create_request_payload(&chat) {
            Ok(payload) => {
                debug!("Request payload created successfully");
                trace!("Model: {}", payload.model);
                trace!("Max tokens: {:?}", payload.max_tokens);
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
        info!("Parsing response from Mistral API");
        trace!("Raw response: {}", raw_response_text);

        // First try to parse as an error response
        if let Ok(error_response) = serde_json::from_str::<MistralErrorResponse>(&raw_response_text)
        {
            if error_response.error.is_some() {
                let error = error_response.error.unwrap();
                error!("Mistral API returned an error: {}", error.message);
                return Err(Error::ProviderUnavailable(error.message));
            }
        }

        // If not an error, parse as a successful response
        debug!("Deserializing response JSON");
        let mistral_response = match serde_json::from_str::<MistralResponse>(&raw_response_text) {
            Ok(response) => {
                debug!("Response deserialized successfully");
                debug!("Response id: {}", response.id);
                debug!("Response model: {}", response.model);
                if !response.choices.is_empty() {
                    debug!("Number of choices: {}", response.choices.len());
                    debug!(
                        "First choice finish reason: {:?}",
                        response.choices[0].finish_reason
                    );
                }
                if let Some(usage) = &response.usage {
                    debug!(
                        "Token usage - prompt: {}, completion: {}, total: {}",
                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                    );
                }
                response
            }
            Err(e) => {
                error!("Failed to deserialize response: {}", e);
                error!("Raw response: {}", raw_response_text);
                return Err(Error::Serialization(e));
            }
        };

        // Convert to our message format
        debug!("Converting Mistral response to Message");
        let message = Message::from(&mistral_response);

        info!("Response parsed successfully");
        trace!("Response message processed");

        Ok(message)
    }
}

impl MistralProvider {
    /// Creates a request payload from a Chat object
    ///
    /// This method converts the Chat's messages and settings into a Mistral-specific
    /// format for the API request.
    #[instrument(skip(self, chat), level = "debug")]
    fn create_request_payload<M: ModelInfo + MistralModelInfo>(
        &self,
        chat: &Chat<M>,
    ) -> Result<MistralRequest> {
        info!("Creating request payload for chat with Mistral model");
        debug!("System prompt length: {}", chat.system_prompt.len());
        debug!("Messages in history: {}", chat.history.len());
        debug!("Max output tokens: {}", chat.max_output_tokens);

        let model_id = chat.model.mistral_model_id();
        debug!("Using model ID: {}", model_id);

        // Convert all messages including system prompt
        debug!("Converting messages to Mistral format");
        let mut messages: Vec<MistralMessage> = Vec::new();

        // Add system prompt if present
        if !chat.system_prompt.is_empty() {
            debug!("Adding system prompt");
            messages.push(MistralMessage {
                role: "system".to_string(),
                content: chat.system_prompt.clone(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Add conversation history
        for msg in &chat.history {
            debug!("Converting message with role: {}", msg.role_str());
            messages.push(MistralMessage::from(msg));
        }

        debug!("Converted {} messages for the request", messages.len());

        // Add tools if present
        let tools = if chat.has_toolbox() {
            let tool_descriptions = chat.tool_descriptions();
            debug!(
                "Converting {} tool descriptions to Mistral format",
                tool_descriptions.len()
            );

            if !tool_descriptions.is_empty() {
                Some(
                    tool_descriptions
                        .into_iter()
                        .map(|desc| MistralTool {
                            r#type: "function".to_string(),
                            function: MistralFunction {
                                name: desc.name,
                                description: desc.description,
                                parameters: desc.parameters,
                            },
                        })
                        .collect(),
                )
            } else {
                None
            }
        } else {
            debug!("No toolbox provided");
            None
        };

        // Create the tool choice setting
        let tool_choice = if tools.is_some() {
            Some("auto".to_string())
        } else {
            None
        };

        // Create the request
        debug!("Creating MistralRequest");
        let request = MistralRequest {
            model: model_id,
            messages,
            temperature: None,
            top_p: None,
            max_tokens: Some(chat.max_output_tokens),
            stream: None,
            random_seed: None,
            safe_prompt: None,
            tools,
            tool_choice,
        };

        info!("Request payload created successfully");
        Ok(request)
    }
}

/// Represents a message in the Mistral API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MistralMessage {
    /// The role of the message sender (system, user, assistant, etc.)
    pub role: String,
    /// The content of the message
    pub content: String,
    /// The name of the function
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<MistralToolCall>>,
    /// Tool call ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Represents a tool function in the Mistral API format
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MistralFunction {
    /// The name of the function
    pub name: String,
    /// The description of the function
    pub description: String,
    /// The parameters schema as a JSON object
    pub parameters: serde_json::Value,
}

/// Represents a tool in the Mistral API format
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MistralTool {
    /// The type of the tool (currently always "function")
    pub r#type: String,
    /// The function definition
    pub function: MistralFunction,
}

/// Represents a function call in the Mistral API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MistralFunctionCall {
    /// The name of the function
    pub name: String,
    /// The arguments as a JSON string
    pub arguments: String,
}

/// Represents a tool call in the Mistral API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MistralToolCall {
    /// The ID of the tool call
    pub id: String,
    /// The type of the tool (currently always "function")
    pub r#type: String,
    /// The function call
    pub function: MistralFunctionCall,
}

/// Represents a request to the Mistral API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MistralRequest {
    /// The model to use
    pub model: String,
    /// The messages to send
    pub messages: Vec<MistralMessage>,
    /// Temperature (randomness)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// Stream mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Random seed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
    /// Safe prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_prompt: Option<bool>,
    /// Tools available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<MistralTool>>,
    /// Tool choice strategy (auto, none, or a specific tool)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
}

/// Represents a response from the Mistral API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MistralResponse {
    /// Response ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model used
    pub model: String,
    /// Choices generated
    pub choices: Vec<MistralChoice>,
    /// Usage statistics
    pub usage: Option<MistralUsage>,
}

/// Represents a choice in a Mistral response
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MistralChoice {
    /// The index of the choice
    pub index: usize,
    /// The message generated
    pub message: MistralMessage,
    /// The reason generation stopped
    pub finish_reason: Option<String>,
}

/// Represents usage statistics in a Mistral response
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MistralUsage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total number of tokens
    pub total_tokens: u32,
}

/// Represents an error response from the Mistral API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MistralErrorResponse {
    /// The error details
    pub error: Option<MistralError>,
}

/// Represents an error from the Mistral API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MistralError {
    /// The error message
    pub message: String,
    /// The error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// The error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Convert from our Message to Mistral's message format
impl From<&Message> for MistralMessage {
    fn from(msg: &Message) -> Self {
        let role = match msg {
            Message::System { .. } => "system",
            Message::User { .. } => "user",
            Message::Assistant { .. } => "assistant",
            Message::Tool { .. } => "tool",
        }
        .to_string();

        let (content, name, tool_calls, tool_call_id) = match msg {
            Message::System { content, .. } => (content.clone(), None, None, None),
            Message::User { content, name, .. } => {
                let content_str = match content {
                    Content::Text(text) => text.clone(),
                    Content::Parts(parts) => {
                        // For now, we just concatenate all text parts
                        // A more complete implementation would handle multimodal content
                        parts
                            .iter()
                            .filter_map(|part| match part {
                                ContentPart::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<String>>()
                            .join("\n")
                    }
                };
                (content_str, name.clone(), None, None)
            }
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let content_str = match content {
                    Some(Content::Text(text)) => text.clone(),
                    Some(Content::Parts(parts)) => {
                        // Concatenate text parts
                        parts
                            .iter()
                            .filter_map(|part| match part {
                                ContentPart::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<String>>()
                            .join("\n")
                    }
                    None => String::new(),
                };

                // Convert tool calls if present
                let mistral_tool_calls = if !tool_calls.is_empty() {
                    let mut calls = Vec::with_capacity(tool_calls.len());

                    for tc in tool_calls {
                        calls.push(MistralToolCall {
                            id: tc.id.clone(),
                            r#type: tc.tool_type.clone(),
                            function: MistralFunctionCall {
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            },
                        });
                    }

                    Some(calls)
                } else {
                    None
                };

                (content_str, None, mistral_tool_calls, None)
            }
            Message::Tool {
                tool_call_id,
                content,
                ..
            } => (content.clone(), None, None, Some(tool_call_id.clone())),
        };

        MistralMessage {
            role,
            content,
            name,
            tool_calls,
            tool_call_id,
        }
    }
}

/// Convert from Mistral's response to our message format
impl From<&MistralResponse> for Message {
    fn from(response: &MistralResponse) -> Self {
        // Get the first choice (there should be at least one)
        if response.choices.is_empty() {
            return Message::assistant("No response generated");
        }

        let choice = &response.choices[0];
        let message = &choice.message;

        // Create appropriate Message variant based on role
        let mut msg = match message.role.as_str() {
            "assistant" => {
                let content = Some(Content::Text(message.content.clone()));

                // Convert tool calls if present
                if let Some(mistral_tool_calls) = &message.tool_calls {
                    if !mistral_tool_calls.is_empty() {
                        let mut tool_calls = Vec::with_capacity(mistral_tool_calls.len());

                        for call in mistral_tool_calls {
                            let tool_call = crate::message::ToolCall {
                                id: call.id.clone(),
                                tool_type: call.r#type.clone(),
                                function: crate::message::Function {
                                    name: call.function.name.clone(),
                                    arguments: call.function.arguments.clone(),
                                },
                            };
                            tool_calls.push(tool_call);
                        }

                        Message::Assistant {
                            content,
                            tool_calls,
                            metadata: Default::default(),
                        }
                    } else {
                        // No tool calls, just content
                        if let Some(Content::Text(text)) = content {
                            Message::assistant(text)
                        } else {
                            Message::Assistant {
                                content,
                                tool_calls: Vec::new(),
                                metadata: Default::default(),
                            }
                        }
                    }
                } else {
                    // No tool calls
                    if let Some(Content::Text(text)) = content {
                        Message::assistant(text)
                    } else {
                        Message::Assistant {
                            content,
                            tool_calls: Vec::new(),
                            metadata: Default::default(),
                        }
                    }
                }
            }
            "user" => {
                if let Some(name) = &message.name {
                    Message::user_with_name(name, message.content.clone())
                } else {
                    Message::user(message.content.clone())
                }
            }
            "system" => Message::system(message.content.clone()),
            "tool" => {
                if let Some(tool_call_id) = &message.tool_call_id {
                    Message::tool(tool_call_id, message.content.clone())
                } else {
                    // This shouldn't happen, but fall back to user message
                    Message::user(message.content.clone())
                }
            }
            _ => Message::user(message.content.clone()), // Default to user for unknown roles
        };

        // Add token usage information to metadata if available
        if let Some(usage) = &response.usage {
            msg = msg.with_metadata(
                "prompt_tokens",
                serde_json::Value::Number(usage.prompt_tokens.into()),
            );
            msg = msg.with_metadata(
                "completion_tokens",
                serde_json::Value::Number(usage.completion_tokens.into()),
            );
            msg = msg.with_metadata(
                "total_tokens",
                serde_json::Value::Number(usage.total_tokens.into()),
            );
        }

        msg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_conversion() {
        // Test simple text message
        let msg = Message::user("Hello, world!");
        let mistral_msg = MistralMessage::from(&msg);

        assert_eq!(mistral_msg.role, "user");
        assert_eq!(mistral_msg.content, "Hello, world!");

        // Test system message
        let msg = Message::system("You are a helpful assistant.");
        let mistral_msg = MistralMessage::from(&msg);

        assert_eq!(mistral_msg.role, "system");
        assert_eq!(mistral_msg.content, "You are a helpful assistant.");

        // Test assistant message
        let msg = Message::assistant("I can help with that.");
        let mistral_msg = MistralMessage::from(&msg);

        assert_eq!(mistral_msg.role, "assistant");
        assert_eq!(mistral_msg.content, "I can help with that.");
    }

    #[test]
    fn test_error_response_parsing() {
        let error_json = r#"{
            "error": {
                "message": "The model does not exist",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        }"#;

        let error_response: MistralErrorResponse = serde_json::from_str(error_json).unwrap();
        assert!(error_response.error.is_some());
        let error = error_response.error.unwrap();
        assert_eq!(error.error_type, "invalid_request_error");
        assert_eq!(error.code, Some("model_not_found".to_string()));
    }
}
