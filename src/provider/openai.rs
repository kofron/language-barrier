use crate::error::{Error, Result};
use crate::message::{Content, ContentPart, Message, MessageRole};
use crate::provider::HTTPProvider;
use crate::{Chat, ModelInfo};
use reqwest::{Method, Request, Url};
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{debug, error, info, instrument, trace, warn};

/// Configuration for the OpenAI provider
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the API
    pub base_url: String,
    /// Organization ID (optional)
    pub organization: Option<String>,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: env::var("OPENAI_ORGANIZATION").ok(),
        }
    }
}

/// Implementation of the OpenAI provider
#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    /// Configuration for the provider
    config: OpenAIConfig,
}

impl OpenAIProvider {
    /// Creates a new OpenAIProvider with default configuration
    ///
    /// This method will use the OPENAI_API_KEY environment variable for authentication.
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::openai::OpenAIProvider;
    ///
    /// let provider = OpenAIProvider::new();
    /// ```
    #[instrument(level = "debug")]
    pub fn new() -> Self {
        info!("Creating new OpenAIProvider with default configuration");
        let config = OpenAIConfig::default();
        debug!("API key set: {}", !config.api_key.is_empty());
        debug!("Base URL: {}", config.base_url);
        debug!("Organization set: {}", config.organization.is_some());

        Self { config }
    }

    /// Creates a new OpenAIProvider with custom configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::openai::{OpenAIProvider, OpenAIConfig};
    ///
    /// let config = OpenAIConfig {
    ///     api_key: "your-api-key".to_string(),
    ///     base_url: "https://api.openai.com/v1".to_string(),
    ///     organization: None,
    /// };
    ///
    /// let provider = OpenAIProvider::with_config(config);
    /// ```
    #[instrument(skip(config), level = "debug")]
    pub fn with_config(config: OpenAIConfig) -> Self {
        info!("Creating new OpenAIProvider with custom configuration");
        debug!("API key set: {}", !config.api_key.is_empty());
        debug!("Base URL: {}", config.base_url);
        debug!("Organization set: {}", config.organization.is_some());

        Self { config }
    }
}

impl Default for OpenAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: ModelInfo + OpenAIModelInfo> HTTPProvider<M> for OpenAIProvider {
    fn accept(&self, chat: Chat<M>) -> Result<Request> {
        info!("Creating request for OpenAI model: {:?}", chat.model);
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
        request.headers_mut().insert("Content-Type", content_type_header);
        
        // Add organization header if present
        if let Some(org) = &self.config.organization {
            match org.parse() {
                Ok(header) => {
                    request.headers_mut().insert("OpenAI-Organization", header);
                    debug!("Added organization header");
                }
                Err(e) => {
                    warn!("Failed to set organization header: {}", e);
                    // Continue without organization header
                }
            }
        }

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
        info!("Parsing response from OpenAI API");
        trace!("Raw response: {}", raw_response_text);

        // First try to parse as an error response
        if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&raw_response_text) {
            if let Some(error) = error_response.error {
                error!("OpenAI API returned an error: {}", error.message);
                return Err(Error::ProviderUnavailable(error.message));
            }
        }

        // If not an error, parse as a successful response
        debug!("Deserializing response JSON");
        let openai_response = match serde_json::from_str::<OpenAIResponse>(&raw_response_text) {
            Ok(response) => {
                debug!("Response deserialized successfully");
                debug!("Response model: {}", response.model);
                if !response.choices.is_empty() {
                    debug!("Number of choices: {}", response.choices.len());
                    debug!("First choice finish reason: {:?}", response.choices[0].finish_reason);
                }
                if let Some(usage) = &response.usage {
                    debug!("Token usage - prompt: {}, completion: {}, total: {}", 
                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens);
                }
                response
            }
            Err(e) => {
                error!("Failed to deserialize response: {}", e);
                error!("Raw response: {}", raw_response_text);
                return Err(Error::Serialization(e));
            }
        };

        // Convert to our message format using the From implementation
        debug!("Converting OpenAI response to Message");
        let message = Message::from(&openai_response);

        info!("Response parsed successfully");
        trace!("Message content: {:?}", message.content);

        Ok(message)
    }
}

// Trait to get OpenAI-specific model IDs
pub trait OpenAIModelInfo {
    fn openai_model_id(&self) -> String;
}

impl OpenAIProvider {
    /// Creates a request payload from a Chat object
    ///
    /// This method converts the Chat's messages and settings into an OpenAI-specific
    /// format for the API request.
    #[instrument(skip(self, chat), level = "debug")]
    fn create_request_payload<M: ModelInfo + OpenAIModelInfo>(&self, chat: &Chat<M>) -> Result<OpenAIRequest> {
        info!("Creating request payload for chat with OpenAI model");
        debug!("System prompt length: {}", chat.system_prompt.len());
        debug!("Messages in history: {}", chat.history.len());
        debug!("Max output tokens: {}", chat.max_output_tokens);

        let model_id = chat.model.openai_model_id();
        debug!("Using model ID: {}", model_id);

        // Convert all messages including system prompt
        debug!("Converting messages to OpenAI format");
        let mut messages: Vec<OpenAIMessage> = Vec::new();
        
        // Add system prompt if present
        if !chat.system_prompt.is_empty() {
            debug!("Adding system prompt");
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(chat.system_prompt.clone()),
                function_call: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        
        // Add conversation history
        for msg in &chat.history {
            debug!("Converting message with role: {:?}", msg.role);
            messages.push(OpenAIMessage::from(msg));
        }

        debug!("Converted {} messages for the request", messages.len());

        // Add tools if present
        let tools = if chat.has_toolbox() {
            let tool_descriptions = chat.tool_descriptions();
            debug!("Converting {} tool descriptions to OpenAI format", tool_descriptions.len());
            
            if !tool_descriptions.is_empty() {
                Some(
                    tool_descriptions
                        .into_iter()
                        .map(|desc| OpenAITool {
                            r#type: "function".to_string(),
                            function: OpenAIFunction {
                                name: desc.name,
                                description: desc.description,
                                parameters: desc.parameters,
                            },
                        })
                        .collect()
                )
            } else {
                None
            }
        } else {
            debug!("No toolbox provided");
            None
        };

        // Create the tool choice setting
        let tool_choice = if tools.is_some() { Some("auto".to_string()) } else { None };
        
        // Create the request
        debug!("Creating OpenAIRequest");
        let request = OpenAIRequest {
            model: model_id,
            messages,
            temperature: None,
            top_p: None,
            n: None,
            max_tokens: Some(chat.max_output_tokens),
            presence_penalty: None,
            frequency_penalty: None,
            stream: None,
            tools,
            tool_choice,
        };

        info!("Request payload created successfully");
        Ok(request)
    }
}

/// Represents a message in the OpenAI API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OpenAIMessage {
    /// The role of the message sender (system, user, assistant, etc.)
    pub role: String,
    /// The content of the message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The function call (deprecated in favor of tool_calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<OpenAIFunctionCall>,
    /// The name of the function
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    /// Tool call ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Represents a tool function in the OpenAI API format
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAIFunction {
    /// The name of the function
    pub name: String,
    /// The description of the function
    pub description: String,
    /// The parameters schema as a JSON object
    pub parameters: serde_json::Value,
}

/// Represents a tool in the OpenAI API format
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAITool {
    /// The type of the tool (currently always "function")
    pub r#type: String,
    /// The function definition
    pub function: OpenAIFunction,
}

/// Represents a function call in the OpenAI API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OpenAIFunctionCall {
    /// The name of the function
    pub name: String,
    /// The arguments as a JSON string
    pub arguments: String,
}

/// Represents a tool call in the OpenAI API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OpenAIToolCall {
    /// The ID of the tool call
    pub id: String,
    /// The type of the tool (currently always "function")
    pub r#type: String,
    /// The function call
    pub function: OpenAIFunctionCall,
}

/// Represents a request to the OpenAI API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAIRequest {
    /// The model to use
    pub model: String,
    /// The messages to send
    pub messages: Vec<OpenAIMessage>,
    /// Temperature (randomness)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Stream mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Tools available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    /// Tool choice strategy (auto, none, or a specific tool)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
}

/// Represents a response from the OpenAI API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAIResponse {
    /// Response ID
    pub id: String,
    /// Object type (always "chat.completion")
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model used
    pub model: String,
    /// Choices generated
    pub choices: Vec<OpenAIChoice>,
    /// Usage statistics
    pub usage: Option<OpenAIUsage>,
}

/// Represents a choice in an OpenAI response
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAIChoice {
    /// The index of the choice
    pub index: usize,
    /// The message generated
    pub message: OpenAIMessage,
    /// The reason generation stopped
    pub finish_reason: Option<String>,
}

/// Represents usage statistics in an OpenAI response
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAIUsage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total number of tokens
    pub total_tokens: u32,
}

/// Represents an error response from the OpenAI API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAIErrorResponse {
    /// The error details
    pub error: Option<OpenAIError>,
}

/// Represents an error from the OpenAI API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OpenAIError {
    /// The error message
    pub message: String,
    /// The error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// The error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Convert from our Message to OpenAI's message format
impl From<&Message> for OpenAIMessage {
    fn from(msg: &Message) -> Self {
        let role = match msg.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Function => "function",
            MessageRole::Tool => "tool",
        }.to_string();

        let content = match &msg.content {
            Some(Content::Text(text)) => Some(text.clone()),
            Some(Content::Parts(parts)) => {
                // For now, we just concatenate all text parts
                // A more complete implementation would handle multimodal content
                let combined_text = parts.iter()
                    .filter_map(|part| match part {
                        ContentPart::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<String>>()
                    .join("\n");
                
                if combined_text.is_empty() {
                    None
                } else {
                    Some(combined_text)
                }
            },
            None => None,
        };

        // Convert function call if present
        let function_call = if let Some(fc) = &msg.function_call {
            Some(OpenAIFunctionCall {
                name: fc.name.clone(),
                arguments: fc.arguments.clone(),
            })
        } else {
            None
        };

        // Convert tool calls if present
        let tool_calls = if let Some(tcs) = &msg.tool_calls {
            if !tcs.is_empty() {
                let mut calls = Vec::with_capacity(tcs.len());
                
                for tc in tcs {
                    calls.push(OpenAIToolCall {
                        id: tc.id.clone(),
                        r#type: tc.tool_type.clone(),
                        function: OpenAIFunctionCall {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    });
                }
                
                Some(calls)
            } else {
                None
            }
        } else {
            None
        };

        OpenAIMessage {
            role,
            content,
            function_call,
            name: msg.name.clone(),
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

/// Convert from OpenAI's response to our message format
impl From<&OpenAIResponse> for Message {
    fn from(response: &OpenAIResponse) -> Self {
        // Get the first choice (there should be at least one)
        if response.choices.is_empty() {
            return Message::assistant("No response generated");
        }

        let choice = &response.choices[0];
        let message = &choice.message;

        let role = match message.role.as_str() {
            "assistant" => MessageRole::Assistant,
            "user" => MessageRole::User,
            "system" => MessageRole::System,
            "function" => MessageRole::Function,
            "tool" => MessageRole::Tool,
            _ => MessageRole::User, // Default to user for unknown roles
        };

        let content = message.content.as_ref().map(|text| Content::Text(text.clone()));

        // Convert tool calls if present
        let tool_calls = if let Some(openai_tool_calls) = &message.tool_calls {
            if !openai_tool_calls.is_empty() {
                let mut tool_calls = Vec::with_capacity(openai_tool_calls.len());
                
                for call in openai_tool_calls {
                    let tool_call = crate::message::ToolCall {
                        id: call.id.clone(),
                        tool_type: call.r#type.clone(),
                        function: crate::message::FunctionCall {
                            name: call.function.name.clone(),
                            arguments: call.function.arguments.clone(),
                        },
                    };
                    tool_calls.push(tool_call);
                }
                
                Some(tool_calls)
            } else {
                None
            }
        } else {
            None
        };

        // Convert function call if present and no tool calls (older OpenAI API)
        let function_call = if tool_calls.is_none() && message.function_call.is_some() {
            let fc = message.function_call.as_ref().unwrap();
            Some(crate::message::FunctionCall {
                name: fc.name.clone(),
                arguments: fc.arguments.clone(),
            })
        } else {
            None
        };

        let mut msg = Message {
            role,
            content,
            name: message.name.clone(),
            function_call,
            tool_calls,
            tool_call_id: message.tool_call_id.clone(),
            metadata: Default::default(),
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
        let openai_msg = OpenAIMessage::from(&msg);
        
        assert_eq!(openai_msg.role, "user");
        assert_eq!(openai_msg.content, Some("Hello, world!".to_string()));
        
        // Test system message
        let msg = Message::system("You are a helpful assistant.");
        let openai_msg = OpenAIMessage::from(&msg);
        
        assert_eq!(openai_msg.role, "system");
        assert_eq!(openai_msg.content, Some("You are a helpful assistant.".to_string()));
        
        // Test assistant message
        let msg = Message::assistant("I can help with that.");
        let openai_msg = OpenAIMessage::from(&msg);
        
        assert_eq!(openai_msg.role, "assistant");
        assert_eq!(openai_msg.content, Some("I can help with that.".to_string()));
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
        
        let error_response: OpenAIErrorResponse = serde_json::from_str(error_json).unwrap();
        assert!(error_response.error.is_some());
        let error = error_response.error.unwrap();
        assert_eq!(error.error_type, "invalid_request_error");
        assert_eq!(error.code, Some("model_not_found".to_string()));
    }
}