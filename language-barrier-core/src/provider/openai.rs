use crate::error::{Error, Result};
use crate::message::{Content, ContentPart, Message};
use crate::provider::HTTPProvider;
use crate::{Chat, LlmToolInfo, OpenAi};
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
    /// use language_barrier_core::provider::openai::OpenAIProvider;
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
    /// use language_barrier_core::provider::openai::{OpenAIProvider, OpenAIConfig};
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

impl HTTPProvider<OpenAi> for OpenAIProvider {
    fn accept(&self, model: OpenAi, chat: &Chat) -> Result<Request> {
        info!("Creating request for OpenAI model: {:?}", model);
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
        let payload = match self.create_request_payload(model, chat) {
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
        if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&raw_response_text)
        {
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

        // Convert to our message format using the From implementation
        debug!("Converting OpenAI response to Message");
        let message = Message::from(&openai_response);

        info!("Response parsed successfully");
        trace!("Response message processed");

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
    fn create_request_payload(&self, model: OpenAi, chat: &Chat) -> Result<OpenAIRequest> {
        info!("Creating request payload for chat with OpenAI model");
        debug!("System prompt length: {}", chat.system_prompt.len());
        debug!("Messages in history: {}", chat.history.len());
        debug!("Max output tokens: {}", chat.max_output_tokens);

        let model_id = model.openai_model_id();
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
            debug!("Converting message with role: {}", msg.role_str());
            messages.push(OpenAIMessage::from(msg));
        }

        // OpenAI requires that every message with role "tool" directly follows the
        // corresponding assistant message that contains a matching `tool_call`.
        // When callers accidentally provide the messages in a different order
        // (for example: user → *tool* → assistant) the API rejects the request
        // with a 400:
        //   "messages with role 'tool' must be a response to a preceeding message \
        //    with 'tool_calls'".
        //
        // To make the client more robust we perform a best-effort re-ordering pass
        // so that each tool response is immediately preceded by its initiating
        // assistant message.  If we *cannot* find the corresponding assistant
        // message we treat this as a programming error and bail out early with
        // `Error::Other`.

        debug!("Re-ordering assistant / tool messages to satisfy OpenAI requirements");

        let mut i = 0;
        while i < messages.len() {
            // SAFETY: we only read, not mutate, inside the first borrow scope.
            if messages[i].role == "tool" {
                let tool_call_id_opt = messages[i].tool_call_id.clone();

                if let Some(ref tool_call_id) = tool_call_id_opt {
                    let has_preceding_assistant = if i == 0 {
                        false
                    } else {
                        let prev = &messages[i - 1];
                        prev.role == "assistant"
                            && prev
                                .tool_calls
                                .as_ref()
                                .is_some_and(|calls| calls.iter().any(|c| c.id == *tool_call_id))
                    };

                    if !has_preceding_assistant {
                        // Look ahead for the matching assistant message
                        if let Some(pos) =
                            messages.iter().enumerate().skip(i + 1).find_map(|(j, m)| {
                                if m.role == "assistant" {
                                    if let Some(calls) = &m.tool_calls {
                                        if calls.iter().any(|c| c.id == *tool_call_id) {
                                            return Some(j);
                                        }
                                    }
                                }
                                None
                            })
                        {
                            debug!(
                                "Found assistant (index {}) corresponding to tool message (index {}), re-ordering",
                                pos, i
                            );
                            let assistant_msg = messages.remove(pos);
                            messages.insert(i, assistant_msg);
                            // After inserting, the tool message is now at i+1, so we
                            // advance past both.
                            i += 2;
                            continue;
                        } else {
                            error!(
                                "Orphaned tool message with id '{}' at index {} (no matching assistant)",
                                tool_call_id, i
                            );
                            return Err(crate::error::Error::Other(format!(
                                "Tool message with id '{}' has no corresponding assistant message",
                                tool_call_id
                            )));
                        }
                    }
                } else {
                    // Tool message without id is invalid – drop it.
                    error!("Tool message without tool_call_id at index {}", i);
                    return Err(crate::error::Error::Other(
                        "Tool message missing tool_call_id".to_string(),
                    ));
                }
            }
            i += 1;
        }

        debug!(
            "Converted {} messages for the request after re-ordering",
            messages.len()
        );

        // Add tools if present
        let tools = chat
            .tools
            .as_ref()
            .map(|tools| tools.iter().map(OpenAITool::from).collect());

        // Create the tool choice setting
        let tool_choice = if let Some(choice) = &chat.tool_choice {
            // Use the explicitly configured choice
            match choice {
                crate::tool::ToolChoice::Auto => Some(serde_json::json!("auto")),
                // OpenAI uses "required" for what we call "Any"
                crate::tool::ToolChoice::Any => Some(serde_json::json!("required")),
                crate::tool::ToolChoice::None => Some(serde_json::json!("none")),
                crate::tool::ToolChoice::Specific(name) => {
                    // For specific tool, we need to create an object with type and function properties
                    Some(serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": name
                        }
                    }))
                }
            }
        } else if tools.is_some() {
            // Default to auto if tools are present but no choice specified
            Some(serde_json::json!("auto"))
        } else {
            None
        };

        // Create the request
        debug!("Creating OpenAIRequest");

        // Check if this is an O-series model (starts with "o-")
        let is_o_series = model_id.starts_with("o");

        let request = OpenAIRequest {
            model: model_id,
            messages,
            temperature: None,
            top_p: None,
            n: None,
            // For O-series models, use max_completion_tokens instead of max_tokens
            max_tokens: if is_o_series {
                None
            } else {
                Some(chat.max_output_tokens)
            },
            max_completion_tokens: if is_o_series {
                Some(chat.max_output_tokens)
            } else {
                None
            },
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

impl From<&LlmToolInfo> for OpenAITool {
    fn from(value: &LlmToolInfo) -> Self {
        OpenAITool {
            r#type: "function".to_string(),
            function: OpenAIFunction {
                name: value.name.clone(),
                description: value.description.clone(),
                parameters: value.parameters.clone(),
            },
        }
    }
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
    /// Maximum number of tokens to generate (for GPT models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// Maximum number of tokens to generate (for O-series models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<usize>,
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
    pub tool_choice: Option<serde_json::Value>,
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
        let role = match msg {
            Message::System { .. } => "system",
            Message::User { .. } => "user",
            Message::Assistant { .. } => "assistant",
            Message::Tool { .. } => "tool",
        }
        .to_string();

        let (content, name, function_call, tool_calls, tool_call_id) = match msg {
            Message::System { content, .. } => (Some(content.clone()), None, None, None, None),
            Message::User { content, name, .. } => {
                let content_str = match content {
                    Content::Text(text) => Some(text.clone()),
                    Content::Parts(parts) => {
                        // For text parts, concatenate them
                        let combined_text = parts
                            .iter()
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
                    }
                };
                (content_str, name.clone(), None, None, None)
            }
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let content_str = match content {
                    Some(Content::Text(text)) => Some(text.clone()),
                    Some(Content::Parts(parts)) => {
                        // For text parts, concatenate them
                        let combined_text = parts
                            .iter()
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
                    }
                    None => None,
                };

                // Convert tool calls if present
                let openai_tool_calls = if !tool_calls.is_empty() {
                    let mut calls = Vec::with_capacity(tool_calls.len());

                    for tc in tool_calls {
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
                };

                (content_str, None, None, openai_tool_calls, None)
            }
            Message::Tool {
                tool_call_id,
                content,
                ..
            } => (
                Some(content.clone()),
                None,
                None,
                None,
                Some(tool_call_id.clone()),
            ),
        };

        OpenAIMessage {
            role,
            content,
            function_call,
            name,
            tool_calls,
            tool_call_id,
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

        // Create appropriate Message variant based on role
        let mut msg = match message.role.as_str() {
            "assistant" => {
                let content = message
                    .content
                    .as_ref()
                    .map(|text| Content::Text(text.clone()));

                // Handle tool calls if present
                if let Some(openai_tool_calls) = &message.tool_calls {
                    if !openai_tool_calls.is_empty() {
                        let mut tool_calls = Vec::with_capacity(openai_tool_calls.len());

                        for call in openai_tool_calls {
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
                } else if let Some(fc) = &message.function_call {
                    // Handle legacy function_call (older OpenAI API)
                    let tool_call = crate::message::ToolCall {
                        id: format!("legacy_function_{}", fc.name),
                        tool_type: "function".to_string(),
                        function: crate::message::Function {
                            name: fc.name.clone(),
                            arguments: fc.arguments.clone(),
                        },
                    };

                    Message::Assistant {
                        content,
                        tool_calls: vec![tool_call],
                        metadata: Default::default(),
                    }
                } else {
                    // Simple content only
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
                    if let Some(content) = &message.content {
                        Message::user_with_name(name, content)
                    } else {
                        Message::user_with_name(name, "")
                    }
                } else if let Some(content) = &message.content {
                    Message::user(content)
                } else {
                    Message::user("")
                }
            }
            "system" => {
                if let Some(content) = &message.content {
                    Message::system(content)
                } else {
                    Message::system("")
                }
            }
            "tool" => {
                if let Some(tool_call_id) = &message.tool_call_id {
                    if let Some(content) = &message.content {
                        Message::tool(tool_call_id, content)
                    } else {
                        Message::tool(tool_call_id, "")
                    }
                } else {
                    // This shouldn't happen, but fall back to user message
                    if let Some(content) = &message.content {
                        Message::user(content)
                    } else {
                        Message::user("")
                    }
                }
            }
            _ => {
                // Default to user for unknown roles
                if let Some(content) = &message.content {
                    Message::user(content)
                } else {
                    Message::user("")
                }
            }
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
        assert_eq!(
            openai_msg.content,
            Some("You are a helpful assistant.".to_string())
        );

        // Test assistant message
        let msg = Message::assistant("I can help with that.");
        let openai_msg = OpenAIMessage::from(&msg);

        assert_eq!(openai_msg.role, "assistant");
        assert_eq!(
            openai_msg.content,
            Some("I can help with that.".to_string())
        );

        // Test assistant message with tool calls
        let tool_call = crate::message::ToolCall {
            id: "tool_123".to_string(),
            tool_type: "function".to_string(),
            function: crate::message::Function {
                name: "get_weather".to_string(),
                arguments: "{\"location\":\"San Francisco\"}".to_string(),
            },
        };

        let msg = Message::Assistant {
            content: Some(Content::Text("I'll check the weather".to_string())),
            tool_calls: vec![tool_call],
            metadata: Default::default(),
        };

        let openai_msg = OpenAIMessage::from(&msg);

        assert_eq!(openai_msg.role, "assistant");
        assert_eq!(
            openai_msg.content,
            Some("I'll check the weather".to_string())
        );
        assert!(openai_msg.tool_calls.is_some());
        let tool_calls = openai_msg.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tool_123");
        assert_eq!(tool_calls[0].function.name, "get_weather");
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

    #[test]
    fn test_tool_messages_reordered() {
        use crate::Chat;
        use crate::message::{Function, ToolCall};
        use crate::model::OpenAi;

        let tool_call_id = "call_123";

        // Build chat with invalid order: user -> tool -> assistant(tool_calls)
        let chat = Chat::default()
            .add_message(crate::message::Message::user("hello"))
            .add_message(crate::message::Message::tool(tool_call_id, "OK"))
            .add_message(crate::message::Message::assistant_with_tool_calls(vec![
                ToolCall {
                    id: tool_call_id.to_string(),
                    tool_type: "function".to_string(),
                    function: Function {
                        name: "respond_chat".to_string(),
                        arguments: "{}".to_string(),
                    },
                },
            ]));

        let provider = OpenAIProvider::new();
        let model = OpenAi::GPT35Turbo;

        // Use the private method directly (we are in the same module).
        let request = provider
            .create_request_payload(model, &chat)
            .expect("payload generation failed");

        let roles: Vec<_> = request.messages.iter().map(|m| m.role.as_str()).collect();

        // Expected order: user, assistant, tool
        let user_idx = roles.iter().position(|r| *r == "user").unwrap();
        let assistant_idx = roles.iter().position(|r| *r == "assistant").unwrap();
        let tool_idx = roles.iter().position(|r| *r == "tool").unwrap();

        assert!(assistant_idx > user_idx);
        assert!(tool_idx > assistant_idx);
    }

    #[test]
    fn test_orphan_tool_returns_error() {
        use crate::Chat;

        // tool message without matching assistant
        let chat = Chat::default()
            .add_message(crate::message::Message::user("hello"))
            .add_message(crate::message::Message::tool("call_999", "OK"));

        let provider = OpenAIProvider::new();
        let model = crate::model::OpenAi::GPT35Turbo;

        let res = provider.create_request_payload(model, &chat);
        assert!(res.is_err(), "Expected error for orphaned tool");
    }
}
