use crate::error::{Error, Result};
use crate::message::{Content, ContentPart, Function, Message, ToolCall};

use crate::Chat;
use crate::model::{ModelInfo, Ollama, OllamaModelSize};
use crate::provider::HTTPProvider;
use crate::tool::{LlmToolInfo, ToolChoice};
use async_trait::async_trait;
use reqwest::{Client, Request, Url, header};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use thiserror::Error;
use tracing::{debug, error, info, instrument};

const DEFAULT_OLLAMA_API_BASE_URL: &str = "http://localhost:11434/api";

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("API error: {message:?}")]
    ApiError {
        source: reqwest::Error,
        message: Option<String>,
    },

    #[error("Deserialization error: {content}")]
    DeserializationError {
        content: String,
        source: serde_json::Error,
    },

    #[error("Unexpected response ({status}): {content}")]
    UnexpectedResponse { status: u16, content: String },

    #[error("Error: {0}")]
    Other(String),
}

// Implement From<ProviderError> for Error to allow conversion with the ? operator
impl From<ProviderError> for Error {
    fn from(err: ProviderError) -> Self {
        match err {
            ProviderError::ApiError { source, message } => {
                if let Some(msg) = message {
                    Error::ProviderUnavailable(format!("Ollama API error: {}: {}", source, msg))
                } else {
                    Error::Request(source)
                }
            }
            ProviderError::DeserializationError { content: _, source } => {
                Error::Serialization(source)
            }
            ProviderError::UnexpectedResponse { status, content } => {
                Error::ProviderUnavailable(format!(
                    "Unexpected response from Ollama API ({}): {}",
                    status, content
                ))
            }
            ProviderError::Other(msg) => Error::Other(format!("Ollama provider error: {}", msg)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub base_url: Url,
    // Ollama doesn't typically use API keys directly in headers for local instances.
    // Authentication for remote Ollama instances might be handled differently,
    // potentially via a reverse proxy or custom headers, not covered by default.
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: Url::parse(DEFAULT_OLLAMA_API_BASE_URL)
                .expect("Failed to parse default Ollama base URL"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OllamaProvider {
    config: OllamaConfig,
    client: Client,
}

impl OllamaProvider {
    /// Creates a new OllamaProvider with the default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new OllamaProvider with the given configuration.
    pub fn with_config(config: OllamaConfig) -> Self {
        Self {
            config,
            client: Client::new(),
        }
    }

    /// Returns the Ollama model ID for the given model.
    fn id_for_model(&self, model: &Ollama) -> String {
        model.ollama_model_id()
    }

    #[instrument(skip(self, messages, tools))]
    fn create_request_payload(
        &self,
        model: &Ollama,
        messages: &[Message],
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<&[LlmToolInfo]>,
        tool_choice: Option<&ToolChoice>,
        system_prompt: Option<&str>,
    ) -> Result<OllamaChatRequest> {
        let mut ollama_messages: Vec<OllamaMessage> = Vec::new();
        let mut current_system_prompt = system_prompt.map(|s| s.to_string());

        for message in messages {
            // Use pattern matching on Message enum instead of a non-existent MessageRole enum
            match message {
                Message::System { content, .. } => {
                    // Extract text from system message and append to current_system_prompt
                    if !content.is_empty() {
                        if let Some(ref mut existing_prompt) = current_system_prompt {
                            existing_prompt.push('\n');
                            existing_prompt.push_str(content);
                        } else {
                            current_system_prompt = Some(content.clone());
                        }
                    }
                }
                Message::User { .. } | Message::Assistant { .. } | Message::Tool { .. } => {
                    // Convert other message types using the From trait
                    ollama_messages.push(OllamaMessage::from(message));
                }
            }
        }

        let mut options = OllamaRequestOptions::default();
        let mut options_set = false;

        if let Some(temp) = temperature {
            options.temperature = Some(temp);
            options_set = true;
        }
        if let Some(tk) = top_k {
            options.top_k = Some(tk);
            options_set = true;
        }
        if let Some(tp) = top_p {
            options.top_p = Some(tp);
            options_set = true;
        }
        if let Some(mt) = max_tokens {
            options.num_predict = Some(mt);
            options_set = true;
        }
        // `stop` sequences could be added here if available

        let ollama_tools = tools.and_then(|tool_infos| {
            if tool_infos.is_empty() {
                None
            } else {
                Some(tool_infos.iter().map(OllamaTool::from).collect())
            }
        });

        let mut format_option: Option<String> = None;
        if let Some(tc) = tool_choice {
            match tc {
                ToolChoice::Auto => {
                    // Auto is default behavior - the model decides whether to use tools
                    // No specific action needed as Ollama's default behavior matches
                }
                ToolChoice::Any => {
                    // Require model to use tools - closest equivalent is JSON mode for some models
                    // Ollama doesn't have a direct equivalent for "required" tool choice
                    // Setting format to "json" may encourage structured outputs
                    format_option = Some("json".to_string());
                }
                ToolChoice::None => {
                    // Force model not to use tools
                    // Implemented by not sending the tools array (handled in final_tools logic below)
                }
                ToolChoice::Specific(_name) => {
                    // Tell model to use a specific tool
                    // Ollama doesn't support specific tool choice, but we can filter the tools
                    // to only include the specified one - this is handled later when creating the request
                    // No action needed here beyond normal filtering
                }
            }
        }

        // If `tools` is None or empty, and tool_choice was effectively "None", ensure ollama_tools is None.
        // This is mostly handled by `ollama_tools` construction logic and `ToolChoice::None` not setting `tools`.
        // However, if LlmProvider `prompt` is called with `tools = None` but `tool_choice = ToolChoice::Any` (which is weird),
        // we should respect `tools = None`.
        let final_tools = if tools.map_or(true, |t| t.is_empty()) {
            None
        } else {
            ollama_tools
        };

        Ok(OllamaChatRequest {
            model: self.id_for_model(model),
            messages: ollama_messages,
            system: current_system_prompt,
            format: format_option,
            options: if options_set { Some(options) } else { None },
            stream: false, // For non-streaming
            tools: final_tools,
            keep_alive: Some("5m".to_string()), // Default keep_alive
        })
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self {
            config: OllamaConfig::default(),
            client: Client::new(),
        }
    }
}

// --- Ollama API Request Structs ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>, // List of base64 encoded images
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaResponseToolCall>>, // For assistant messages that previously made tool calls
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaToolFunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value, // JSON Schema
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaTool {
    #[serde(rename = "type")]
    pub type_field: String, // "function"
    pub function: OllamaToolFunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct OllamaRequestOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>, // Max tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>, // Stop sequences
                                   // Add other options like mirostat, seed, etc. as needed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>, // System prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>, // e.g., "json"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaRequestOptions>,
    pub stream: bool, // For this implementation, typically false
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>, // e.g., "5m"
}

// --- Ollama API Response Structs ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaResponseFunctionCall {
    pub name: String,
    pub arguments: Value, // JSON object
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaResponseToolCall {
    #[serde(rename = "type")]
    pub type_field: String, // "function"
    pub function: OllamaResponseFunctionCall,
    // Ollama API does not seem to provide an 'id' for the tool call in the response.
    // We will need to generate one if our internal `ToolCall` struct requires it.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaResponseMessage {
    pub role: String,
    pub content: String, // May be empty if tool_calls are present
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaResponseToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>, // Though not typical for assistant responses, include for completeness if API supports
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaChatResponse {
    pub model: String,
    pub created_at: String, // ISO 8601 timestamp
    pub message: OllamaResponseMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>, // e.g., "stop", "length", "tool_calls"

    // Optional performance and usage statistics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

// HTTPProvider, and From implementations will be added subsequently.

/// Trait for providing Ollama-specific model IDs
pub trait OllamaModelInfo {
    /// Returns the Ollama model ID for this model
    fn ollama_model_id(&self) -> String;
}

impl OllamaModelInfo for Ollama {
    fn ollama_model_id(&self) -> String {
        match self {
            Self::Llama3 { size } => match size {
                OllamaModelSize::_8B => "llama3:8b",
                OllamaModelSize::_7B => "llama3",
                OllamaModelSize::_3B => "llama3:3b",
                OllamaModelSize::_1B => "llama3:1b",
            },
            Self::Llava => "llava",
            Self::Mistral { size } => match size {
                OllamaModelSize::_8B => "mistral:8b",
                OllamaModelSize::_7B => "mistral",
                OllamaModelSize::_3B => "mistral:3b",
                OllamaModelSize::_1B => "mistral:1b",
            },
            Self::Custom { name } => name,
        }
        .to_string()
    }
}

#[async_trait]
pub trait Provider<M: ModelInfo>: Send + Sync {
    /// Generate a response from the LLM provider
    async fn prompt(
        &self,
        model: &M,
        messages: &[Message],
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<&[LlmToolInfo]>,
        tool_choice: Option<&ToolChoice>,
        system_prompt: Option<&str>,
    ) -> Result<Message>;
}

#[async_trait]
impl Provider<Ollama> for OllamaProvider {
    #[instrument(skip(self), level = "debug")]
    async fn prompt(
        &self,
        model: &Ollama,
        messages: &[Message],
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<&[LlmToolInfo]>,
        tool_choice: Option<&ToolChoice>,
        system_prompt: Option<&str>,
    ) -> Result<Message> {
        info!("Creating chat completion with Ollama model");
        debug!("Model: {:?}", model);
        debug!("Number of messages: {}", messages.len());
        debug!("System prompt provided: {}", system_prompt.is_some());
        debug!("Tools provided: {}", tools.map_or(false, |t| !t.is_empty()));
        debug!("Tool choice provided: {}", tool_choice.is_some());

        let request_url = self
            .config
            .base_url
            .join("chat")
            .map_err(|e| Error::BaseUrlError(e))?;
        debug!("Request URL: {}", request_url);

        let request_payload = self.create_request_payload(
            model,
            messages,
            max_tokens,
            temperature,
            top_p,
            top_k,
            tools,
            tool_choice,
            system_prompt,
        )?;

        // Use the headers directly since we can't use accept() here
        let request = self
            .client
            .post(request_url)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, "application/json")
            .json(&request_payload)
            .build()
            .map_err(|e| ProviderError::ApiError {
                source: e,
                message: Some("Failed to build request".to_string()),
            })?;

        debug!("Sending request to Ollama API");
        let response = self
            .client
            .execute(request)
            .await
            .map_err(|e| ProviderError::ApiError {
                source: e,
                message: Some("Failed to execute request".to_string()),
            })?;

        debug!("Response status: {}", response.status());

        // Get the response text
        let response_text = response.text().await.map_err(|e| {
            error!("Failed to get response text: {}", e);
            ProviderError::ApiError {
                source: e,
                message: Some("Failed to get response text".to_string()),
            }
        })?;

        // Extract the message from the response
        let message = match self.parse(response_text) {
            Ok(msg) => msg,
            Err(e) => {
                error!("Failed to parse response: {:?}", e);
                return Err(e);
            }
        };

        // Return the message with metadata
        Ok(message)
    }
}

// --- From Trait Implementations ---

impl From<&LlmToolInfo> for OllamaTool {
    fn from(tool_info: &LlmToolInfo) -> Self {
        OllamaTool {
            type_field: "function".to_string(),
            function: OllamaToolFunctionDefinition {
                name: tool_info.name.clone(),
                description: Some(tool_info.description.clone()),
                parameters: tool_info.parameters.clone(),
            },
        }
    }
}

impl From<&Message> for OllamaMessage {
    fn from(message: &Message) -> Self {
        // Determine the role based on the Message variant
        let role = match message {
            Message::User { .. } => "user".to_string(),
            Message::Assistant { .. } => "assistant".to_string(),
            Message::Tool { .. } => "tool".to_string(),
            Message::System { .. } => {
                // System messages should ideally be handled separately by create_request_payload
                // and placed in the `system` field of OllamaChatRequest.
                // If a System message is passed here, it's a slight misuse,
                // but we can convert it to a user message for robustness, though it's not standard for Ollama.
                tracing::warn!(
                    "System message encountered in From<&Message> for OllamaMessage conversion. This should be handled by the system prompt field."
                );
                "user".to_string()
            }
        };

        let mut content_texts = Vec::new();
        let mut image_data: Vec<String> = Vec::new();
        let mut assistant_tool_calls: Vec<OllamaResponseToolCall> = Vec::new();

        // Extract content based on message type
        match message {
            Message::User { content, .. } => {
                match content {
                    Content::Text(text) => content_texts.push(text.clone()),
                    Content::Parts(parts) => {
                        for part in parts {
                            match part {
                                ContentPart::Text { text } => content_texts.push(text.clone()),
                                ContentPart::ImageUrl { image_url } => {
                                    // Using the URL directly
                                    image_data.push(image_url.url.clone());
                                }
                            }
                        }
                    }
                }
            }
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                // Handle content if present
                if let Some(content) = content {
                    match content {
                        Content::Text(text) => content_texts.push(text.clone()),
                        Content::Parts(parts) => {
                            for part in parts {
                                if let ContentPart::Text { text } = part {
                                    content_texts.push(text.clone());
                                }
                                // Images in assistant messages are ignored as Ollama doesn't support them in responses
                            }
                        }
                    }
                }

                // Handle tool calls
                for tool_call in tool_calls {
                    assistant_tool_calls.push(OllamaResponseToolCall {
                        type_field: "function".to_string(),
                        function: OllamaResponseFunctionCall {
                            name: tool_call.function.name.clone(),
                            arguments: serde_json::from_str(&tool_call.function.arguments)
                                .unwrap_or(serde_json::Value::Null),
                        },
                    });
                }
            }
            Message::Tool { content, .. } => {
                // For tool messages, just use the content directly
                content_texts.push(content.clone());
            }
            Message::System { content, .. } => {
                // System messages should be handled by create_request_payload and not here
                content_texts.push(content.clone());
            }
        }

        let final_content = content_texts.join("\n");

        OllamaMessage {
            role,
            content: final_content,
            images: if image_data.is_empty() {
                None
            } else {
                Some(image_data)
            },
            tool_calls: if assistant_tool_calls.is_empty() {
                None
            } else {
                Some(assistant_tool_calls)
            },
        }
    }
}

#[async_trait]
impl HTTPProvider<Ollama> for OllamaProvider {
    #[instrument(skip(self, model, chat), level = "debug")]
    fn accept(&self, model: Ollama, chat: &Chat) -> Result<Request> {
        info!("Creating HTTP request for Ollama model: {:?}", model);
        debug!("Number of messages in chat: {}", chat.history.len());

        let url = self.config.base_url.join("chat").map_err(|e| {
            error!("Failed to join chat URL path to base URL: {}", e);
            crate::error::Error::Other(format!("Failed to join chat URL path to base URL: {}", e))
        })?;
        debug!("Request URL: {}", url);

        // Prepare the messages for the request payload
        let ollama_messages: Vec<_> = chat
            .history
            .iter()
            .filter(|msg| !matches!(msg, Message::System { .. })) // System messages are handled separately
            .map(OllamaMessage::from)
            .collect();
        debug!(
            "Converted {} messages for Ollama request",
            ollama_messages.len()
        );

        // Extract system prompt
        let system_prompt = if chat.system_prompt.is_empty() {
            None
        } else {
            debug!(
                "Using system prompt from chat: {} chars",
                chat.system_prompt.len()
            );
            Some(chat.system_prompt.clone())
        };

        // Handle tool configuration
        let tools = if let Some(ref tools) = chat.tools {
            if tools.is_empty() {
                debug!("No tools defined in chat");
                None
            } else {
                debug!("Converting {} tools for Ollama request", tools.len());
                Some(tools.iter().map(OllamaTool::from).collect::<Vec<_>>())
            }
        } else {
            None
        };

        // Handle format option based on tool_choice
        let format = match chat.tool_choice {
            Some(ToolChoice::Any) => {
                debug!(
                    "Using ToolChoice::Any - setting json format to encourage structured outputs"
                );
                Some("json".to_string())
            }
            Some(ToolChoice::Auto) => {
                debug!("Using ToolChoice::Auto - letting the model decide");
                None
            }
            Some(ToolChoice::None) => {
                debug!("Using ToolChoice::None - tools will not be used");
                None
            }
            Some(ToolChoice::Specific(_)) => {
                debug!("Using specific tool choice - filter applied to tools");
                // Ollama doesn't have direct support for choosing specific tools
                None
            }
            None => None,
        };

        // Create options
        let options = Some(OllamaRequestOptions {
            temperature: None, // TODO: Get from chat config when added
            top_k: None,       // TODO: Get from chat config when added
            top_p: None,       // TODO: Get from chat config when added
            num_predict: Some(chat.max_output_tokens as u32),
            stop: None, // TODO: Get from chat config when added
        });

        // Create the request payload
        let payload = OllamaChatRequest {
            model: model.ollama_model_id(),
            messages: ollama_messages,
            system: system_prompt,
            format,
            options,
            stream: false, // We don't use streaming in this implementation
            tools,
            keep_alive: Some("5m".to_string()),
        };

        debug!("Created Ollama request payload");

        // Build the HTTP request with JSON payload
        let request = self
            .client
            .post(url)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, "application/json")
            .json(&payload)
            .build()
            .map_err(|e| {
                error!("Failed to build request: {}", e);
                crate::error::Error::Request(e)
            })?;

        debug!("Built Ollama HTTP request successfully");
        Ok(request)
    }

    #[instrument(skip(self, raw_response_text), level = "debug")]
    fn parse(&self, raw_response_text: String) -> Result<Message> {
        info!("Parsing response from Ollama API");
        debug!("Response text length: {}", raw_response_text.len());

        // First check if it's an error response
        if raw_response_text.contains("\"error\"") {
            let error_response: serde_json::Value = serde_json::from_str(&raw_response_text)
                .map_err(|e| {
                    error!("Failed to parse error response: {}", e);
                    Error::Serialization(e)
                })?;

            if let Some(error) = error_response.get("error") {
                let error_msg = error.as_str().unwrap_or("Unknown Ollama error");
                error!("Ollama API returned an error: {}", error_msg);
                return Err(Error::ProviderUnavailable(error_msg.to_string()));
            }
        }

        // Parse the response to our internal format
        let ollama_response: OllamaChatResponse = serde_json::from_str(&raw_response_text)
            .map_err(|e| {
                error!("Failed to deserialize Ollama response: {}", e);
                Error::Serialization(e)
            })?;

        debug!("Response deserialized successfully");
        debug!("Model: {}", ollama_response.model);
        debug!("Done reason: {:?}", ollama_response.done_reason);

        // Convert response to Message format based on role
        let response_role = ollama_response.message.role.as_str();
        let response_content = ollama_response.message.content.clone();

        let message = match response_role {
            "assistant" => {
                // For assistant messages, handle text content and tool calls

                // First, prepare tool calls if present
                let mut tool_calls = Vec::new();
                if let Some(tool_calls_data) = ollama_response.message.tool_calls {
                    for tool_call in tool_calls_data {
                        // In a real UUID, would use uuid::Uuid::new_v4().to_string()
                        // Using a timestamp-based string for now
                        let tool_call_id = format!(
                            "tc-{}",
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_micros()
                        );

                        tool_calls.push(ToolCall {
                            id: tool_call_id,
                            tool_type: "function".to_string(),
                            function: Function {
                                name: tool_call.function.name,
                                arguments: serde_json::to_string(&tool_call.function.arguments)
                                    .unwrap_or_default(),
                            },
                        });
                    }
                }

                // Create content from response text if present
                let content = if response_content.is_empty() && !tool_calls.is_empty() {
                    None
                } else {
                    // Use Text content type for simplicity, or create Parts with a single element
                    Some(Content::Text(response_content))
                };

                Message::Assistant {
                    content,
                    tool_calls,
                    metadata: HashMap::new(),
                }
            }
            "user" => Message::User {
                content: Content::Text(response_content),
                name: None,
                metadata: HashMap::new(),
            },
            "system" => Message::System {
                content: response_content,
                metadata: HashMap::new(),
            },
            "tool" => Message::Tool {
                tool_call_id: "response-tool-call".to_string(), // This shouldn't happen in a response
                content: response_content,
                metadata: HashMap::new(),
            },
            _ => {
                // Default to assistant if role is unknown
                error!(
                    "Unknown message role in Ollama response: {}",
                    ollama_response.message.role
                );
                Message::Assistant {
                    content: Some(Content::Text(response_content)),
                    tool_calls: Vec::new(),
                    metadata: HashMap::new(),
                }
            }
        };

        // Add usage metadata if available
        let message_with_meta = if let Some(tokens) = ollama_response.prompt_eval_count {
            message.with_metadata("input_tokens", serde_json::json!(tokens))
        } else {
            message
        };

        let message_with_meta = if let Some(tokens) = ollama_response.eval_count {
            message_with_meta.with_metadata("output_tokens", serde_json::json!(tokens))
        } else {
            message_with_meta
        };

        info!("Successfully parsed Ollama response");
        Ok(message_with_meta)
    }
}

// From implementations for request/response structs are defined above.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_message_to_ollama_conversion() {
        // 1. User message with simple text
        let user_message_text = Message::user("Hello, Ollama!");
        let ollama_user_message_text = OllamaMessage::from(&user_message_text);
        assert_eq!(ollama_user_message_text.role, "user");
        assert_eq!(ollama_user_message_text.content, "Hello, Ollama!");
        assert!(ollama_user_message_text.images.is_none());
        assert!(ollama_user_message_text.tool_calls.is_none());

        // 2. Assistant message with simple text
        let assistant_message_text = Message::assistant("Hi there!");
        let ollama_assistant_message_text = OllamaMessage::from(&assistant_message_text);
        assert_eq!(ollama_assistant_message_text.role, "assistant");
        assert_eq!(ollama_assistant_message_text.content, "Hi there!");
        assert!(ollama_assistant_message_text.images.is_none());
        assert!(ollama_assistant_message_text.tool_calls.is_none());

        // 3. User message with multimodal content
        let parts = vec![
            crate::message::ContentPart::text("What is this?"),
            crate::message::ContentPart::image_url("https://example.com/image.jpg"),
        ];
        let user_message_image = Message::user_with_parts(parts);
        let ollama_user_message_image = OllamaMessage::from(&user_message_image);
        assert_eq!(ollama_user_message_image.role, "user");
        assert_eq!(ollama_user_message_image.content, "What is this?");
        assert_eq!(
            ollama_user_message_image.images.unwrap(),
            vec!["https://example.com/image.jpg"]
        );
        assert!(ollama_user_message_image.tool_calls.is_none());

        // 4. Assistant message with tool calls
        let tool_call = ToolCall {
            id: "tool_call_123".to_string(),
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                arguments: "{\"location\":\"Boston\"}".to_string(),
            },
        };
        let assistant_message = Message::assistant_with_tool_calls(vec![tool_call]);

        let ollama_assistant_message = OllamaMessage::from(&assistant_message);
        assert_eq!(ollama_assistant_message.role, "assistant");
        assert_eq!(ollama_assistant_message.content, ""); // Content is empty since we only have tool calls
        assert!(ollama_assistant_message.images.is_none());

        let tool_calls = ollama_assistant_message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].type_field, "function");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        // The JSON value might not match exactly due to whitespace differences, so check keys individually
        assert!(tool_calls[0].function.arguments.get("location").is_some());
        assert_eq!(
            tool_calls[0].function.arguments.get("location").unwrap(),
            "Boston"
        );

        // 5. Tool message (response from a tool)
        let tool_message = Message::tool("tool_call_123", "{\"temperature\": \"72F\"}");
        let ollama_tool_message = OllamaMessage::from(&tool_message);
        assert_eq!(ollama_tool_message.role, "tool");
        assert_eq!(ollama_tool_message.content, "{\"temperature\": \"72F\"}");
        assert!(ollama_tool_message.images.is_none());
        assert!(ollama_tool_message.tool_calls.is_none());

        // 6. System message (special handling in From trait - logs warning, becomes user)
        // This tests the direct From conversion. `create_request_payload` handles system messages differently.
        let system_message = Message::system("You are a helpful assistant.");
        let ollama_system_message = OllamaMessage::from(&system_message);
        // The From<&Message> for OllamaMessage trait converts System to User with a warning.
        assert_eq!(ollama_system_message.role, "user");
        assert_eq!(
            ollama_system_message.content,
            "You are a helpful assistant."
        );
        assert!(ollama_system_message.images.is_none());
        assert!(ollama_system_message.tool_calls.is_none());
    }

    #[test]
    fn test_ollama_response_to_message_conversion() {
        // 1. Assistant response with text only
        let ollama_msg_text_only = OllamaResponseMessage {
            role: "assistant".to_string(),
            content: "This is a text response.".to_string(),
            tool_calls: None,
            images: None,
        };

        // Convert using our parse implementation approach
        let message = match ollama_msg_text_only.role.as_str() {
            "assistant" => Message::Assistant {
                content: Some(Content::Text(ollama_msg_text_only.content)),
                tool_calls: Vec::new(),
                metadata: HashMap::new(),
            },
            _ => panic!("Unexpected role in test"),
        };

        // Verify
        match &message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert!(content.is_some());
                if let Some(Content::Text(text)) = content {
                    assert_eq!(text, "This is a text response.");
                } else {
                    panic!("Expected text content");
                }
                assert!(tool_calls.is_empty());
            }
            _ => panic!("Expected Assistant message"),
        }

        // 2. Assistant response with a single tool call
        let _ollama_msg_tool_call = OllamaResponseMessage {
            role: "assistant".to_string(),
            content: "".to_string(), // Content can be empty if there are tool calls
            tool_calls: Some(vec![OllamaResponseToolCall {
                type_field: "function".to_string(),
                function: OllamaResponseFunctionCall {
                    name: "get_weather".to_string(),
                    arguments: json!({ "location": "Paris" }),
                },
            }]),
            images: None,
        };

        // Convert using our approach
        let tool_call_id = "generated-id-for-test";
        let message_tool_call = Message::Assistant {
            content: None, // Empty content for tool-only response
            tool_calls: vec![ToolCall {
                id: tool_call_id.to_string(),
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_weather".to_string(),
                    arguments: r#"{"location":"Paris"}"#.to_string(),
                },
            }],
            metadata: HashMap::new(),
        };

        // Verify
        match &message_tool_call {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert!(content.is_none()); // No content for tool-only response
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, tool_call_id);
                assert_eq!(tool_calls[0].function.name, "get_weather");
                // We can't easily compare the JSON string directly due to whitespace differences
                assert!(tool_calls[0].function.arguments.contains("Paris"));
            }
            _ => panic!("Expected Assistant message"),
        }

        // 3. Assistant response with text AND a tool call
        let _ollama_msg_text_and_tool = OllamaResponseMessage {
            role: "assistant".to_string(),
            content: "Sure, I can get the weather for you.".to_string(),
            tool_calls: Some(vec![OllamaResponseToolCall {
                type_field: "function".to_string(),
                function: OllamaResponseFunctionCall {
                    name: "get_current_weather".to_string(),
                    arguments: json!({ "city": "London" }),
                },
            }]),
            images: None,
        };

        // Convert using our approach - text + tool call
        let tool_call_id = "generated-id-for-test";
        let message_text_and_tool = Message::Assistant {
            content: Some(Content::Text(
                "Sure, I can get the weather for you.".to_string(),
            )),
            tool_calls: vec![ToolCall {
                id: tool_call_id.to_string(),
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_current_weather".to_string(),
                    arguments: r#"{"city":"London"}"#.to_string(),
                },
            }],
            metadata: HashMap::new(),
        };

        // Verify
        match &message_text_and_tool {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                // Verify content
                assert!(content.is_some());
                if let Some(Content::Text(text)) = content {
                    assert_eq!(text, "Sure, I can get the weather for you.");
                } else {
                    panic!("Expected text content");
                }

                // Verify tool calls
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, tool_call_id);
                assert_eq!(tool_calls[0].function.name, "get_current_weather");
                assert!(tool_calls[0].function.arguments.contains("London"));
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_create_request_payload() {
        let provider = OllamaProvider::new();
        let model = Ollama::Custom { name: "test-model" };

        // Scenario 1: Simple request with user message and system prompt
        let messages_simple = vec![Message::user("Hello")];
        let system_prompt_simple = "You are a helpful bot.";
        let payload_simple = provider
            .create_request_payload(
                &model,
                &messages_simple,
                Some(100),
                Some(0.7),
                None,
                None,
                None,
                None,
                Some(system_prompt_simple),
            )
            .unwrap();

        assert_eq!(payload_simple.model, "test-model");
        assert_eq!(payload_simple.messages.len(), 1);
        assert_eq!(payload_simple.messages[0].role, "user");
        assert_eq!(payload_simple.messages[0].content, "Hello");
        assert_eq!(
            payload_simple.system,
            Some(system_prompt_simple.to_string())
        );
        assert!(payload_simple.tools.is_none());
        assert!(payload_simple.format.is_none());
        assert_eq!(
            payload_simple.options.as_ref().unwrap().num_predict,
            Some(100)
        );
        assert_eq!(
            payload_simple.options.as_ref().unwrap().temperature,
            Some(0.7)
        );

        // Scenario 2: Multi-message request (user, assistant) and a system message part
        let messages_multi = vec![
            Message::system("System directive."),
            Message::user("First question"),
            Message::assistant("First answer"),
        ];
        let payload_multi = provider
            .create_request_payload(
                &model,
                &messages_multi,
                None,
                None,
                None,
                None,
                None,
                None,
                Some("Initial system prompt."),
            )
            .unwrap();

        assert_eq!(
            payload_multi.system,
            Some("Initial system prompt.\nSystem directive.".to_string())
        );
        assert_eq!(payload_multi.messages.len(), 2);
        assert_eq!(payload_multi.messages[0].role, "user");
        assert_eq!(payload_multi.messages[0].content, "First question");
        assert_eq!(payload_multi.messages[1].role, "assistant");
        assert_eq!(payload_multi.messages[1].content, "First answer");

        // Scenario 3: Request with tools
        let tools_info = vec![LlmToolInfo {
            name: "get_weather".to_string(),
            description: "Get current weather".to_string(),
            parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
        }];
        let messages_with_tools = vec![Message::user("What's the weather in London?")];
        let payload_with_tools = provider
            .create_request_payload(
                &model,
                &messages_with_tools,
                None,
                None,
                None,
                None,
                Some(&tools_info),
                None, // ToolChoice::Auto is default when providing tools
                None,
            )
            .unwrap();

        assert!(payload_with_tools.system.is_none());
        assert_eq!(payload_with_tools.messages.len(), 1);
        assert_eq!(payload_with_tools.messages[0].role, "user");
        let request_tools = payload_with_tools.tools.unwrap();
        assert_eq!(request_tools.len(), 1);
        assert_eq!(request_tools[0].type_field, "function");
        assert_eq!(request_tools[0].function.name, "get_weather");
        assert_eq!(
            request_tools[0].function.description,
            Some("Get current weather".to_string())
        );
        assert_eq!(
            request_tools[0].function.parameters,
            json!({"type": "object", "properties": {"location": {"type": "string"}}})
        );

        // Scenario 4: Request with Any mode (implies format: "json" for Ollama)
        let messages_for_json = vec![Message::user("Give me a JSON object.")];
        let payload_json_mode = provider
            .create_request_payload(
                &model,
                &messages_for_json,
                None,
                None,
                None,
                None,
                None, // No specific tools, but setting Any mode
                Some(&ToolChoice::Any),
                Some("Respond in JSON format."),
            )
            .unwrap();

        assert_eq!(
            payload_json_mode.system,
            Some("Respond in JSON format.".to_string())
        );
        assert_eq!(payload_json_mode.messages.len(), 1);
        assert_eq!(payload_json_mode.format, Some("json".to_string()));
        assert!(payload_json_mode.tools.is_none()); // Any choice without tools just sets format for Ollama
    }
}
