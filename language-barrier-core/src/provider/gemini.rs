use crate::error::{Error, Result};
use crate::message::{Content, ContentPart, Message};
use crate::provider::HTTPProvider;
use crate::{Chat, LlmToolInfo, ModelInfo};
use reqwest::{Method, Request, Url};
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{debug, error, info, instrument, trace, warn};

/// Configuration for the Gemini provider
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the API
    pub base_url: String,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: env::var("GEMINI_API_KEY").unwrap_or_default(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }
}

/// Implementation of the Gemini provider
#[derive(Debug, Clone)]
pub struct GeminiProvider {
    /// Configuration for the provider
    config: GeminiConfig,
}

impl GeminiProvider {
    /// Creates a new GeminiProvider with default configuration
    ///
    /// This method will use the GEMINI_API_KEY environment variable for authentication.
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier_core::provider::gemini::GeminiProvider;
    ///
    /// let provider = GeminiProvider::new();
    /// ```
    #[instrument(level = "debug")]
    pub fn new() -> Self {
        info!("Creating new GeminiProvider with default configuration");
        let config = GeminiConfig::default();
        debug!("API key set: {}", !config.api_key.is_empty());
        debug!("Base URL: {}", config.base_url);

        Self { config }
    }

    /// Creates a new GeminiProvider with custom configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier_core::provider::gemini::{GeminiProvider, GeminiConfig};
    ///
    /// let config = GeminiConfig {
    ///     api_key: "your-api-key".to_string(),
    ///     base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
    /// };
    ///
    /// let provider = GeminiProvider::with_config(config);
    /// ```
    #[instrument(skip(config), level = "debug")]
    pub fn with_config(config: GeminiConfig) -> Self {
        info!("Creating new GeminiProvider with custom configuration");
        debug!("API key set: {}", !config.api_key.is_empty());
        debug!("Base URL: {}", config.base_url);

        Self { config }
    }
}

impl Default for GeminiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: ModelInfo + GeminiModelInfo> HTTPProvider<M> for GeminiProvider {
    fn accept(&self, chat: Chat<M>) -> Result<Request> {
        info!("Creating request for Gemini model: {:?}", chat.model);
        debug!("Messages in chat history: {}", chat.history.len());

        let model_id = chat.model.gemini_model_id();
        let url_str = format!(
            "{}/models/{}:generateContent?key={}",
            self.config.base_url, model_id, self.config.api_key
        );

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
        let content_type_header = match "application/json".parse() {
            Ok(header) => header,
            Err(e) => {
                error!("Failed to set content type: {}", e);
                return Err(Error::Other("Failed to set content type".into()));
            }
        };

        request
            .headers_mut()
            .insert("Content-Type", content_type_header);

        trace!("Request headers set: {:#?}", request.headers());

        // Create the request payload
        debug!("Creating request payload");
        let payload = match self.create_request_payload(&chat) {
            Ok(payload) => {
                debug!("Request payload created successfully");
                trace!("Number of contents: {}", payload.contents.len());
                trace!(
                    "System instruction present: {}",
                    payload.system_instruction.is_some()
                );
                trace!(
                    "Generation config present: {}",
                    payload.generation_config.is_some()
                );
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
        info!("Parsing response from Gemini API");
        trace!("Raw response: {}", raw_response_text);

        // First try to parse as an error response
        if let Ok(error_response) = serde_json::from_str::<GeminiErrorResponse>(&raw_response_text)
        {
            if let Some(error) = error_response.error {
                error!("Gemini API returned an error: {}", error.message);
                return Err(Error::ProviderUnavailable(error.message));
            }
        }

        // If not an error, parse as a successful response
        debug!("Deserializing response JSON");
        let gemini_response = match serde_json::from_str::<GeminiResponse>(&raw_response_text) {
            Ok(response) => {
                debug!("Response deserialized successfully");
                if !response.candidates.is_empty() {
                    debug!(
                        "Content parts: {}",
                        response.candidates[0].content.parts.len()
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
        debug!("Converting Gemini response to Message");
        let message = Message::from(&gemini_response);

        info!("Response parsed successfully");
        trace!("Response message processed");

        Ok(message)
    }
}

// Trait to get Gemini-specific model IDs
pub trait GeminiModelInfo {
    fn gemini_model_id(&self) -> String;
}

impl GeminiProvider {
    /// Creates a request payload from a Chat object
    ///
    /// This method converts the Chat's messages and settings into a Gemini-specific
    /// format for the API request.
    #[instrument(skip(self, chat), level = "debug")]
    fn create_request_payload<M: ModelInfo + GeminiModelInfo>(
        &self,
        chat: &Chat<M>,
    ) -> Result<GeminiRequest> {
        info!("Creating request payload for chat with Gemini model");
        debug!("System prompt length: {}", chat.system_prompt.len());
        debug!("Messages in history: {}", chat.history.len());
        debug!("Max output tokens: {}", chat.max_output_tokens);

        // Convert system prompt if present
        let system_instruction = if !chat.system_prompt.is_empty() {
            debug!("Including system prompt in request");
            trace!("System prompt: {}", chat.system_prompt);
            Some(GeminiContent {
                parts: vec![GeminiPart::text(chat.system_prompt.clone())],
                role: None,
            })
        } else {
            debug!("No system prompt provided");
            None
        };

        // Convert messages to contents
        debug!("Converting messages to Gemini format");
        let mut contents: Vec<GeminiContent> = Vec::new();
        let mut current_role_str: Option<&'static str> = None;
        let mut current_parts: Vec<GeminiPart> = Vec::new();

        for msg in &chat.history {
            // Get the current role string
            let msg_role_str = msg.role_str();

            // If role changes, finish the current content and start a new one
            if current_role_str.is_some()
                && current_role_str != Some(msg_role_str)
                && !current_parts.is_empty()
            {
                let role = match current_role_str {
                    Some("user") => Some("user".to_string()),
                    Some("assistant") => Some("model".to_string()),
                    _ => None,
                };

                contents.push(GeminiContent {
                    parts: std::mem::take(&mut current_parts),
                    role,
                });
            }

            current_role_str = Some(msg_role_str);

            // Convert message content to parts based on the message variant
            match msg {
                Message::System { content, .. } => {
                    current_parts.push(GeminiPart::text(content.clone()));
                }
                Message::User { content, .. } => match content {
                    Content::Text(text) => {
                        current_parts.push(GeminiPart::text(text.clone()));
                    }
                    Content::Parts(parts) => {
                        for part in parts {
                            match part {
                                ContentPart::Text { text } => {
                                    current_parts.push(GeminiPart::text(text.clone()));
                                }
                                ContentPart::ImageUrl { image_url } => {
                                    current_parts.push(GeminiPart::inline_data(
                                        image_url.url.clone(),
                                        "image/jpeg".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                },
                Message::Assistant { content, .. } => {
                    if let Some(content_data) = content {
                        match content_data {
                            Content::Text(text) => {
                                current_parts.push(GeminiPart::text(text.clone()));
                            }
                            Content::Parts(parts) => {
                                for part in parts {
                                    match part {
                                        ContentPart::Text { text } => {
                                            current_parts.push(GeminiPart::text(text.clone()));
                                        }
                                        ContentPart::ImageUrl { image_url } => {
                                            current_parts.push(GeminiPart::inline_data(
                                                image_url.url.clone(),
                                                "image/jpeg".to_string(),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Message::Tool {
                    tool_call_id,
                    content,
                    ..
                } => {
                    // For Gemini, include both the tool call ID and the content
                    current_parts.push(GeminiPart::text(format!(
                        "Tool result for call {}: {}",
                        tool_call_id, content
                    )));
                }
            }
        }

        // Add any remaining parts
        if !current_parts.is_empty() {
            let role = match current_role_str {
                Some("user") => Some("user".to_string()),
                Some("assistant") => Some("model".to_string()),
                _ => None,
            };

            contents.push(GeminiContent {
                parts: current_parts,
                role,
            });
        }

        debug!("Converted {} contents for the request", contents.len());

        // Create generation config
        let generation_config = Some(GeminiGenerationConfig {
            max_output_tokens: Some(chat.max_output_tokens),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
        });

        // Convert tool descriptions if a tool registry is provided
        let tools = chat.tools.as_ref().map(|tools| {
            vec![GeminiTool {
                function_declarations: tools
                    .iter()
                    .map(GeminiFunctionDeclaration::from)
                    .collect(),
            }]
        });
        
        // Note: For Gemini, tool_choice is handled through the API's behavior
        // We don't modify the tools list based on the choice

        // Create the tool_config setting based on Google's specific format
        let tool_config = if let Some(choice) = &chat.tool_choice {
            match choice {
                crate::tool::ToolChoice::Auto => Some(GeminiToolConfig {
                    function_calling_config: GeminiFunctionCallingConfig {
                        mode: "auto".to_string(),
                        allowed_function_names: None,
                    },
                }),
                crate::tool::ToolChoice::Any => Some(GeminiToolConfig {
                    function_calling_config: GeminiFunctionCallingConfig {
                        mode: "any".to_string(),
                        allowed_function_names: None,
                    },
                }),
                crate::tool::ToolChoice::None => Some(GeminiToolConfig {
                    function_calling_config: GeminiFunctionCallingConfig {
                        mode: "none".to_string(),
                        allowed_function_names: None,
                    },
                }),
                crate::tool::ToolChoice::Specific(name) => Some(GeminiToolConfig {
                    function_calling_config: GeminiFunctionCallingConfig {
                        mode: "auto".to_string(), // Use mode auto with specific allowed function
                        allowed_function_names: Some(vec![name.clone()]),
                    },
                }),
            }
        } else if tools.is_some() {
            // Default to auto if tools are present but no choice specified
            Some(GeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: "auto".to_string(),
                    allowed_function_names: None,
                },
            })
        } else {
            None
        };

        // Create the request
        debug!("Creating GeminiRequest");
        let request = GeminiRequest {
            contents,
            system_instruction,
            generation_config,
            tools,
            tool_config,
        };

        info!("Request payload created successfully");
        Ok(request)
    }
}

/// Represents a content part in Gemini API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiPart {
    /// The text content (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// The inline data (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<GeminiInlineData>,

    /// The function call (optional)
    #[serde(skip_serializing_if = "Option::is_none", rename = "functionCall")]
    pub function_call: Option<GeminiFunctionCall>,
}

/// Represents a function call in the Gemini API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionCall {
    /// The name of the function
    pub name: String,
    /// The arguments as a JSON Value
    pub args: serde_json::Value,
}

impl GeminiPart {
    /// Create a new text part
    fn text(text: String) -> Self {
        GeminiPart {
            text: Some(text),
            inline_data: None,
            function_call: None,
        }
    }

    /// Create a new inline data part
    fn inline_data(data: String, mime_type: String) -> Self {
        GeminiPart {
            text: None,
            inline_data: Some(GeminiInlineData { data, mime_type }),
            function_call: None,
        }
    }
}

/// Represents inline data in Gemini API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiInlineData {
    /// The data (base64 encoded)
    pub data: String,
    /// The MIME type
    pub mime_type: String,
}

/// Represents a content object in Gemini API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiContent {
    /// The parts of the content
    pub parts: Vec<GeminiPart>,
    /// The role of the content (user, model, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

/// Represents a generation config in Gemini API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiGenerationConfig {
    /// The maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
    /// The temperature (randomness) of the generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// The top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// The top-k sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Sequences that will stop generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

/// Represents a function declaration in the Gemini API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionDeclaration {
    /// The name of the function
    pub name: String,
    /// The description of the function
    pub description: String,
    /// The parameters schema
    pub parameters: serde_json::Value,
}

impl From<&LlmToolInfo> for GeminiFunctionDeclaration {
    fn from(value: &LlmToolInfo) -> Self {
        GeminiFunctionDeclaration {
            name: value.name.clone(),
            description: value.description.clone(),
            parameters: value.parameters.clone(),
        }
    }
}

/// Represents a function in the Gemini API format (tools are called functions in Gemini)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunction {
    /// The name of the function
    pub name: String,
    /// The description of the function
    pub description: String,
    /// The parameters definition
    pub parameters: serde_json::Value,
}

/// Represents a tool in the Gemini API format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiTool {
    /// The function declaration
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}

// Gemini uses GeminiToolConfig instead of a direct tool_choice field

/// Tool config for Gemini API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiToolConfig {
    /// Function calling configuration
    #[serde(rename = "function_calling_config")]
    pub function_calling_config: GeminiFunctionCallingConfig,
}

/// Function calling config for Gemini API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionCallingConfig {
    /// The mode (auto, any, none)
    pub mode: String,
    /// List of specific function names that are allowed (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Represents a request to the Gemini API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiRequest {
    /// The contents to send to the model
    pub contents: Vec<GeminiContent>,
    /// The system instruction (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    /// The generation config (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    /// The tools (functions) available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    /// Tool configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<GeminiToolConfig>,
}

/// Represents a response from the Gemini API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiResponse {
    /// The candidates (typically one)
    pub candidates: Vec<GeminiCandidate>,
    /// Usage information (may not be present in all responses)
    #[serde(rename = "usageMetadata", skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
    /// The model version
    #[serde(rename = "modelVersion", skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
}

/// Represents a candidate in a Gemini response
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiCandidate {
    /// The content of the candidate
    pub content: GeminiContent,
    /// The finish reason (using camelCase as in the API)
    #[serde(skip_serializing_if = "Option::is_none", rename = "finishReason")]
    pub finish_reason: Option<String>,
    /// The index of the candidate (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<i32>,
    /// The average log probability (optional)
    #[serde(skip_serializing_if = "Option::is_none", rename = "avgLogprobs")]
    pub avg_logprobs: Option<f64>,
}

/// Represents token details for a specific modality
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiTokenDetails {
    /// The modality (TEXT, IMAGE, etc.)
    pub modality: String,
    /// The token count for this modality
    #[serde(rename = "tokenCount")]
    pub token_count: u32,
}

/// Represents usage metadata in a Gemini response
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiUsageMetadata {
    /// Token count in the prompt
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: u32,
    /// Token count in the response
    #[serde(rename = "candidatesTokenCount", default)]
    pub candidates_token_count: u32,
    /// Total token count
    #[serde(rename = "totalTokenCount", default)]
    pub total_token_count: u32,
    /// Detailed token breakdown for the prompt
    #[serde(
        rename = "promptTokensDetails",
        skip_serializing_if = "Option::is_none"
    )]
    pub prompt_tokens_details: Option<Vec<GeminiTokenDetails>>,
    /// Detailed token breakdown for the candidates
    #[serde(
        rename = "candidatesTokensDetails",
        skip_serializing_if = "Option::is_none"
    )]
    pub candidates_tokens_details: Option<Vec<GeminiTokenDetails>>,
}

/// Represents an error response from the Gemini API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiErrorResponse {
    /// The error details
    pub error: Option<GeminiError>,
}

/// Represents an error from the Gemini API
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiError {
    /// The error code
    pub code: i32,
    /// The error message
    pub message: String,
    /// The error status
    pub status: String,
}

/// Convert from Gemini's response to our message format
impl From<&GeminiResponse> for Message {
    fn from(response: &GeminiResponse) -> Self {
        // Check if we have candidates
        if response.candidates.is_empty() {
            return Message::assistant("No response generated");
        }

        // Get the first candidate
        let candidate = &response.candidates[0];

        // Extract text content and tool calls separately
        let mut text_content_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_call_id_counter = 0;

        // Process each part of the response
        for part in &candidate.content.parts {
            // Handle function calls
            if let Some(function_call) = &part.function_call {
                tool_call_id_counter += 1;
                let tool_id = format!("gemini_call_{}", tool_call_id_counter);

                let args_str =
                    serde_json::to_string(&function_call.args).unwrap_or_else(|_| "{}".to_string());

                let tool_call = crate::message::ToolCall {
                    id: tool_id,
                    tool_type: "function".to_string(),
                    function: crate::message::Function {
                        name: function_call.name.clone(),
                        arguments: args_str,
                    },
                };

                tool_calls.push(tool_call);
            }

            // Handle text content
            if let Some(text) = &part.text {
                text_content_parts.push(ContentPart::text(text.clone()));
            } else if let Some(inline_data) = &part.inline_data {
                // Just convert to text representation for now
                text_content_parts.push(ContentPart::text(format!(
                    "[Image: {} ({})]",
                    inline_data.data, inline_data.mime_type
                )));
            }
        }

        // Create the content
        let content = if text_content_parts.len() == 1 {
            // If there's only one text part, use simple Text content
            match &text_content_parts[0] {
                ContentPart::Text { text } => Some(Content::Text(text.clone())),
                _ => Some(Content::Parts(text_content_parts)),
            }
        } else if !text_content_parts.is_empty() {
            // Multiple content parts
            Some(Content::Parts(text_content_parts))
        } else {
            // No text content, may have only function calls
            None
        };

        // Create a new assistant message with appropriate content and tool calls
        let mut msg = if !tool_calls.is_empty() {
            // If we have tool calls
            Message::Assistant {
                content,
                tool_calls,
                metadata: Default::default(),
            }
        } else if let Some(Content::Text(text)) = content {
            // Simple text response
            Message::assistant(text)
        } else {
            // Other content types (multipart or none)
            Message::Assistant {
                content,
                tool_calls: Vec::new(),
                metadata: Default::default(),
            }
        };

        // Add usage info if available
        if let Some(usage) = &response.usage_metadata {
            msg = msg.with_metadata(
                "prompt_tokens",
                serde_json::Value::Number(usage.prompt_token_count.into()),
            );
            msg = msg.with_metadata(
                "completion_tokens",
                serde_json::Value::Number(usage.candidates_token_count.into()),
            );
            msg = msg.with_metadata(
                "total_tokens",
                serde_json::Value::Number(usage.total_token_count.into()),
            );
        }

        msg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests will be implemented as we get more information about the API
    #[test]
    fn test_gemini_part_serialization() {
        let text_part = GeminiPart::text("Hello, world!".to_string());
        let serialized = serde_json::to_string(&text_part).unwrap();
        let expected = r#"{"text":"Hello, world!"}"#;
        assert_eq!(serialized, expected);

        let inline_data_part =
            GeminiPart::inline_data("base64data".to_string(), "image/jpeg".to_string());
        let serialized = serde_json::to_string(&inline_data_part).unwrap();
        let expected = r#"{"inline_data":{"data":"base64data","mime_type":"image/jpeg"}}"#;
        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_error_response_parsing() {
        let error_json = r#"{
            "error": {
                "code": 400,
                "message": "Invalid JSON payload received.",
                "status": "INVALID_ARGUMENT"
            }
        }"#;

        let error_response: GeminiErrorResponse = serde_json::from_str(error_json).unwrap();
        assert!(error_response.error.is_some());
        let error = error_response.error.unwrap();
        assert_eq!(error.code, 400);
        assert_eq!(error.status, "INVALID_ARGUMENT");
    }
}
