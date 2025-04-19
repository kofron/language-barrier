use crate::error::{Error, Result};
use crate::message::{Content, ContentPart, Message, MessageRole};
use crate::provider::HTTPProvider;
use crate::{Chat, ModelInfo};
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
    /// use language_barrier::provider::gemini::GeminiProvider;
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
    /// use language_barrier::provider::gemini::{GeminiProvider, GeminiConfig};
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
        trace!("Message content: {:?}", message.content);

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
        let mut current_role: Option<MessageRole> = None;
        let mut current_parts: Vec<GeminiPart> = Vec::new();

        for msg in &chat.history {
            // If role changes, finish the current content and start a new one
            if current_role.is_some() && current_role != Some(msg.role) {
                if !current_parts.is_empty() {
                    let role = match current_role {
                        Some(MessageRole::User) => Some("user".to_string()),
                        Some(MessageRole::Assistant) => Some("model".to_string()),
                        _ => None,
                    };

                    contents.push(GeminiContent {
                        parts: std::mem::take(&mut current_parts),
                        role,
                    });
                }
            }

            current_role = Some(msg.role);

            // Convert message content to parts
            match &msg.content {
                Some(Content::Text(text)) => {
                    current_parts.push(GeminiPart::text(text.clone()));
                }
                Some(Content::Parts(parts)) => {
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
                None => {}
            }
        }

        // Add any remaining parts
        if !current_parts.is_empty() {
            let role = match current_role {
                Some(MessageRole::User) => Some("user".to_string()),
                Some(MessageRole::Assistant) => Some("model".to_string()),
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

        // Create the request
        debug!("Creating GeminiRequest");
        let request = GeminiRequest {
            contents,
            system_instruction,
            generation_config,
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
}

impl GeminiPart {
    /// Create a new text part
    fn text(text: String) -> Self {
        GeminiPart {
            text: Some(text),
            inline_data: None,
        }
    }

    /// Create a new inline data part
    fn inline_data(data: String, mime_type: String) -> Self {
        GeminiPart {
            text: None,
            inline_data: Some(GeminiInlineData { data, mime_type }),
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

        // Convert the parts to content parts
        let content_parts: Vec<ContentPart> = candidate
            .content
            .parts
            .iter()
            .map(|part| {
                if let Some(text) = &part.text {
                    ContentPart::text(text.clone())
                } else if let Some(inline_data) = &part.inline_data {
                    // Just convert to text representation for now
                    ContentPart::text(format!(
                        "[Image: {} ({})]",
                        inline_data.data, inline_data.mime_type
                    ))
                } else {
                    // Empty part as fallback
                    ContentPart::text("")
                }
            })
            .collect();

        let content = if content_parts.len() == 1 {
            // If there's only one text part, use simple Text content
            match &content_parts[0] {
                ContentPart::Text { text } => Some(Content::Text(text.clone())),
                _ => Some(Content::Parts(content_parts)),
            }
        } else {
            // Multiple content parts
            Some(Content::Parts(content_parts))
        };

        let mut msg = Message {
            role: MessageRole::Assistant,
            content,
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
            metadata: Default::default(),
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
