use async_trait::async_trait;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex};

use crate::client::{
    GenerationMetadata, GenerationOptions, GenerationResult, LlmProvider, ToolChoice, UsageInfo,
};
use crate::error::{Error, Result};
use crate::message::{Content, Message, MessageRole, ToolCall};
use crate::model::{Model, ModelCapability, ModelFamily};

/// Configuration for the Google provider
#[derive(Debug, Clone)]
pub struct GoogleConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the API
    pub base_url: String,
    /// API version header
    pub api_version: String,
    /// Default model to use
    pub default_model: String,
    /// Project ID (for some authentication methods)
    pub project_id: Option<String>,
}

impl Default for GoogleConfig {
    fn default() -> Self {
        Self {
            api_key: env::var("GOOGLE_API_KEY").unwrap_or_default(),
            base_url: "https://generativelanguage.googleapis.com/v1".to_string(),
            api_version: "v1".to_string(),
            default_model: "gemini-1.5-pro".to_string(),
            project_id: None,
        }
    }
}

/// Implementation of the Google Generative AI API
pub struct GoogleProvider {
    /// HTTP client for making requests
    client: Client,
    /// Configuration for the provider
    config: GoogleConfig,
    /// Cache of available models
    models: Arc<Mutex<Option<Vec<Model>>>>,
}

impl Default for GoogleProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GoogleProvider {
    /// Creates a new Google provider with the default configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::GoogleProvider;
    ///
    /// let provider = GoogleProvider::new();
    /// ```
    pub fn new() -> Self {
        Self::with_config(GoogleConfig::default())
    }

    /// Creates a new Google provider with a custom configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::{GoogleProvider, GoogleConfig};
    ///
    /// let config = GoogleConfig {
    ///     api_key: "your_api_key".to_string(),
    ///     base_url: "https://generativelanguage.googleapis.com/v1".to_string(),
    ///     api_version: "v1".to_string(),
    ///     default_model: "gemini-1.5-pro".to_string(),
    ///     project_id: None,
    /// };
    ///
    /// let provider = GoogleProvider::with_config(config);
    /// ```
    pub fn with_config(config: GoogleConfig) -> Self {
        // Create HTTP client with default headers
        let mut headers = header::HeaderMap::new();

        // Add Content-Type header
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .unwrap_or_default();

        Self {
            client,
            config,
            models: Arc::new(Mutex::new(None)),
        }
    }

    /// Creates a new Google provider with an API key
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::GoogleProvider;
    ///
    /// let provider = GoogleProvider::with_api_key("your_api_key");
    /// ```
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let config = GoogleConfig {
            api_key,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Helper method to convert our message format to Google's message format
    fn convert_messages(&self, messages: &[Message]) -> Vec<GoogleMessage> {
        let mut google_messages = Vec::new();
        let mut current_role: Option<String> = None;
        let mut current_parts: Vec<GooglePart> = Vec::new();

        // Process messages in order, grouping by role
        for message in messages {
            // Get role string
            let role = match message.role {
                MessageRole::System => "system".to_string(),
                MessageRole::User => "user".to_string(),
                MessageRole::Assistant => "model".to_string(), // Google uses "model" for assistant
                MessageRole::Function => "function".to_string(),
                MessageRole::Tool => "function".to_string(), // Map tool to function for Google
            };

            // Check if we're starting a new role
            if current_role.as_ref() != Some(&role) {
                // If we have accumulated parts, create a message
                if !current_parts.is_empty() && current_role.is_some() {
                    google_messages.push(GoogleMessage {
                        role: current_role.unwrap(),
                        parts: current_parts,
                    });
                    current_parts = Vec::new();
                }
                current_role = Some(role);
            }

            // Add content to current parts
            if let Some(content) = &message.content {
                match content {
                    Content::Text(text) => {
                        current_parts.push(GooglePart::Text {
                            text: text.clone(),
                        });
                    }
                    Content::Parts(parts) => {
                        // Convert our content parts to Google's format
                        for part in parts {
                            match part {
                                crate::message::ContentPart::Text { text } => {
                                    current_parts.push(GooglePart::Text {
                                        text: text.clone(),
                                    });
                                }
                                crate::message::ContentPart::ImageUrl { image_url } => {
                                    current_parts.push(GooglePart::InlineData {
                                        inline_data: GoogleInlineData {
                                            mime_type: "image/jpeg".to_string(), // Default to JPEG
                                            data: image_url.url.clone(),
                                        },
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Add function calls or responses if present
            if let Some(function_call) = &message.function_call {
                current_parts.push(GooglePart::FunctionCall {
                    function_call: GoogleFunctionCall {
                        name: function_call.name.clone(),
                        args: serde_json::from_str(&function_call.arguments)
                            .unwrap_or_else(|_| serde_json::json!({})),
                    },
                });
            }

            if message.role == MessageRole::Function || message.role == MessageRole::Tool {
                // For function/tool responses
                if let Some(Content::Text(text)) = &message.content {
                    let name = message.name.clone().unwrap_or_default();
                    current_parts.push(GooglePart::FunctionResponse {
                        function_response: GoogleFunctionResponse {
                            name,
                            response: serde_json::from_str(text).unwrap_or_else(|_| serde_json::json!({
                                "result": text
                            })),
                        },
                    });
                }
            }

            if let Some(tool_calls) = &message.tool_calls {
                for tool_call in tool_calls {
                    current_parts.push(GooglePart::FunctionCall {
                        function_call: GoogleFunctionCall {
                            name: tool_call.function.name.clone(),
                            args: serde_json::from_str(&tool_call.function.arguments)
                                .unwrap_or_else(|_| serde_json::json!({})),
                        },
                    });
                }
            }
        }

        // Add any remaining parts
        if !current_parts.is_empty() && current_role.is_some() {
            google_messages.push(GoogleMessage {
                role: current_role.unwrap(),
                parts: current_parts,
            });
        }

        // If there are no messages, add a default user message
        if google_messages.is_empty() {
            google_messages.push(GoogleMessage {
                role: "user".to_string(),
                parts: vec![GooglePart::Text {
                    text: "Hello".to_string(),
                }],
            });
        }

        google_messages
    }

    /// Convert ToolChoice to Google's tool_choice format
    fn convert_tool_choice(&self, tool_choice: &ToolChoice) -> Option<serde_json::Value> {
        match tool_choice {
            ToolChoice::Auto => None, // Google uses null for Auto
            ToolChoice::None => Some(serde_json::json!({
                "functionCalling": {
                    "mode": "NONE"
                }
            })),
            ToolChoice::Tool(name) => Some(serde_json::json!({
                "functionCalling": {
                    "mode": "ANY",
                    "functionDeclarations": [{ "name": name }]
                }
            })),
            ToolChoice::Any => Some(serde_json::json!({
                "functionCalling": {
                    "mode": "AUTO"
                }
            })),
        }
    }

    /// Convert tool definitions to Google's tool format
    fn convert_tools(&self, tool_definitions: &[serde_json::Value]) -> Option<Vec<GoogleTool>> {
        let mut google_tools = Vec::new();

        // Create a tool declaration for each function
        let mut function_declarations = Vec::new();

        for tool_def in tool_definitions {
            // Extract function data from OpenAI-compatible format
            if let Some(function) = tool_def.get("function") {
                if let (Some(name), Some(description), Some(parameters)) = (
                    function.get("name").and_then(|n| n.as_str()),
                    function.get("description").and_then(|d| d.as_str()),
                    function.get("parameters"),
                ) {
                    function_declarations.push(GoogleFunctionDeclaration {
                        name: name.to_string(),
                        description: description.to_string(),
                        parameters: parameters.clone(),
                    });
                }
            }
        }

        if !function_declarations.is_empty() {
            google_tools.push(GoogleTool {
                function_declarations,
            });
        }

        if google_tools.is_empty() {
            None
        } else {
            Some(google_tools)
        }
    }

    /// Extract the API key from the config or URL for API calls
    fn get_api_url(&self, endpoint: &str, model: &str) -> String {
        if !self.config.api_key.is_empty() {
            format!("{}/{}/{}?key={}", self.config.base_url, model, endpoint, self.config.api_key)
        } else {
            format!("{}/{}/{}", self.config.base_url, model, endpoint)
        }
    }
}

#[async_trait]
impl LlmProvider for GoogleProvider {
    async fn list_models(&self) -> Result<Vec<Model>> {
        // Check cache first
        {
            let models = self.models.lock().unwrap();
            if let Some(models) = &*models {
                return Ok(models.clone());
            }
        }

        // Define the supported Gemini models
        let models = vec![
            Model::new(
                "gemini-1.5-pro",
                "Gemini 1.5 Pro",
                ModelFamily::Gemini,
                vec![
                    ModelCapability::ChatCompletion,
                    ModelCapability::TextGeneration,
                    ModelCapability::Vision,
                    ModelCapability::ToolCalling,
                ],
                "google",
            )
            .with_context_window(1000000),
            Model::new(
                "gemini-1.5-flash",
                "Gemini 1.5 Flash",
                ModelFamily::Gemini,
                vec![
                    ModelCapability::ChatCompletion,
                    ModelCapability::TextGeneration,
                    ModelCapability::Vision,
                    ModelCapability::ToolCalling,
                ],
                "google",
            )
            .with_context_window(1000000),
            Model::new(
                "gemini-1.5-flash-latest",
                "Gemini 1.5 Flash Latest",
                ModelFamily::Gemini,
                vec![
                    ModelCapability::ChatCompletion,
                    ModelCapability::TextGeneration,
                    ModelCapability::Vision,
                    ModelCapability::ToolCalling,
                ],
                "google",
            )
            .with_context_window(1000000),
            Model::new(
                "gemini-1.0-pro",
                "Gemini 1.0 Pro",
                ModelFamily::Gemini,
                vec![
                    ModelCapability::ChatCompletion,
                    ModelCapability::TextGeneration,
                    ModelCapability::Vision,
                    ModelCapability::ToolCalling,
                ],
                "google",
            )
            .with_context_window(32768),
            Model::new(
                "gemini-1.0-pro-vision",
                "Gemini 1.0 Pro Vision",
                ModelFamily::Gemini,
                vec![
                    ModelCapability::ChatCompletion,
                    ModelCapability::TextGeneration,
                    ModelCapability::Vision,
                ],
                "google",
            )
            .with_context_window(32768),
            Model::new(
                "gemini-1.0-ultra",
                "Gemini 1.0 Ultra",
                ModelFamily::Gemini,
                vec![
                    ModelCapability::ChatCompletion,
                    ModelCapability::TextGeneration,
                    ModelCapability::Vision,
                    ModelCapability::ToolCalling,
                ],
                "google",
            )
            .with_context_window(32768),
        ];

        // Cache the models
        {
            let mut models_cache = self.models.lock().unwrap();
            *models_cache = Some(models.clone());
        }

        Ok(models)
    }

    async fn generate(
        &self,
        model: &str,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        // Validate model
        let models = self.list_models().await?;
        if !models.iter().any(|m| m.id == model) {
            return Err(Error::UnsupportedModel(format!(
                "Model '{}' not found in Google provider",
                model
            )));
        }

        // Convert messages to Google format
        let google_messages = self.convert_messages(messages);

        // Prepare the request payload
        let mut request_payload = HashMap::new();
        request_payload.insert("contents", serde_json::json!(google_messages));

        // Add generation configuration if parameters are provided
        let mut generation_config = HashMap::new();
        let mut add_generation_config = false;

        // Add temperature if specified
        if let Some(temperature) = options.temperature {
            generation_config.insert("temperature", serde_json::json!(temperature));
            add_generation_config = true;
        }

        // Add top_p if specified
        if let Some(top_p) = options.top_p {
            generation_config.insert("topP", serde_json::json!(top_p));
            add_generation_config = true;
        }

        // Add max_tokens if specified
        if let Some(max_tokens) = options.max_tokens {
            generation_config.insert("maxOutputTokens", serde_json::json!(max_tokens));
            add_generation_config = true;
        }

        // Add stop sequences if specified
        if let Some(stop) = &options.stop {
            generation_config.insert("stopSequences", serde_json::json!(stop));
            add_generation_config = true;
        }

        if add_generation_config {
            request_payload.insert("generationConfig", serde_json::json!(generation_config));
        }

        // Add safety settings
        // Google has different safety settings that we're leaving at their defaults

        // Tool functionality has been temporarily removed
        if false { // Never executes
            let tool_defs = Vec::<serde_json::Value>::new();
            if let Some(google_tools) = self.convert_tools(&tool_defs) {
                request_payload.insert("tools", serde_json::json!(google_tools));
            }
        }

        // Add tool_choice if specified
        if let Some(tool_choice) = &options.tool_choice {
            if let Some(google_tool_choice) = self.convert_tool_choice(tool_choice) {
                request_payload.insert("toolConfig", google_tool_choice);
            }
        }

        // Add any extra parameters
        for (key, value) in &options.extra_params {
            request_payload.insert(key, value.clone());
        }

        // Make the API request
        let url = self.get_api_url("generateContent", model);
        let response = self
            .client
            .post(&url)
            .json(&request_payload)
            .send()
            .await
            .map_err(Error::Request)?;

        // Check for success
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            return match status.as_u16() {
                401 => Err(Error::Authentication(format!(
                    "Google API authentication failed: {}",
                    error_text
                ))),
                403 => Err(Error::Authentication(format!(
                    "Google API authentication failed (invalid or missing API key): {}",
                    error_text
                ))),
                429 => Err(Error::RateLimit(format!(
                    "Google API rate limit exceeded: {}",
                    error_text
                ))),
                _ => Err(Error::Other(format!(
                    "Google API error ({}): {}",
                    status, error_text
                ))),
            };
        }

        // Parse the response
        let response_text = response.text().await.map_err(Error::Request)?;

        let google_response: GoogleResponse =
            serde_json::from_str(&response_text).map_err(Error::Serialization)?;

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
        if let Some(candidate) = google_response.candidates.first() {
            if let Some(content) = &candidate.content {
                // Process content parts
                for part in &content.parts {
                    match part {
                        GoogleResponsePart::Text { text } => {
                            message.content = Some(Content::Text(text.clone()));
                        }
                        GoogleResponsePart::FunctionCall { function_call } => {
                            // Get or create tool_calls
                            let mut tool_calls = message.tool_calls.unwrap_or_default();

                            // Add the new tool call
                            let tool_call = ToolCall {
                                id: format!("call_{}", tool_calls.len() + 1), // Generate an ID
                                tool_type: "function".to_string(),
                                function: crate::message::FunctionCall {
                                    name: function_call.name.clone(),
                                    arguments: serde_json::to_string(&function_call.args)
                                        .unwrap_or_default(),
                                },
                            };

                            tool_calls.push(tool_call);
                            message.tool_calls = Some(tool_calls);
                            
                            // For tool calls, content is typically not included
                            message.content = None;
                        }
                    }
                }
            }
        }

        // Create usage information if available
        let usage = if let Some(usage_metadata) = google_response.usage_metadata {
            Some(UsageInfo {
                prompt_tokens: usage_metadata.prompt_token_count,
                completion_tokens: usage_metadata.candidates_token_count,
                total_tokens: usage_metadata.prompt_token_count + usage_metadata.candidates_token_count,
            })
        } else {
            None
        };

        // Create generation metadata
        let metadata = GenerationMetadata {
            id: google_response.candidates.first().map_or_else(
                || "google-response".to_string(),
                |c| c.finish_reason.clone().unwrap_or_else(|| "google-response".to_string()),
            ),
            model: model.to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            usage,
            extra: HashMap::new(),
        };

        Ok(GenerationResult { message, metadata })
    }

    async fn supports_tool_calling(&self, model: &str) -> Result<bool> {
        // Most Gemini models support tool calling
        match model {
            "gemini-1.5-pro" | "gemini-1.5-flash" | "gemini-1.5-flash-latest" | "gemini-1.0-pro" | "gemini-1.0-ultra" => Ok(true),
            _ => Ok(false),
        }
    }

    async fn has_capability(&self, model: &str, capability: ModelCapability) -> Result<bool> {
        let models = self.list_models().await?;

        if let Some(model) = models.iter().find(|m| m.id == model) {
            Ok(model.has_capability(capability))
        } else {
            Err(Error::UnsupportedModel(format!(
                "Model '{}' not found",
                model
            )))
        }
    }
}

/// Represents a message in the Google API format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleMessage {
    /// The role of the message sender
    pub role: String,
    /// The parts of the message
    pub parts: Vec<GooglePart>,
}

/// Represents a part of a message in the Google API format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum GooglePart {
    /// Text part
    #[serde(rename = "text")]
    Text {
        /// The text content
        text: String,
    },
    /// Inline data (for images)
    #[serde(rename = "inlineData")]
    InlineData {
        /// The inline data
        inline_data: GoogleInlineData,
    },
    /// Function call (for tool usage)
    #[serde(rename = "functionCall")]
    FunctionCall {
        /// The function call
        function_call: GoogleFunctionCall,
    },
    /// Function response (for tool results)
    #[serde(rename = "functionResponse")]
    FunctionResponse {
        /// The function response
        function_response: GoogleFunctionResponse,
    },
}

/// Represents inline data (e.g., an image) in the Google API format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleInlineData {
    /// The MIME type of the data
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    /// The data (typically base64-encoded)
    pub data: String,
}

/// Represents a function call in the Google API format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleFunctionCall {
    /// The name of the function
    pub name: String,
    /// The arguments to the function
    pub args: serde_json::Value,
}

/// Represents a function response in the Google API format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleFunctionResponse {
    /// The name of the function
    pub name: String,
    /// The response from the function
    pub response: serde_json::Value,
}

/// Represents a tool in the Google API format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleTool {
    /// Function declarations for the tool
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<GoogleFunctionDeclaration>,
}

/// Represents a function declaration in the Google API format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleFunctionDeclaration {
    /// The name of the function
    pub name: String,
    /// The description of the function
    pub description: String,
    /// The parameters of the function
    pub parameters: serde_json::Value,
}

/// Represents a usage metadata in the Google API format
#[derive(Debug, Serialize, Deserialize)]
struct GoogleUsageMetadata {
    /// Number of tokens in the prompt
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: u32,
    /// Number of tokens in the candidates
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: u32,
    /// Total number of tokens
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: u32,
}

/// Represents a response from the Google API
#[derive(Debug, Serialize, Deserialize)]
struct GoogleResponse {
    /// The candidates generated
    pub candidates: Vec<GoogleCandidate>,
    /// The prompt feedback
    #[serde(rename = "promptFeedback")]
    pub prompt_feedback: Option<GooglePromptFeedback>,
    /// Usage metadata
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GoogleUsageMetadata>,
}

/// Represents a candidate in the Google API response
#[derive(Debug, Serialize, Deserialize)]
struct GoogleCandidate {
    /// The content of the candidate
    pub content: Option<GoogleContent>,
    /// The reason why generation finished
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
    /// The safety ratings
    #[serde(rename = "safetyRatings")]
    pub safety_ratings: Option<Vec<GoogleSafetyRating>>,
}

/// Represents the content of a candidate in the Google API response
#[derive(Debug, Serialize, Deserialize)]
struct GoogleContent {
    /// The role of the content
    pub role: String,
    /// The parts of the content
    pub parts: Vec<GoogleResponsePart>,
}

/// Represents a part of a response in the Google API format
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum GoogleResponsePart {
    /// Text part
    #[serde(rename = "text")]
    Text {
        /// The text content
        text: String,
    },
    /// Function call (for tool usage)
    #[serde(rename = "functionCall")]
    FunctionCall {
        /// The function call
        function_call: GoogleFunctionCall,
    },
}

/// Represents a safety rating in the Google API response
#[derive(Debug, Serialize, Deserialize)]
struct GoogleSafetyRating {
    /// The category of the safety rating
    pub category: String,
    /// The probability of the content being harmful
    pub probability: String,
}

/// Represents a prompt feedback in the Google API response
#[derive(Debug, Serialize, Deserialize)]
struct GooglePromptFeedback {
    /// The safety ratings
    #[serde(rename = "safetyRatings")]
    pub safety_ratings: Vec<GoogleSafetyRating>,
    /// Whether the prompt was blocked
    #[serde(rename = "blockReason")]
    pub block_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::MessageRole;
    use serde_json::json;

    // Helper function to create a mock Google API response
    fn mock_response(text_response: &str) -> String {
        format!(
            r#"{{
                "candidates": [
                    {{
                        "content": {{
                            "parts": [
                                {{
                                    "text": "{}"
                                }}
                            ],
                            "role": "model"
                        }},
                        "finishReason": "STOP",
                        "safetyRatings": []
                    }}
                ],
                "usageMetadata": {{
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30
                }}
            }}"#,
            text_response
        )
    }

    // Helper function to create a mock Google API function call response
    fn mock_function_call_response(function_name: &str, args: serde_json::Value) -> String {
        format!(
            r#"{{
                "candidates": [
                    {{
                        "content": {{
                            "parts": [
                                {{
                                    "functionCall": {{
                                        "name": "{}",
                                        "args": {}
                                    }}
                                }}
                            ],
                            "role": "model"
                        }},
                        "finishReason": "STOP",
                        "safetyRatings": []
                    }}
                ],
                "usageMetadata": {{
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30
                }}
            }}"#,
            function_name,
            args
        )
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = GoogleProvider::new();
        let models = provider.list_models().await.unwrap();

        // Make sure we've got the expected models
        assert!(models.iter().any(|m| m.id == "gemini-1.5-pro"));
        assert!(models.iter().any(|m| m.id == "gemini-1.0-pro"));

        // Check that they have the right capabilities
        let pro = models
            .iter()
            .find(|m| m.id == "gemini-1.5-pro")
            .unwrap();
        assert!(pro.has_capability(ModelCapability::ChatCompletion));
        assert!(pro.has_capability(ModelCapability::Vision));
        assert!(pro.has_capability(ModelCapability::ToolCalling));

        // Check model family
        assert_eq!(pro.family, ModelFamily::Gemini);
    }

    #[tokio::test]
    async fn test_convert_messages() {
        let provider = GoogleProvider::new();

        // Test basic message conversion
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello, who are you?"),
        ];

        let google_messages = provider.convert_messages(&messages);

        // Should have system and user messages
        assert_eq!(google_messages.len(), 2);
        assert_eq!(google_messages[0].role, "system");
        assert_eq!(google_messages[1].role, "user");

        if let GooglePart::Text { text } = &google_messages[1].parts[0] {
            assert_eq!(text, "Hello, who are you?");
        } else {
            panic!("Expected text part");
        }

        // Test conversion with assistant message
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello, who are you?"),
            Message::assistant("I am Gemini, an AI assistant."),
        ];

        let google_messages = provider.convert_messages(&messages);

        // Should have system, user, and model messages
        assert_eq!(google_messages.len(), 3);
        assert_eq!(google_messages[0].role, "system");
        assert_eq!(google_messages[1].role, "user");
        assert_eq!(google_messages[2].role, "model");

        // Test that function calls are properly converted
        let function_call = crate::message::FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"New York"}"#.to_string(),
        };

        let messages = vec![
            Message::user("What's the weather in New York?"),
            Message {
                role: MessageRole::Assistant,
                content: None,
                name: None,
                function_call: Some(function_call),
                tool_calls: None,
                tool_call_id: None,
                metadata: HashMap::new(),
            },
        ];

        let google_messages = provider.convert_messages(&messages);

        // Should have user and model messages
        assert_eq!(google_messages.len(), 2);
        assert_eq!(google_messages[0].role, "user");
        assert_eq!(google_messages[1].role, "model");

        // The model message should have a function call
        match &google_messages[1].parts[0] {
            GooglePart::FunctionCall { function_call } => {
                assert_eq!(function_call.name, "get_weather");
                assert_eq!(function_call.args["location"], "New York");
            }
            _ => panic!("Expected function call part"),
        }
    }

    #[tokio::test]
    async fn test_supports_tool_calling() {
        let provider = GoogleProvider::new();

        // Models that support tool calling
        assert!(provider
            .supports_tool_calling("gemini-1.5-pro")
            .await
            .unwrap());
        assert!(provider
            .supports_tool_calling("gemini-1.0-pro")
            .await
            .unwrap());

        // Model that doesn't support tool calling
        assert!(!provider
            .supports_tool_calling("gemini-1.0-pro-vision")
            .await
            .unwrap());
    }

    // #[tokio::test]
    // async fn test_register_and_execute_tool() {
    //     // Test commented out as tool functionality has been temporarily removed
    // }
}