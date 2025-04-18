use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::message::{Content, FunctionCall, Message};
use crate::model::{Model, ModelCapability};

/// Specifies how a model should choose which tools to use
#[derive(Debug, Clone)]
pub enum ToolChoice {
    /// The model will automatically decide which tools to use, if any
    Auto,
    /// The model should not use any tools
    None,
    /// The model should use the specified tool
    Tool(String),
    /// The model should use any of the tools provided
    Any,
}

/// Configuration options for generating messages
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    /// Temperature controls randomness in generation (0.0 to 2.0)
    pub temperature: Option<f32>,
    /// Top-p controls diversity via nucleus sampling (0.0 to 1.0)
    pub top_p: Option<f32>,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,
    /// Stop sequences to end generation
    pub stop: Option<Vec<String>>,
    /// Whether to stream the response (when supported)
    pub stream: bool,
    /// Tool choice strategy
    pub tool_choice: Option<ToolChoice>,
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl GenerationOptions {
    /// Creates new generation options with default values
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::GenerationOptions;
    ///
    /// let options = GenerationOptions::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the temperature and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::GenerationOptions;
    ///
    /// let options = GenerationOptions::new().with_temperature(0.7);
    /// ```
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the top_p value and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::GenerationOptions;
    ///
    /// let options = GenerationOptions::new().with_top_p(0.9);
    /// ```
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the max_tokens and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::GenerationOptions;
    ///
    /// let options = GenerationOptions::new().with_max_tokens(1000);
    /// ```
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the stop sequences and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::GenerationOptions;
    ///
    /// let options = GenerationOptions::new()
    ///     .with_stop(vec!["END".to_string()]);
    /// ```
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Sets the streaming option and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::GenerationOptions;
    ///
    /// let options = GenerationOptions::new().with_stream(true);
    /// ```
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Adds an extra parameter and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::GenerationOptions;
    /// use serde_json::json;
    ///
    /// let options = GenerationOptions::new()
    ///     .with_extra_param("frequency_penalty", json!(0.5));
    /// ```
    pub fn with_extra_param(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
    
    
    /// Sets the tool choice strategy and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::{GenerationOptions, ToolChoice};
    ///
    /// let options = GenerationOptions::new()
    ///     .with_tool_choice(ToolChoice::Auto);
    /// ```
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }
}

/// Represents metadata about a generated response
#[derive(Debug, Clone)]
pub struct GenerationMetadata {
    /// The ID of the generated response
    pub id: String,
    /// The model used to generate the response
    pub model: String,
    /// The timestamp of the generation
    pub created: u64,
    /// Usage information such as token counts
    pub usage: Option<UsageInfo>,
    /// Additional provider-specific metadata
    pub extra: HashMap<String, serde_json::Value>,
}

/// Usage information for a generated response
#[derive(Debug, Clone)]
pub struct UsageInfo {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total number of tokens used
    pub total_tokens: u32,
}

/// Result of a message generation request
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// The generated message
    pub message: Message,
    /// Metadata about the generation
    pub metadata: GenerationMetadata,
}

/// The main trait that defines the contract for LLM providers
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// List available models for this provider
    ///
    /// Returns a list of models that are available for use with this provider.
    /// 
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::{LlmProvider, MockProvider};
    /// 
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let provider = MockProvider::new();
    /// let models = provider.list_models().await?;
    /// println!("Available models: {}", models.len());
    /// # Ok(())
    /// # }
    /// ```
    async fn list_models(&self) -> Result<Vec<Model>>;

    /// Generate a response from a list of messages
    ///
    /// Takes a list of messages representing a conversation and returns a
    /// generated response. The model specified should be compatible with the provider.
    ///
    /// # Arguments
    ///
    /// * `model` - The ID of the model to use for generation
    /// * `messages` - The conversation history
    /// * `options` - Configuration options for generation
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::{GenerationOptions, LlmProvider, MockProvider};
    /// use language_barrier::message::Message;
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let provider = MockProvider::new();
    /// let messages = vec![
    ///     Message::system("You are a helpful assistant."),
    ///     Message::user("Hello, who are you?"),
    /// ];
    /// let options = GenerationOptions::new().with_temperature(0.7);
    /// 
    /// let result = provider.generate("mock-model", &messages, options).await?;
    /// println!("Response: {:?}", result.message.content);
    /// # Ok(())
    /// # }
    /// ```
    async fn generate(
        &self,
        model: &str,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult>;

    /// Check if a model supports tool calling
    ///
    /// # Arguments
    ///
    /// * `model` - The ID of the model to check
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::{LlmProvider, MockProvider};
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let provider = MockProvider::new();
    /// let supports_tools = provider.supports_tool_calling("mock-model").await?;
    /// println!("Model supports tool calling: {}", supports_tools);
    /// # Ok(())
    /// # }
    /// ```
    async fn supports_tool_calling(&self, model: &str) -> Result<bool>;

    /// Check if a model has a specific capability
    ///
    /// # Arguments
    ///
    /// * `model` - The ID of the model to check
    /// * `capability` - The capability to check for
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::{LlmProvider, MockProvider};
    /// use language_barrier::model::ModelCapability;
    ///
    /// # async fn example() -> language_barrier::error::Result<()> {
    /// let provider = MockProvider::new();
    /// let has_capability = provider
    ///     .has_capability("mock-model", ModelCapability::ChatCompletion)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn has_capability(&self, model: &str, capability: ModelCapability) -> Result<bool>;
}

/// A mock implementation of the LlmProvider trait for testing
pub struct MockProvider {
    models: Vec<Model>,
    responses: Arc<Mutex<HashMap<String, Message>>>,
    metadata: GenerationMetadata,
}

// We'll implement our own approach rather than trying to make Box<dyn Tool> cloneable

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MockProvider {
    /// Creates a new MockProvider with default configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::MockProvider;
    ///
    /// let provider = MockProvider::new();
    /// ```
    pub fn new() -> Self {
        // Create a mock model
        let model = Model::new(
            "mock-model",
            "Mock Model",
            crate::model::ModelFamily::Other,
            vec![
                ModelCapability::ChatCompletion,
                ModelCapability::TextGeneration,
                ModelCapability::ToolCalling,
            ],
            "MockProvider",
        );

        // Default response metadata
        let metadata = GenerationMetadata {
            id: "mock-response-id".to_string(),
            model: "mock-model".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            usage: Some(UsageInfo {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
            extra: HashMap::new(),
        };

        Self {
            models: vec![model],
            responses: Arc::new(Mutex::new(HashMap::new())),
            metadata,
        }
    }

    /// Add a model to the mock provider
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::MockProvider;
    /// use language_barrier::model::{Model, ModelFamily, ModelCapability};
    ///
    /// let mut provider = MockProvider::new();
    /// let model = Model::new(
    ///     "custom-model",
    ///     "Custom Model",
    ///     ModelFamily::Other,
    ///     vec![ModelCapability::ChatCompletion],
    ///     "MockProvider",
    /// );
    /// provider.add_model(model);
    /// ```
    pub fn add_model(&mut self, model: Model) -> &mut Self {
        self.models.push(model);
        self
    }

    /// Add a predefined response for a specific message pattern
    ///
    /// This allows you to set up the mock to return specific responses
    /// for specific input patterns.
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::MockProvider;
    /// use language_barrier::message::{Message, Content};
    ///
    /// let mut provider = MockProvider::new();
    /// provider.add_response(
    ///     "Hello",
    ///     Message::assistant("Hi there! How can I help you?"),
    /// );
    /// ```
    pub fn add_response(&mut self, pattern: impl Into<String>, response: Message) -> &mut Self {
        {
            let mut responses = self.responses.lock().unwrap();
            responses.insert(pattern.into(), response);
        }
        self
    }

    /// Set custom metadata for the mock responses
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::{MockProvider, GenerationMetadata, UsageInfo};
    /// use std::collections::HashMap;
    ///
    /// let mut provider = MockProvider::new();
    /// let metadata = GenerationMetadata {
    ///     id: "custom-id".to_string(),
    ///     model: "custom-model".to_string(),
    ///     created: 1234567890,
    ///     usage: Some(UsageInfo {
    ///         prompt_tokens: 50,
    ///         completion_tokens: 100,
    ///         total_tokens: 150,
    ///     }),
    ///     extra: HashMap::new(),
    /// };
    /// provider.set_metadata(metadata);
    /// ```
    pub fn set_metadata(&mut self, metadata: GenerationMetadata) -> &mut Self {
        self.metadata = metadata;
        self
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    async fn list_models(&self) -> Result<Vec<Model>> {
        Ok(self.models.clone())
    }

    async fn generate(
        &self,
        model: &str,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        // Check if the model exists
        if !self.models.iter().any(|m| m.id == model) {
            return Err(crate::error::Error::UnsupportedModel(format!(
                "Model '{}' not found",
                model
            )));
        }

        // Get the last user message or return a default response
        let last_user_message = messages
            .iter()
            .rev()
            .find(|m| m.role == crate::message::MessageRole::User);

        // Note: Tool functionality has been temporarily removed and will be reimplemented later

        // Normal response logic for non-tool requests
        let response = match last_user_message {
            Some(message) => {
                if let Some(content) = &message.content {
                    match content {
                        Content::Text(text) => {
                            // Try to find a matching response pattern
                            let responses = self.responses.lock().unwrap();
                            if let Some(response) = responses.iter().find_map(|(pattern, resp)| {
                                if text.contains(pattern) {
                                    Some(resp.clone())
                                } else {
                                    None
                                }
                            }) {
                                response
                            } else {
                                // Default response if no pattern matches
                                Message::assistant("I'm a mock assistant. This is a default response.")
                            }
                        }
                        Content::Parts(_) => {
                            // Simple response for multimodal content
                            Message::assistant("I received your multimodal message. This is a mock response.")
                        }
                    }
                } else {
                    // Response when the message has no content
                    Message::assistant("I received your message but it had no content. This is a mock response.")
                }
            }
            None => {
                // Default response when there's no user message
                Message::assistant("Hello! I'm a mock assistant. How can I help you today?")
            }
        };

        // Create and return the result
        let mut metadata = self.metadata.clone();
        metadata.model = model.to_string();
        metadata.created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(GenerationResult {
            message: response,
            metadata,
        })
    }

    async fn supports_tool_calling(&self, model: &str) -> Result<bool> {
        // Check if the model exists
        let model = self
            .models
            .iter()
            .find(|m| m.id == model)
            .ok_or_else(|| {
                crate::error::Error::UnsupportedModel(format!("Model '{}' not found", model))
            })?;

        // Check if the model has the ToolCalling capability
        Ok(model.has_capability(ModelCapability::ToolCalling))
    }

    async fn has_capability(&self, model: &str, capability: ModelCapability) -> Result<bool> {
        // Check if the model exists and has the capability
        let model = self
            .models
            .iter()
            .find(|m| m.id == model)
            .ok_or_else(|| {
                crate::error::Error::UnsupportedModel(format!("Model '{}' not found", model))
            })?;

        Ok(model.has_capability(capability))
    }
}

/// A more advanced mock provider that can simulate function calls and tool usage
pub struct AdvancedMockProvider {
    inner: MockProvider,
    function_calls: Arc<Mutex<HashMap<String, FunctionCall>>>,
}

impl Default for AdvancedMockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedMockProvider {
    /// Creates a new AdvancedMockProvider with the same default configuration as MockProvider
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::AdvancedMockProvider;
    ///
    /// let provider = AdvancedMockProvider::new();
    /// ```
    pub fn new() -> Self {
        Self {
            inner: MockProvider::new(),
            function_calls: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add a model to the mock provider
    pub fn add_model(&mut self, model: Model) -> &mut Self {
        self.inner.add_model(model);
        self
    }

    /// Add a predefined response for a specific message pattern
    pub fn add_response(&mut self, pattern: impl Into<String>, response: Message) -> &mut Self {
        self.inner.add_response(pattern.into(), response);
        self
    }

    /// Set custom metadata for the mock responses
    pub fn set_metadata(&mut self, metadata: GenerationMetadata) -> &mut Self {
        self.inner.set_metadata(metadata);
        self
    }

    /// Add a predefined function call response for a specific message pattern
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::client::AdvancedMockProvider;
    /// use language_barrier::message::FunctionCall;
    ///
    /// let mut provider = AdvancedMockProvider::new();
    /// let function_call = FunctionCall {
    ///     name: "get_weather".to_string(),
    ///     arguments: r#"{"location":"New York","unit":"celsius"}"#.to_string(),
    /// };
    /// provider.add_function_call("weather", function_call);
    /// ```
    pub fn add_function_call(
        &mut self,
        pattern: impl Into<String>,
        function_call: FunctionCall,
    ) -> &mut Self {
        {
            let mut function_calls = self.function_calls.lock().unwrap();
            function_calls.insert(pattern.into(), function_call);
        }
        self
    }

}

#[async_trait]
impl LlmProvider for AdvancedMockProvider {
    async fn list_models(&self) -> Result<Vec<Model>> {
        self.inner.list_models().await
    }

    async fn generate(
        &self,
        model: &str,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        // Get the basic response from the inner mock provider
        let mut result = self.inner.generate(model, messages, options).await?;

        // Get the last user message
        if let Some(message) = messages
            .iter()
            .rev()
            .find(|m| m.role == crate::message::MessageRole::User)
        {
            if let Some(Content::Text(text)) = &message.content {
                // Check if we should add a function call
                let function_calls = self.function_calls.lock().unwrap();
                for (pattern, function_call) in function_calls.iter() {
                    if text.contains(pattern) {
                        result.message.function_call = Some(function_call.clone());
                        result.message.content = None; // Clear content when there's a function call
                        return Ok(result);
                    }
                }
            }
        }

        Ok(result)
    }

    async fn supports_tool_calling(&self, model: &str) -> Result<bool> {
        self.inner.supports_tool_calling(model).await
    }

    async fn has_capability(&self, model: &str, capability: ModelCapability) -> Result<bool> {
        self.inner.has_capability(model, capability).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{MessageRole, Content};
    use crate::model::ModelCapability;

    #[tokio::test]
    async fn test_mock_provider_list_models() {
        let provider = MockProvider::new();
        let models = provider.list_models().await.unwrap();
        
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "mock-model");
    }

    #[tokio::test]
    async fn test_mock_provider_generate() {
        let provider = MockProvider::new();
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello, who are you?"),
        ];
        let options = GenerationOptions::new();
        
        let result = provider.generate("mock-model", &messages, options).await.unwrap();
        
        assert_eq!(result.metadata.model, "mock-model");
        assert!(matches!(result.message.role, MessageRole::Assistant));
    }

    #[tokio::test]
    async fn test_mock_provider_custom_response() {
        let mut provider = MockProvider::new();
        provider.add_response(
            "weather",
            Message::assistant("The weather is sunny today!"),
        );
        
        let messages = vec![
            Message::user("What's the weather like?"),
        ];
        let options = GenerationOptions::new();
        
        let result = provider.generate("mock-model", &messages, options).await.unwrap();
        
        if let Some(Content::Text(text)) = result.message.content {
            assert_eq!(text, "The weather is sunny today!");
        } else {
            panic!("Expected text content");
        }
    }

    #[tokio::test]
    async fn test_mock_provider_unsupported_model() {
        let provider = MockProvider::new();
        let messages = vec![
            Message::user("Hello"),
        ];
        let options = GenerationOptions::new();
        
        let result = provider.generate("non-existent-model", &messages, options).await;
        
        assert!(result.is_err());
        match result {
            Err(crate::error::Error::UnsupportedModel(_)) => (),
            _ => panic!("Expected UnsupportedModel error"),
        }
    }

    #[tokio::test]
    async fn test_mock_provider_capabilities() {
        let provider = MockProvider::new();
        
        let has_chat = provider
            .has_capability("mock-model", ModelCapability::ChatCompletion)
            .await
            .unwrap();
        let has_vision = provider
            .has_capability("mock-model", ModelCapability::Vision)
            .await
            .unwrap();
        
        assert!(has_chat);
        assert!(!has_vision);
    }

    #[tokio::test]
    async fn test_advanced_mock_provider_function_call() {
        let mut provider = AdvancedMockProvider::new();
        let function_call = FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"New York","unit":"celsius"}"#.to_string(),
        };
        provider.add_function_call("weather", function_call.clone());
        
        let messages = vec![
            Message::user("What's the weather in New York?"),
        ];
        let options = GenerationOptions::new();
        
        let result = provider.generate("mock-model", &messages, options).await.unwrap();
        
        assert!(result.message.function_call.is_some());
        assert_eq!(result.message.function_call.unwrap().name, "get_weather");
    }

    // #[tokio::test]
    // async fn test_advanced_mock_provider_tool_calls() {
    //     // Test commented out as tool functionality has been temporarily removed
    // }

    #[tokio::test]
    async fn test_generation_options() {
        let options = GenerationOptions::new()
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_max_tokens(1000)
            .with_stop(vec!["END".to_string()])
            .with_stream(true)
            .with_extra_param("frequency_penalty", serde_json::json!(0.5));
        
        assert_eq!(options.temperature, Some(0.7));
        assert_eq!(options.top_p, Some(0.9));
        assert_eq!(options.max_tokens, Some(1000));
        assert_eq!(options.stop, Some(vec!["END".to_string()]));
        assert!(options.stream);
        assert_eq!(
            options.extra_params.get("frequency_penalty"),
            Some(&serde_json::json!(0.5))
        );
    }
}