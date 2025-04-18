use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a language model capability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelCapability {
    /// Text generation
    TextGeneration,
    /// Chat completion
    ChatCompletion,
    /// Image generation
    ImageGeneration,
    /// Image understanding (vision)
    Vision,
    /// Audio transcription
    AudioTranscription,
    /// Audio generation (text-to-speech)
    TextToSpeech,
    /// Embeddings generation
    Embeddings,
    /// Tool/function calling
    ToolCalling,
}

/// Represents a family of related models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFamily {
    /// GPT family (OpenAI)
    GPT,
    /// Claude family (Anthropic)
    Claude,
    /// Llama family (Meta)
    Llama,
    /// Gemini family (Google)
    Gemini,
    /// Mistral family (Mistral AI)
    Mistral,
    /// DALL-E family (OpenAI)
    DALLE,
    /// Stable Diffusion family (Stability AI)
    StableDiffusion,
    /// Other model family
    Other,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelFamily::GPT => write!(f, "GPT"),
            ModelFamily::Claude => write!(f, "Claude"),
            ModelFamily::Llama => write!(f, "Llama"),
            ModelFamily::Gemini => write!(f, "Gemini"),
            ModelFamily::Mistral => write!(f, "Mistral"),
            ModelFamily::DALLE => write!(f, "DALL-E"),
            ModelFamily::StableDiffusion => write!(f, "Stable Diffusion"),
            ModelFamily::Other => write!(f, "Other"),
        }
    }
}

/// Model that can be converted to a string ID for API requests
pub trait ModelInfo: Send + Sync + fmt::Debug {
    /// The model ID string that should be passed to the provider's API
    fn model_id(&self) -> String;
    
    /// The name of the model for display purposes
    fn name(&self) -> String;
    
    /// The family this model belongs to
    fn family(&self) -> ModelFamily;
    
    /// Capabilities of this model
    fn capabilities(&self) -> Vec<ModelCapability>;
    
    /// Context window size in tokens
    fn context_window(&self) -> usize;
    
    /// Whether the model supports a specific capability
    fn has_capability(&self, capability: ModelCapability) -> bool {
        self.capabilities().contains(&capability)
    }
}

/// Represents an Anthropic Claude model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicModel {
    /// Claude 3 Opus (February 2024)
    Opus3,
    /// Claude 3 Sonnet (February 2024)
    Sonnet3,
    /// Claude 3.5 Sonnet (July 2024)
    Sonnet35,
    /// Claude 3.7 Sonnet (TBD 2025)
    Sonnet37,
    /// Claude 3 Haiku (March 2024)
    Haiku3,
    /// Claude 3.5 Haiku (TBD 2024)
    Haiku35,
    /// Claude 2.1
    Claude21,
    /// Claude 2.0
    Claude20,
    /// Claude Instant 1.2
    ClaudeInstant12,
}

impl ModelInfo for AnthropicModel {
    fn model_id(&self) -> String {
        match self {
            Self::Opus3 => "claude-3-opus-20240229".to_string(),
            Self::Sonnet3 => "claude-3-sonnet-20240229".to_string(),
            Self::Sonnet35 => "claude-3.5-sonnet-20240620".to_string(),
            Self::Sonnet37 => "claude-3-7-sonnet-20250219".to_string(),
            Self::Haiku3 => "claude-3-haiku-20240307".to_string(), 
            Self::Haiku35 => "claude-3.5-haiku-20240307".to_string(),
            Self::Claude21 => "claude-2.1".to_string(),
            Self::Claude20 => "claude-2.0".to_string(),
            Self::ClaudeInstant12 => "claude-instant-1.2".to_string(),
        }
    }
    
    fn name(&self) -> String {
        match self {
            Self::Opus3 => "Claude 3 Opus".to_string(),
            Self::Sonnet3 => "Claude 3 Sonnet".to_string(),
            Self::Sonnet35 => "Claude 3.5 Sonnet".to_string(),
            Self::Sonnet37 => "Claude 3.7 Sonnet".to_string(),
            Self::Haiku3 => "Claude 3 Haiku".to_string(),
            Self::Haiku35 => "Claude 3.5 Haiku".to_string(),
            Self::Claude21 => "Claude 2.1".to_string(),
            Self::Claude20 => "Claude 2.0".to_string(),
            Self::ClaudeInstant12 => "Claude Instant 1.2".to_string(),
        }
    }
    
    fn family(&self) -> ModelFamily {
        ModelFamily::Claude
    }
    
    fn capabilities(&self) -> Vec<ModelCapability> {
        match self {
            Self::Opus3 | Self::Sonnet3 | Self::Sonnet35 | Self::Sonnet37 | Self::Haiku3 | Self::Haiku35 => vec![
                ModelCapability::ChatCompletion,
                ModelCapability::TextGeneration,
                ModelCapability::Vision,
                ModelCapability::ToolCalling,
            ],
            Self::Claude21 | Self::Claude20 | Self::ClaudeInstant12 => vec![
                ModelCapability::ChatCompletion,
                ModelCapability::TextGeneration,
            ],
        }
    }
    
    fn context_window(&self) -> usize {
        match self {
            Self::Opus3 | Self::Sonnet3 | Self::Sonnet35 | Self::Sonnet37 | Self::Haiku3 | Self::Haiku35 | Self::Claude21 => 200_000,
            Self::Claude20 | Self::ClaudeInstant12 => 100_000,
        }
    }
}

/// Represents a Google Gemini model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoogleModel {
    /// Gemini 1.5 Pro
    Gemini15Pro,
    /// Gemini 1.5 Flash
    Gemini15Flash,
    /// Gemini 1.5 Flash 8B (smaller parameter version)
    Gemini15Flash8B,
    /// Gemini 2 Flash
    Gemini2Flash,
    /// Gemini 2 Flash Lite
    Gemini2FlashLite,
    /// Gemini 2.5 Flash
    Gemini25Flash,
    /// Gemini 2.5 Pro
    Gemini25Pro,
    /// Gemini 1.0 Pro
    Gemini10Pro,
    /// Gemini 1.0 Pro Vision
    Gemini10ProVision,
    /// Gemini 1.0 Ultra
    Gemini10Ultra,
}

impl ModelInfo for GoogleModel {
    fn model_id(&self) -> String {
        match self {
            Self::Gemini15Pro => "gemini-1.5-pro".to_string(),
            Self::Gemini15Flash => "gemini-1.5-flash".to_string(),
            Self::Gemini15Flash8B => "gemini-1.5-flash-8b".to_string(),
            Self::Gemini2Flash => "gemini-2-flash".to_string(),
            Self::Gemini2FlashLite => "gemini-2-flash-lite".to_string(),
            Self::Gemini25Flash => "gemini-2.5-flash".to_string(),
            Self::Gemini25Pro => "gemini-2.5-pro".to_string(),
            Self::Gemini10Pro => "gemini-1.0-pro".to_string(),
            Self::Gemini10ProVision => "gemini-1.0-pro-vision".to_string(),
            Self::Gemini10Ultra => "gemini-1.0-ultra".to_string(),
        }
    }
    
    fn name(&self) -> String {
        match self {
            Self::Gemini15Pro => "Gemini 1.5 Pro".to_string(),
            Self::Gemini15Flash => "Gemini 1.5 Flash".to_string(),
            Self::Gemini15Flash8B => "Gemini 1.5 Flash 8B".to_string(),
            Self::Gemini2Flash => "Gemini 2 Flash".to_string(),
            Self::Gemini2FlashLite => "Gemini 2 Flash Lite".to_string(),
            Self::Gemini25Flash => "Gemini 2.5 Flash".to_string(),
            Self::Gemini25Pro => "Gemini 2.5 Pro".to_string(),
            Self::Gemini10Pro => "Gemini 1.0 Pro".to_string(),
            Self::Gemini10ProVision => "Gemini 1.0 Pro Vision".to_string(),
            Self::Gemini10Ultra => "Gemini 1.0 Ultra".to_string(),
        }
    }
    
    fn family(&self) -> ModelFamily {
        ModelFamily::Gemini
    }
    
    fn capabilities(&self) -> Vec<ModelCapability> {
        match self {
            Self::Gemini15Pro | Self::Gemini15Flash | Self::Gemini15Flash8B | 
            Self::Gemini2Flash | Self::Gemini2FlashLite | 
            Self::Gemini25Flash | Self::Gemini25Pro | 
            Self::Gemini10Pro | Self::Gemini10Ultra => vec![
                ModelCapability::ChatCompletion,
                ModelCapability::TextGeneration,
                ModelCapability::Vision,
                ModelCapability::ToolCalling,
            ],
            Self::Gemini10ProVision => vec![
                ModelCapability::ChatCompletion,
                ModelCapability::TextGeneration,
                ModelCapability::Vision,
            ],
        }
    }
    
    fn context_window(&self) -> usize {
        match self {
            Self::Gemini15Pro | Self::Gemini15Flash | Self::Gemini15Flash8B => 1_000_000,
            Self::Gemini2Flash | Self::Gemini2FlashLite | Self::Gemini25Flash | Self::Gemini25Pro => 2_000_000,
            Self::Gemini10Pro | Self::Gemini10ProVision | Self::Gemini10Ultra => 32_768,
        }
    }
}

/// Represents a specific language model with its properties
/// This generic model type is being maintained for backward compatibility
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    /// The unique identifier of the model
    pub id: String,
    /// The name of the model
    pub name: String,
    /// The model family
    pub family: ModelFamily,
    /// The capabilities of the model
    pub capabilities: Vec<ModelCapability>,
    /// The provider of the model
    pub provider: String,
    /// The version of the model (if applicable)
    pub version: Option<String>,
    /// The context window size of the model
    pub context_window: Option<usize>,
    /// Whether the model is currently available
    #[serde(default)]
    pub available: bool,
}

impl Model {
    /// Creates a new model with basic properties
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::model::{Model, ModelFamily, ModelCapability};
    ///
    /// let model = Model::new(
    ///     "gpt-4",
    ///     "GPT-4",
    ///     ModelFamily::GPT,
    ///     vec![ModelCapability::ChatCompletion],
    ///     "openai",
    /// );
    /// ```
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        family: ModelFamily,
        capabilities: Vec<ModelCapability>,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            family,
            capabilities,
            provider: provider.into(),
            version: None,
            context_window: None,
            available: true,
        }
    }

    /// Sets the version and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::model::{Model, ModelFamily, ModelCapability};
    ///
    /// let model = Model::new(
    ///     "gpt-4",
    ///     "GPT-4",
    ///     ModelFamily::GPT,
    ///     vec![ModelCapability::ChatCompletion],
    ///     "openai",
    /// ).with_version("turbo");
    /// ```
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Sets the context window size and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::model::{Model, ModelFamily, ModelCapability};
    ///
    /// let model = Model::new(
    ///     "gpt-4",
    ///     "GPT-4",
    ///     ModelFamily::GPT,
    ///     vec![ModelCapability::ChatCompletion],
    ///     "openai",
    /// ).with_context_window(8192);
    /// ```
    pub fn with_context_window(mut self, context_window: usize) -> Self {
        self.context_window = Some(context_window);
        self
    }

    /// Sets the availability and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::model::{Model, ModelFamily, ModelCapability};
    ///
    /// let model = Model::new(
    ///     "gpt-4",
    ///     "GPT-4",
    ///     ModelFamily::GPT,
    ///     vec![ModelCapability::ChatCompletion],
    ///     "openai",
    /// ).with_availability(false);
    /// ```
    pub fn with_availability(mut self, available: bool) -> Self {
        self.available = available;
        self
    }

    /// Checks if the model has a specific capability
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::model::{Model, ModelFamily, ModelCapability};
    ///
    /// let model = Model::new(
    ///     "gpt-4",
    ///     "GPT-4",
    ///     ModelFamily::GPT,
    ///     vec![ModelCapability::ChatCompletion],
    ///     "openai",
    /// );
    ///
    /// assert!(model.has_capability(ModelCapability::ChatCompletion));
    /// assert!(!model.has_capability(ModelCapability::Vision));
    /// ```
    pub fn has_capability(&self, capability: ModelCapability) -> bool {
        self.capabilities.contains(&capability)
    }
}

// Allow conversion from ModelInfo types to a generic Model for backward compatibility
impl<T: ModelInfo> From<T> for Model {
    fn from(model_info: T) -> Self {
        Self {
            id: model_info.model_id(),
            name: model_info.name(),
            family: model_info.family(),
            capabilities: model_info.capabilities(),
            provider: match model_info.family() {
                ModelFamily::Claude => "anthropic".to_string(),
                ModelFamily::Gemini => "google".to_string(),
                _ => "unknown".to_string(),
            },
            version: None,
            context_window: Some(model_info.context_window()),
            available: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = Model::new(
            "gpt-4",
            "GPT-4",
            ModelFamily::GPT,
            vec![ModelCapability::ChatCompletion],
            "openai",
        )
        .with_version("turbo")
        .with_context_window(8192);

        assert_eq!(model.id, "gpt-4");
        assert_eq!(model.name, "GPT-4");
        assert_eq!(model.family, ModelFamily::GPT);
        assert_eq!(model.provider, "openai");
        assert_eq!(model.version, Some("turbo".to_string()));
        assert_eq!(model.context_window, Some(8192));
        assert!(model.available);
    }

    #[test]
    fn test_model_capabilities() {
        let model = Model::new(
            "gpt-4",
            "GPT-4",
            ModelFamily::GPT,
            vec![
                ModelCapability::ChatCompletion,
                ModelCapability::TextGeneration,
            ],
            "openai",
        );

        assert!(model.has_capability(ModelCapability::ChatCompletion));
        assert!(model.has_capability(ModelCapability::TextGeneration));
        assert!(!model.has_capability(ModelCapability::Vision));
    }

    #[test]
    fn test_model_family_display() {
        assert_eq!(ModelFamily::GPT.to_string(), "GPT");
        assert_eq!(ModelFamily::Claude.to_string(), "Claude");
        assert_eq!(ModelFamily::Llama.to_string(), "Llama");
    }
    
    #[test]
    fn test_anthropic_model_info() {
        let model = AnthropicModel::Sonnet3;
        
        assert_eq!(model.model_id(), "claude-3-sonnet-20240229");
        assert_eq!(model.name(), "Claude 3 Sonnet");
        assert_eq!(model.family(), ModelFamily::Claude);
        assert!(model.has_capability(ModelCapability::ChatCompletion));
        assert!(model.has_capability(ModelCapability::Vision));
        assert!(model.has_capability(ModelCapability::ToolCalling));
        assert_eq!(model.context_window(), 200_000);
        
        // Test conversion to generic Model
        let generic_model: Model = model.into();
        assert_eq!(generic_model.id, "claude-3-sonnet-20240229");
        assert_eq!(generic_model.name, "Claude 3 Sonnet");
        assert_eq!(generic_model.family, ModelFamily::Claude);
        assert_eq!(generic_model.provider, "anthropic");
        assert_eq!(generic_model.context_window, Some(200_000));
    }
    
    #[test]
    fn test_google_model_info() {
        let model = GoogleModel::Gemini15Pro;
        
        assert_eq!(model.model_id(), "gemini-1.5-pro");
        assert_eq!(model.name(), "Gemini 1.5 Pro");
        assert_eq!(model.family(), ModelFamily::Gemini);
        assert!(model.has_capability(ModelCapability::ChatCompletion));
        assert!(model.has_capability(ModelCapability::Vision));
        assert!(model.has_capability(ModelCapability::ToolCalling));
        assert_eq!(model.context_window(), 1_000_000);
        
        // Test conversion to generic Model
        let generic_model: Model = model.into();
        assert_eq!(generic_model.id, "gemini-1.5-pro");
        assert_eq!(generic_model.name, "Gemini 1.5 Pro");
        assert_eq!(generic_model.family, ModelFamily::Gemini);
        assert_eq!(generic_model.provider, "google");
        assert_eq!(generic_model.context_window, Some(1_000_000));
    }
}