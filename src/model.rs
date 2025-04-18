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

/// Represents a specific language model with its properties
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
}