use std::fmt;

/// Model that can be converted to a string ID for API requests
pub trait ModelInfo: Send + Sync + fmt::Debug + Clone + Copy {
    /// Context window size in tokens
    fn context_window(&self) -> usize;

    /// Maximum number of output tokens
    /// NOTE: we may want to do something smart here to have this be
    /// context-dependent.  for example if you set the right headers
    /// for anthropic, 3.7 can output 128k instead of 64k.
    fn max_output_tokens(&self) -> usize;
}

/// Sonnet 3.5 has two published tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sonnet35Version {
    V1,
    V2,
}

/// Represents an Anthropic Claude model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Claude {
    Sonnet35 { version: Sonnet35Version },
    Sonnet37 { use_extended_thinking: bool },
    Haiku35,
    Haiku3,
    Opus3,
}

/// Represents an Ollama model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ollama {
    /// Llama 3 models
    Llama3 { size: OllamaModelSize },
    /// LlaVA multimodal models
    Llava,
    /// Mistral models
    Mistral { size: OllamaModelSize },
    /// Custom model with specified name
    Custom { name: &'static str },
}

/// Standard model sizes for Ollama models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OllamaModelSize {
    /// 8B parameters
    _8B,
    /// 7B parameters
    _7B,
    /// 3B parameters
    _3B,
    /// 1B parameters
    _1B,
}

impl Default for Claude {
    fn default() -> Self {
        Self::Opus3
    }
}

impl ModelInfo for Claude {
    /// All anthropic models have a 200k token context window.
    fn context_window(&self) -> usize {
        200_000
    }

    fn max_output_tokens(&self) -> usize {
        match self {
            Self::Sonnet37 {
                use_extended_thinking: _,
            } => 64_000,
            Self::Sonnet35 { version: _ } | Self::Haiku35 => 8192,
            Self::Haiku3 | Self::Opus3 => 4096,
        }
    }
}

/// Represents a Google Gemini model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemini {
    /// Gemini 1.5 Flash
    Flash15,
    /// Gemini 2.0 Flash
    Flash20,
    /// Gemini 2.0 Flash-Lite
    Flash20Lite,
    /// Gemini 2.5 Flash Preview
    Flash25Preview,
}

impl ModelInfo for Gemini {
    fn context_window(&self) -> usize {
        // All Gemini Flash models support 1M token context
        1_048_576
    }

    fn max_output_tokens(&self) -> usize {
        match self {
            Self::Flash15 | Self::Flash20 | Self::Flash20Lite => 8_192,
            Self::Flash25Preview => 65_536,
        }
    }
}

// Implement the GeminiModelInfo trait from provider/gemini.rs
impl crate::provider::gemini::GeminiModelInfo for Gemini {
    fn gemini_model_id(&self) -> String {
        match self {
            Self::Flash15 => "gemini-1.5-flash",
            Self::Flash20 => "gemini-2.0-flash",
            Self::Flash20Lite => "gemini-2.0-flash-lite",
            Self::Flash25Preview => "gemini-2.5-flash-preview-04-17",
        }
        .to_string()
    }
}

/// Represents an `OpenAI` GPT model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAi {
    /// GPT-4o model
    GPT4o,
    /// GPT-4o-mini model
    GPT4oMini,
    /// GPT-4 Turbo model
    GPT4Turbo,
    /// GPT-3.5 Turbo model
    GPT35Turbo,
    /// O1
    O1,
    O1Mini,
    O1Pro,
    /// O3
    O3,
    O3Mini,
    O4Mini,
}

impl ModelInfo for OpenAi {
    fn context_window(&self) -> usize {
        match self {
            Self::O1Mini | Self::GPT4o | Self::GPT4oMini | Self::GPT4Turbo => 128_000,
            Self::GPT35Turbo => 16_000,
            _ => 200_000,
        }
    }

    fn max_output_tokens(&self) -> usize {
        match self {
            Self::GPT4o | Self::GPT4oMini | Self::GPT4Turbo | Self::GPT35Turbo => 4_096,
            Self::O1Mini => 65_536,
            _ => 100_000,
        }
    }
}

// Implement the OpenAIModelInfo trait from provider/openai.rs
impl crate::provider::openai::OpenAIModelInfo for OpenAi {
    fn openai_model_id(&self) -> String {
        match self {
            Self::GPT4o => "gpt-4o",
            Self::GPT4oMini => "gpt-4o-mini",
            Self::GPT4Turbo => "gpt-4-turbo",
            Self::GPT35Turbo => "gpt-3.5-turbo",
            Self::O4Mini => "o4-mini-2025-04-16",
            Self::O3 => "o3-2025-04-16",
            Self::O3Mini => "o3-mini-2025-01-31",
            Self::O1 => "o1-2024-12-17",
            Self::O1Mini => "o1-mini-2024-09-12",
            Self::O1Pro => "o1-pro-2025-03-19",
        }
        .to_string()
    }
}

/// Represents a Mistral AI model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mistral {
    /// Mistral Large
    Large,
    /// Mistral Small
    Small,
    /// Open Mistral Nemo
    Nemo,
    /// Codestral
    Codestral,
    /// Mistral Embed
    Embed,
}

impl ModelInfo for Mistral {
    fn context_window(&self) -> usize {
        match self {
            Self::Large | Self::Small | Self::Nemo => 131_072, // 131k
            Self::Codestral => 262_144,                        // 256k
            Self::Embed => 8_192,                              // 8k
        }
    }

    fn max_output_tokens(&self) -> usize {
        // All Mistral models have the same max output tokens
        4_096
    }
}

// Implement the MistralModelInfo trait from provider/mistral.rs
impl crate::provider::mistral::MistralModelInfo for Mistral {
    fn mistral_model_id(&self) -> String {
        match self {
            Self::Large => "mistral-large-latest",
            Self::Small => "mistral-small-latest",
            Self::Nemo => "open-mistral-nemo",
            Self::Codestral => "codestral-latest",
            Self::Embed => "mistral-embed",
        }
        .to_string()
    }
}

impl Default for Ollama {
    fn default() -> Self {
        Self::Llama3 { size: OllamaModelSize::_7B }
    }
}

impl ModelInfo for Ollama {
    fn context_window(&self) -> usize {
        match self {
            Self::Llama3 { size } => match size {
                OllamaModelSize::_8B => 32_768,
                OllamaModelSize::_7B => 32_768,
                OllamaModelSize::_3B => 16_384,
                OllamaModelSize::_1B => 8_192,
            },
            Self::Llava => 8_192,
            Self::Mistral { size } => match size {
                OllamaModelSize::_8B => 32_768,
                OllamaModelSize::_7B => 16_384,
                OllamaModelSize::_3B => 8_192,
                OllamaModelSize::_1B => 4_096,
            },
            Self::Custom { .. } => 8_192, // Default for unknown models
        }
    }

    fn max_output_tokens(&self) -> usize {
        match self {
            Self::Llama3 { .. } => 4_096,
            Self::Llava => 4_096,
            Self::Mistral { .. } => 4_096,
            Self::Custom { .. } => 4_096, // Default for unknown models
        }
    }
}
