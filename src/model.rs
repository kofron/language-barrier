use std::fmt;

/// Model that can be converted to a string ID for API requests
pub trait ModelInfo: Send + Sync + fmt::Debug {
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

impl ModelInfo for Claude {
    /// All anthropic models have a 200k token context window.
    fn context_window(&self) -> usize {
        200_000
    }

    fn max_output_tokens(&self) -> usize {
        match self {
            Self::Sonnet35 { version: _ } => 8192,
            Self::Sonnet37 {
                use_extended_thinking: _,
            } => 64_000,
            Self::Haiku35 => 8192,
            Self::Haiku3 => 4096,
            Self::Opus3 => 4096,
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
            Self::Flash15 => 8_192,
            Self::Flash20 => 8_192,
            Self::Flash20Lite => 8_192,
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
        }.to_string()
    }
}

/// Represents an OpenAI GPT model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPT {
    /// GPT-4o model
    GPT4o,
    /// GPT-4o-mini model
    GPT4oMini,
    /// GPT-4 Turbo model
    GPT4Turbo,
    /// GPT-3.5 Turbo model
    GPT35Turbo,
}

impl ModelInfo for GPT {
    fn context_window(&self) -> usize {
        match self {
            Self::GPT4o => 128_000,
            Self::GPT4oMini => 128_000,
            Self::GPT4Turbo => 128_000,
            Self::GPT35Turbo => 16_000,
        }
    }

    fn max_output_tokens(&self) -> usize {
        match self {
            Self::GPT4o => 4_096,
            Self::GPT4oMini => 4_096,
            Self::GPT4Turbo => 4_096,
            Self::GPT35Turbo => 4_096,
        }
    }
}

// Implement the OpenAIModelInfo trait from provider/openai.rs
impl crate::provider::openai::OpenAIModelInfo for GPT {
    fn openai_model_id(&self) -> String {
        match self {
            Self::GPT4o => "gpt-4o",
            Self::GPT4oMini => "gpt-4o-mini",
            Self::GPT4Turbo => "gpt-4-turbo",
            Self::GPT35Turbo => "gpt-3.5-turbo",
        }.to_string()
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
            Self::Large => 131_072,  // 131k
            Self::Small => 131_072,  // 131k
            Self::Nemo => 131_072,   // 131k
            Self::Codestral => 262_144, // 256k
            Self::Embed => 8_192,    // 8k
        }
    }

    fn max_output_tokens(&self) -> usize {
        match self {
            Self::Large => 4_096,
            Self::Small => 4_096,
            Self::Nemo => 4_096,
            Self::Codestral => 4_096,
            Self::Embed => 4_096,
        }
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
        }.to_string()
    }
}
