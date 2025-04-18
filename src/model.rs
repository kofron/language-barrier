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
