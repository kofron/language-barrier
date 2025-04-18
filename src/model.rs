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
