use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents an AI provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    /// OpenAI
    OpenAI,
    /// Anthropic
    Anthropic,
    /// Google
    Google,
    /// Meta
    Meta,
    /// Mistral AI
    Mistral,
    /// Cohere
    Cohere,
    /// Azure OpenAI
    AzureOpenAI,
    /// AWS Bedrock
    AwsBedrock,
    /// Stability AI
    StabilityAI,
    /// Other provider
    Other,
}

impl Provider {
    /// Returns the provider's API base URL
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::Provider;
    ///
    /// assert_eq!(Provider::OpenAI.api_base(), "https://api.openai.com/v1");
    /// ```
    pub fn api_base(&self) -> &'static str {
        match self {
            Provider::OpenAI => "https://api.openai.com/v1",
            Provider::Anthropic => "https://api.anthropic.com/v1",
            Provider::Google => "https://generativelanguage.googleapis.com/v1",
            Provider::Mistral => "https://api.mistral.ai/v1",
            Provider::Cohere => "https://api.cohere.ai/v1",
            Provider::AzureOpenAI => "", // Configured per deployment
            Provider::AwsBedrock => "",  // Uses AWS SDK
            Provider::StabilityAI => "https://api.stability.ai/v1",
            Provider::Meta => "", // No public API
            Provider::Other => "",
        }
    }

    /// Returns the provider's expected environment variable name for API key
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::provider::Provider;
    ///
    /// assert_eq!(Provider::OpenAI.api_key_env_var(), "OPENAI_API_KEY");
    /// ```
    pub fn api_key_env_var(&self) -> &'static str {
        match self {
            Provider::OpenAI => "OPENAI_API_KEY",
            Provider::Anthropic => "ANTHROPIC_API_KEY",
            Provider::Google => "GOOGLE_API_KEY",
            Provider::Mistral => "MISTRAL_API_KEY",
            Provider::Cohere => "COHERE_API_KEY",
            Provider::AzureOpenAI => "AZURE_OPENAI_API_KEY",
            Provider::AwsBedrock => "AWS_ACCESS_KEY_ID", // Uses AWS credentials
            Provider::StabilityAI => "STABILITY_API_KEY",
            Provider::Meta => "",
            Provider::Other => "API_KEY",
        }
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Provider::OpenAI => write!(f, "OpenAI"),
            Provider::Anthropic => write!(f, "Anthropic"),
            Provider::Google => write!(f, "Google"),
            Provider::Meta => write!(f, "Meta"),
            Provider::Mistral => write!(f, "Mistral AI"),
            Provider::Cohere => write!(f, "Cohere"),
            Provider::AzureOpenAI => write!(f, "Azure OpenAI"),
            Provider::AwsBedrock => write!(f, "AWS Bedrock"),
            Provider::StabilityAI => write!(f, "Stability AI"),
            Provider::Other => write!(f, "Other"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_display() {
        assert_eq!(Provider::OpenAI.to_string(), "OpenAI");
        assert_eq!(Provider::Anthropic.to_string(), "Anthropic");
        assert_eq!(Provider::Google.to_string(), "Google");
    }

    #[test]
    fn test_provider_api_base() {
        assert_eq!(Provider::OpenAI.api_base(), "https://api.openai.com/v1");
        assert_eq!(Provider::Anthropic.api_base(), "https://api.anthropic.com/v1");
    }

    #[test]
    fn test_provider_api_key_env_var() {
        assert_eq!(Provider::OpenAI.api_key_env_var(), "OPENAI_API_KEY");
        assert_eq!(Provider::Anthropic.api_key_env_var(), "ANTHROPIC_API_KEY");
    }
}