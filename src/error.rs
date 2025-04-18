use thiserror::Error;

/// Represents errors that can occur in the language-barrier library
#[derive(Error, Debug)]
pub enum Error {
    /// Error during serialization or deserialization
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Error during HTTP request
    #[error("HTTP request error: {0}")]
    Request(#[from] reqwest::Error),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    /// Authentication error
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Model not supported
    #[error("Model not supported: {0}")]
    UnsupportedModel(String),

    /// Provider not available
    #[error("Provider not available: {0}")]
    ProviderUnavailable(String),

    /// Context length exceeded
    #[error("Context length exceeded: {0}")]
    ContextLengthExceeded(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

/// A Result type that uses our Error type
pub type Result<T> = std::result::Result<T, Error>;