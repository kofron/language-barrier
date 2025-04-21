use thiserror::Error;

/// Errors that can occur when working with tools
#[derive(Error, Debug)]
pub enum ToolError {
    /// Tool with the specified name was not found in the registry
    #[error("Tool not found: {0}")]
    NotFound(String),

    /// Error generating JSON schema for tool
    #[error("Failed to generate schema for tool '{0}': {1}")]
    SchemaGenerationError(String, serde_json::Error),

    /// Error parsing arguments for tool
    #[error("Failed to parse arguments for tool '{0}': {1}")]
    ArgumentParsingError(String, serde_json::Error),

    /// Invalid arguments for tool
    #[error("Invalid arguments for tool: {0}")]
    InvalidArguments(String),

    /// Tool execution encountered an error
    #[error("Tool execution failed: {0}")]
    ExecutionError(String),

    /// Expected output type did not match actual output type
    #[error("Type mismatch after execution for tool '{0}': Expected different output type")]
    OutputTypeMismatch(String),
}

/// Represents errors that can occur in the language-barrier library
#[derive(Error, Debug)]
pub enum Error {
    /// Error during serialization or deserialization
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Error during HTTP request
    #[error("HTTP request error: {0}")]
    Request(#[from] reqwest::Error),

    #[error("Couldn't parse base url")]
    BaseUrlError(#[from] url::ParseError),

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

    /// Tool not found
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Invalid tool parameter
    #[error("Invalid tool parameter: {0}")]
    InvalidToolParameter(String),
    
    /// Invalid tool arguments
    #[error("Invalid tool arguments: {0}")]
    InvalidToolArguments(String),

    /// Tool execution error
    #[error("Tool execution error: {0}")]
    ToolExecutionError(String),
    
    /// Tool-specific error
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    /// Provider feature not supported
    #[error("Provider feature not supported: {0}")]
    ProviderFeatureNotSupported(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

/// A Result type that uses our Error type
pub type Result<T> = std::result::Result<T, Error>;
