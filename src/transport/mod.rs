use std::collections::HashMap;
use async_trait::async_trait;

use crate::error::Result;
use crate::message::Message;
use crate::client::GenerationOptions;
use crate::model::ModelInfo;

/// Visitor trait for transport implementation
/// 
/// This trait defines the interface for transport implementations to use when
/// visiting provider implementations. It allows for separation of conversion logic
/// from the actual HTTP request handling.
#[async_trait]
pub trait TransportVisitor: Send + Sync {
    /// Process a request with the provider and return the response
    /// 
    /// # Arguments
    /// 
    /// * `payload` - The JSON payload to send to the provider's API
    /// * `endpoint` - The API endpoint to call
    /// * `headers` - Additional headers to include in the request
    /// 
    /// # Returns
    /// 
    /// The JSON response from the provider's API
    async fn process_request(
        &self, 
        payload: serde_json::Value,
        endpoint: &str,
        headers: HashMap<String, String>,
    ) -> Result<serde_json::Value>;
}

/// Visitor that can work with Anthropic's API
#[async_trait]
pub trait AnthropicTransportVisitor: TransportVisitor {
    /// Convert a set of messages to an Anthropic-compatible request
    /// 
    /// # Arguments
    /// 
    /// * `model` - The model to use
    /// * `messages` - The messages to convert
    /// * `options` - Generation options
    /// 
    /// # Returns
    /// 
    /// The JSON request payload to send to Anthropic's API
    async fn prepare_anthropic_request<M: ModelInfo>(
        &self,
        model: &M,
        messages: &[Message],
        options: &GenerationOptions,
    ) -> Result<serde_json::Value>;
    
    /// Process an Anthropic API response
    /// 
    /// # Arguments
    /// 
    /// * `response` - The JSON response from Anthropic's API
    /// 
    /// # Returns
    /// 
    /// The processed response
    async fn process_anthropic_response(
        &self,
        response: serde_json::Value,
    ) -> Result<serde_json::Value>;
}

/// Visitor that can work with Google's API
#[async_trait]
pub trait GoogleTransportVisitor: TransportVisitor {
    /// Convert a set of messages to a Google-compatible request
    /// 
    /// # Arguments
    /// 
    /// * `model` - The model to use
    /// * `messages` - The messages to convert
    /// * `options` - Generation options
    /// 
    /// # Returns
    /// 
    /// The JSON request payload to send to Google's API
    async fn prepare_google_request<M: ModelInfo>(
        &self,
        model: &M,
        messages: &[Message],
        options: &GenerationOptions,
    ) -> Result<serde_json::Value>;
    
    /// Process a Google API response
    /// 
    /// # Arguments
    /// 
    /// * `response` - The JSON response from Google's API
    /// 
    /// # Returns
    /// 
    /// The processed response
    async fn process_google_response(
        &self,
        response: serde_json::Value,
    ) -> Result<serde_json::Value>;
}

/// HTTP Transport implementation
pub mod http;

/// Mock Transport implementation for testing
pub mod mock;