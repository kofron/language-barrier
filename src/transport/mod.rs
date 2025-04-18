use async_trait::async_trait;
use reqwest::Request;

use crate::Chat;
use crate::error::Result;
use crate::model::ModelInfo;
use crate::provider::HTTPProvider;

/// Visitor trait for transport implementation
///
/// This trait defines the interface for transport implementations to use when
/// visiting provider implementations. It allows for separation of conversion logic
/// from the actual HTTP request handling.
#[async_trait]
pub trait HTTPTransportVisitor: Send + Sync {
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
    async fn prepare_request<M: ModelInfo, P: HTTPProvider>(
        &self,
        chat: Chat<M>,
        provider: P,
    ) -> Result<Request>;
}

/// HTTP Transport implementation
pub mod http;
