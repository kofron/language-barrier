use std::collections::HashMap;
use async_trait::async_trait;
use reqwest::{Client, header};

use crate::error::{Error, Result};
use crate::message::Message;
use crate::client::GenerationOptions;
use crate::model::ModelInfo;
use super::{TransportVisitor, AnthropicTransportVisitor, GoogleTransportVisitor};

/// HTTP Transport implementation for making API requests to LLM providers
#[derive(Debug, Clone)]
pub struct HttpTransport {
    /// HTTP client for making requests
    client: Client,
}

impl Default for HttpTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpTransport {
    /// Creates a new HTTP transport with default configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::transport::http::HttpTransport;
    ///
    /// let transport = HttpTransport::new();
    /// ```
    pub fn new() -> Self {
        let client = Client::builder()
            .build()
            .unwrap_or_default();

        Self { client }
    }

    /// Creates a new HTTP transport with a custom client
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::transport::http::HttpTransport;
    /// use reqwest::Client;
    ///
    /// let client = Client::new();
    /// let transport = HttpTransport::with_client(client);
    /// ```
    pub fn with_client(client: Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl TransportVisitor for HttpTransport {
    async fn process_request(
        &self,
        payload: serde_json::Value,
        endpoint: &str,
        headers: HashMap<String, String>,
    ) -> Result<serde_json::Value> {
        // Convert the headers for reqwest
        let mut header_map = header::HeaderMap::new();
        for (key, value) in headers {
            if let Ok(header_name) = header::HeaderName::from_bytes(key.as_bytes()) {
                if let Ok(header_value) = header::HeaderValue::from_str(&value) {
                    header_map.insert(header_name, header_value);
                }
            }
        }

        // Add default content-type if not specified
        if !header_map.contains_key(header::CONTENT_TYPE) {
            header_map.insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );
        }

        // Make the API request
        let response = self.client
            .post(endpoint)
            .headers(header_map)
            .json(&payload)
            .send()
            .await
            .map_err(Error::Request)?;

        // Check for success
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            return match status.as_u16() {
                401 => Err(Error::Authentication(format!(
                    "API authentication failed: {}",
                    error_text
                ))),
                429 => Err(Error::RateLimit(format!(
                    "API rate limit exceeded: {}",
                    error_text
                ))),
                _ => Err(Error::Other(format!(
                    "API error ({}): {}",
                    status, error_text
                ))),
            };
        }

        // Parse the response
        let response_text = response.text().await.map_err(Error::Request)?;
        let response_json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| Error::Serialization(e))?;

        Ok(response_json)
    }
}

#[async_trait]
impl AnthropicTransportVisitor for HttpTransport {
    async fn prepare_anthropic_request<M: ModelInfo>(
        &self,
        model: &M,
        _messages: &[Message],
        options: &GenerationOptions,
    ) -> Result<serde_json::Value> {
        // This is just a stub implementation - real implementation will be added later
        let request = serde_json::json!({
            "model": model.model_id(),
            "messages": [],
            "max_tokens": options.max_tokens.unwrap_or(100),
        });
        
        Ok(request)
    }

    async fn process_anthropic_response(
        &self,
        response: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // This is just a stub implementation - real implementation will be added later
        Ok(response)
    }
}

#[async_trait]
impl GoogleTransportVisitor for HttpTransport {
    async fn prepare_google_request<M: ModelInfo>(
        &self,
        _model: &M,
        _messages: &[Message],
        options: &GenerationOptions,
    ) -> Result<serde_json::Value> {
        // This is just a stub implementation - real implementation will be added later
        let request = serde_json::json!({
            "contents": [],
            "generationConfig": {
                "maxOutputTokens": options.max_tokens.unwrap_or(100),
            }
        });
        
        Ok(request)
    }

    async fn process_google_response(
        &self,
        response: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // This is just a stub implementation - real implementation will be added later
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_http_transport_creation() {
        let transport = HttpTransport::new();
        assert!(matches!(transport, HttpTransport { client: _ }));
        
        let client = Client::new();
        let transport = HttpTransport::with_client(client);
        assert!(matches!(transport, HttpTransport { client: _ }));
    }
}