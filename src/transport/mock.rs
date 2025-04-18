use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use async_trait::async_trait;

use crate::error::Result;
use crate::message::Message;
use crate::client::GenerationOptions;
use crate::model::ModelInfo;
use super::{TransportVisitor, AnthropicTransportVisitor, GoogleTransportVisitor};

/// Mock transport for testing
#[derive(Debug, Clone, Default)]
pub struct MockTransport {
    /// Responses to return for specific requests
    responses: Arc<Mutex<HashMap<String, serde_json::Value>>>,
    /// Last request that was processed
    last_request: Arc<Mutex<Option<serde_json::Value>>>,
}

impl MockTransport {
    /// Creates a new mock transport
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::transport::mock::MockTransport;
    ///
    /// let transport = MockTransport::new();
    /// ```
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(HashMap::new())),
            last_request: Arc::new(Mutex::new(None)),
        }
    }

    /// Adds a mock response for a specific request key
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::transport::mock::MockTransport;
    /// use serde_json::json;
    ///
    /// let mut transport = MockTransport::new();
    /// transport.add_response("test_key", json!({
    ///     "id": "response-123",
    ///     "message": "This is a mock response"
    /// }));
    /// ```
    pub fn add_response(&mut self, key: impl Into<String>, response: serde_json::Value) -> &mut Self {
        {
            let mut responses = self.responses.lock().unwrap();
            responses.insert(key.into(), response);
        }
        self
    }

    /// Gets the last request that was processed
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::transport::mock::MockTransport;
    ///
    /// let transport = MockTransport::new();
    /// let last_request = transport.last_request();
    /// assert!(last_request.is_none());
    /// ```
    pub fn last_request(&self) -> Option<serde_json::Value> {
        self.last_request.lock().unwrap().clone()
    }
}

#[async_trait]
impl TransportVisitor for MockTransport {
    async fn process_request(
        &self,
        payload: serde_json::Value,
        endpoint: &str,
        _headers: HashMap<String, String>,
    ) -> Result<serde_json::Value> {
        // Store the last request
        *self.last_request.lock().unwrap() = Some(payload.clone());

        // Generate a key from the endpoint and model ID if present
        let model_id = payload.get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("default");
        
        let key = format!("{}:{}", endpoint, model_id);
        
        // Look up response by key
        let responses = self.responses.lock().unwrap();
        if let Some(response) = responses.get(&key) {
            Ok(response.clone())
        } else {
            // Return a default response if no matching response is found
            Ok(serde_json::json!({
                "id": "mock-response",
                "created": 1234567890,
                "model": model_id,
                "content": [
                    {
                        "type": "text",
                        "text": "This is a mock response"
                    }
                ]
            }))
        }
    }
}

#[async_trait]
impl AnthropicTransportVisitor for MockTransport {
    async fn prepare_anthropic_request<M: ModelInfo>(
        &self,
        model: &M,
        messages: &[Message],
        options: &GenerationOptions,
    ) -> Result<serde_json::Value> {
        // Create a simple mock Anthropic API request
        let request = serde_json::json!({
            "model": model.model_id(),
            "messages": [
                // Just include the number of messages rather than their content
                { "key": format!("{} messages", messages.len()) }
            ],
            "max_tokens": options.max_tokens.unwrap_or(100),
            "system": "You are Claude, a helpful AI assistant."
        });
        
        Ok(request)
    }

    async fn process_anthropic_response(
        &self,
        response: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // For mock, just return the response directly
        Ok(response)
    }
}

#[async_trait]
impl GoogleTransportVisitor for MockTransport {
    async fn prepare_google_request<M: ModelInfo>(
        &self,
        model: &M,
        messages: &[Message],
        options: &GenerationOptions,
    ) -> Result<serde_json::Value> {
        // Create a simple mock Google API request
        let request = serde_json::json!({
            "model": model.model_id(),
            "contents": [
                // Just include the number of messages rather than their content
                { "key": format!("{} messages", messages.len()) }
            ],
            "generationConfig": {
                "maxOutputTokens": options.max_tokens.unwrap_or(100)
            }
        });
        
        Ok(request)
    }

    async fn process_google_response(
        &self,
        response: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // For mock, just return the response directly
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AnthropicModel, GoogleModel};
    
    #[test]
    fn test_mock_transport_creation() {
        let transport = MockTransport::new();
        assert!(transport.last_request().is_none());
        
        let mut transport = MockTransport::new();
        transport.add_response("test", serde_json::json!({"key": "value"}));
        assert!(transport.last_request().is_none());
    }
    
    #[tokio::test]
    async fn test_mock_transport_process_request() {
        let mut transport = MockTransport::new();
        transport.add_response(
            "test_endpoint:claude-model", 
            serde_json::json!({"response": "custom"}),
        );
        
        let payload = serde_json::json!({
            "model": "claude-model",
            "messages": []
        });
        
        let result = transport.process_request(
            payload.clone(),
            "test_endpoint",
            HashMap::new(),
        ).await.unwrap();
        
        assert_eq!(result.get("response").unwrap(), "custom");
        assert_eq!(transport.last_request().unwrap(), payload);
    }
    
    #[tokio::test]
    async fn test_prepare_anthropic_request() {
        let transport = MockTransport::new();
        let model = AnthropicModel::Sonnet3;
        let messages = vec![Message::user("Hello")];
        let options = GenerationOptions::new();
        
        let request = transport.prepare_anthropic_request(&model, &messages, &options)
            .await
            .unwrap();
            
        assert_eq!(request["model"], "claude-3-sonnet-20240229");
        assert!(request.get("messages").is_some());
        assert_eq!(request["system"], "You are Claude, a helpful AI assistant.");
    }
    
    #[tokio::test]
    async fn test_prepare_google_request() {
        let transport = MockTransport::new();
        let model = GoogleModel::Gemini15Pro;
        let messages = vec![Message::user("Hello")];
        let options = GenerationOptions::new();
        
        let request = transport.prepare_google_request(&model, &messages, &options)
            .await
            .unwrap();
            
        assert_eq!(request["model"], "gemini-1.5-pro");
        assert!(request.get("contents").is_some());
        assert!(request.get("generationConfig").is_some());
    }
}