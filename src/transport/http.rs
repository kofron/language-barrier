use reqwest::Client;

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
        let client = Client::builder().build().unwrap_or_default();

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
