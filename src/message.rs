use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents the content of a message, which can be text or other structured data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// Simple text content
    Text(String),
    /// Structured content with parts (for multimodal models)
    Parts(Vec<ContentPart>),
}

impl Content {
    /// Creates a new text content
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Content;
    ///
    /// let content = Content::text("Hello, world!");
    /// ```
    pub fn text(text: impl Into<String>) -> Self {
        Content::Text(text.into())
    }

    /// Creates a new parts content
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::{Content, ContentPart};
    ///
    /// let parts = vec![ContentPart::text("Hello"), ContentPart::text("world")];
    /// let content = Content::parts(parts);
    /// ```
    #[must_use]
    pub fn parts(parts: Vec<ContentPart>) -> Self {
        Content::Parts(parts)
    }

    /// Returns true if the content is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Content;
    ///
    /// let content = Content::text("");
    /// assert!(content.is_empty());
    ///
    /// let content = Content::text("Hello");
    /// assert!(!content.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Content::Text(text) => text.is_empty(),
            Content::Parts(parts) => parts.is_empty() || parts.iter().all(ContentPart::is_empty),
        }
    }
}

/// Represents a part of structured content for multimodal models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text part
    #[serde(rename = "text")]
    Text {
        /// The text content
        text: String,
    },
    /// Image part
    #[serde(rename = "image_url")]
    ImageUrl {
        /// The image URL and metadata
        image_url: ImageUrl,
    },
}

impl ContentPart {
    /// Creates a new text part
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::ContentPart;
    ///
    /// let part = ContentPart::text("Hello, world!");
    /// ```
    pub fn text(text: impl Into<String>) -> Self {
        ContentPart::Text { text: text.into() }
    }

    /// Creates a new image URL part
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::ContentPart;
    ///
    /// let part = ContentPart::image_url("https://example.com/image.jpg");
    /// ```
    pub fn image_url(url: impl Into<String>) -> Self {
        ContentPart::ImageUrl { image_url: ImageUrl::new(url) }
    }

    /// Returns true if the part is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::ContentPart;
    ///
    /// let part = ContentPart::text("");
    /// assert!(part.is_empty());
    ///
    /// let part = ContentPart::text("Hello");
    /// assert!(!part.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            ContentPart::Text { text } => text.is_empty(),
            ContentPart::ImageUrl { .. } => false,
        }
    }
}

/// Represents an image URL with optional metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The URL of the image
    pub url: String,
    /// Optional detail level (for some providers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ImageUrl {
    /// Creates a new image URL
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::ImageUrl;
    ///
    /// let image_url = ImageUrl::new("https://example.com/image.jpg");
    /// ```
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            detail: None,
        }
    }

    /// Sets the detail level and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::ImageUrl;
    ///
    /// let image_url = ImageUrl::new("https://example.com/image.jpg")
    ///     .with_detail("high");
    /// ```
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// Represents a function definition within a tool call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    /// The name of the function
    pub name: String,
    /// The arguments to the function (typically JSON)
    pub arguments: String,
}

/// Represents a tool call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The ID of the tool call
    pub id: String,
    /// The type of the tool call
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition
    pub function: Function,
}

/// Represents a message in a conversation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    /// Message from the system (instructions)
    #[serde(rename = "system")]
    System {
        /// The content of the system message
        content: String,
        /// Additional provider-specific metadata
        #[serde(flatten, skip_serializing_if = "HashMap::is_empty")]
        metadata: HashMap<String, serde_json::Value>,
    },

    /// Message from the user
    #[serde(rename = "user")]
    User {
        /// The content of the user message
        content: Content,
        /// The name of the user (optional)
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Additional provider-specific metadata
        #[serde(flatten, skip_serializing_if = "HashMap::is_empty")]
        metadata: HashMap<String, serde_json::Value>,
    },

    /// Message from the assistant
    #[serde(rename = "assistant")]
    Assistant {
        /// The content of the assistant message
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Content>,
        /// The tool calls made by the assistant
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        tool_calls: Vec<ToolCall>,
        /// Additional provider-specific metadata
        #[serde(flatten, skip_serializing_if = "HashMap::is_empty")]
        metadata: HashMap<String, serde_json::Value>,
    },

    /// Message from a tool
    #[serde(rename = "tool")]
    Tool {
        /// The ID of the tool call this message is responding to
        tool_call_id: String,
        /// The content of the tool response
        content: String,
        /// Additional provider-specific metadata
        #[serde(flatten, skip_serializing_if = "HashMap::is_empty")]
        metadata: HashMap<String, serde_json::Value>,
    },
}

impl Message {
    /// Creates a new system message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Message;
    ///
    /// let msg = Message::system("You are a helpful assistant.");
    /// ```
    pub fn system(content: impl Into<String>) -> Self {
        Message::System {
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Creates a new user message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Message;
    ///
    /// let msg = Message::user("Hello, can you help me?");
    /// ```
    pub fn user(content: impl Into<String>) -> Self {
        Message::User {
            content: Content::Text(content.into()),
            name: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Creates a new user message with a name
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Message;
    ///
    /// let msg = Message::user_with_name("John", "Hello, can you help me?");
    /// ```
    pub fn user_with_name(name: impl Into<String>, content: impl Into<String>) -> Self {
        Message::User {
            content: Content::Text(content.into()),
            name: Some(name.into()),
            metadata: HashMap::new(),
        }
    }

    /// Creates a new user message with multimodal content
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::{Message, Content, ContentPart};
    ///
    /// let parts = vec![
    ///     ContentPart::text("Look at this image:"),
    ///     ContentPart::image_url("https://example.com/image.jpg"),
    /// ];
    /// let msg = Message::user_with_parts(parts);
    /// ```
    #[must_use]
    pub fn user_with_parts(parts: Vec<ContentPart>) -> Self {
        Message::User {
            content: Content::Parts(parts),
            name: None,
            metadata: HashMap::new(),
        }
    }

    /// Creates a new assistant message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Message;
    ///
    /// let msg = Message::assistant("I'm here to help you.");
    /// ```
    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant {
            content: Some(Content::Text(content.into())),
            tool_calls: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Creates a new assistant message with tool calls
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::{Message, ToolCall, Function};
    ///
    /// let tool_call = ToolCall {
    ///     id: "call_123".to_string(),
    ///     tool_type: "function".to_string(),
    ///     function: Function {
    ///         name: "get_weather".to_string(),
    ///         arguments: "{\"location\":\"San Francisco\"}".to_string(),
    ///     },
    /// };
    /// let msg = Message::assistant_with_tool_calls(vec![tool_call]);
    /// ```
    #[must_use]
    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Message::Assistant {
            content: None,
            tool_calls,
            metadata: HashMap::new(),
        }
    }

    /// Creates a new tool message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Message;
    ///
    /// let msg = Message::tool("tool123", "The result is 42.");
    /// ```
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::Tool {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Returns the role of the message as a string
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Message;
    ///
    /// let msg = Message::user("Hello");
    /// assert_eq!(msg.role_str(), "user");
    /// ```
    #[must_use]
    pub fn role_str(&self) -> &'static str {
        match self {
            Message::System { .. } => "system",
            Message::User { .. } => "user",
            Message::Assistant { .. } => "assistant",
            Message::Tool { .. } => "tool",
        }
    }

    /// Adds metadata and returns a new message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Message;
    /// use serde_json::json;
    ///
    /// let msg = Message::user("Hello")
    ///     .with_metadata("priority", json!(5));
    /// ```
    #[must_use]
    pub fn with_metadata(self, key: impl Into<String>, value: serde_json::Value) -> Self {
        match self {
            Message::System { content, mut metadata } => {
                metadata.insert(key.into(), value);
                Message::System { content, metadata }
            }
            Message::User { content, name, mut metadata } => {
                metadata.insert(key.into(), value);
                Message::User { content, name, metadata }
            }
            Message::Assistant { content, tool_calls, mut metadata } => {
                metadata.insert(key.into(), value);
                Message::Assistant { content, tool_calls, metadata }
            }
            Message::Tool { tool_call_id, content, mut metadata } => {
                metadata.insert(key.into(), value);
                Message::Tool { tool_call_id, content, metadata }
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_content_serialization() {
        let text_content = Content::text("Hello, world!");
        let serialized = serde_json::to_string(&text_content).unwrap();
        assert_eq!(serialized, "\"Hello, world!\"");

        let parts_content = Content::parts(vec![
            ContentPart::text("Hello"),
            ContentPart::image_url("https://example.com/image.jpg"),
        ]);
        let serialized = serde_json::to_string(&parts_content).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_message_serialization() {
        // Test user message serialization
        let msg = Message::user("Hello, world!");
        let serialized = serde_json::to_string(&msg).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        // Check externally tagged enum serialization
        assert_eq!(parsed["role"], "user");
        
        // In the new format, content is a property within the User variant,
        // and for text content it's serialized as a string directly
        assert!(parsed.get("content").is_some());
        
        // Test with metadata and name
        let msg = Message::user_with_name("John", "Hello")
            .with_metadata("priority", json!(5));
        let serialized = serde_json::to_string(&msg).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        assert_eq!(parsed["role"], "user");
        assert!(parsed.get("name").is_some());
        assert_eq!(parsed["name"], "John");
        assert!(parsed.get("content").is_some());
        assert_eq!(parsed["priority"], 5);
    }

    #[test]
    fn test_system_message() {
        let msg = Message::system("You are a helpful assistant");
        match msg {
            Message::System { content, metadata } => {
                assert_eq!(content, "You are a helpful assistant");
                assert!(metadata.is_empty());
            }
            _ => panic!("Expected System variant"),
        }
    }

    #[test]
    fn test_user_message() {
        let msg = Message::user_with_name("John", "Hello");
        match msg {
            Message::User { content, name, metadata } => {
                assert_eq!(content, Content::Text("Hello".to_string()));
                assert_eq!(name, Some("John".to_string()));
                assert!(metadata.is_empty());
            }
            _ => panic!("Expected User variant"),
        }
    }

    #[test]
    fn test_assistant_message() {
        let msg = Message::assistant("I'll help you");
        match msg {
            Message::Assistant { content, tool_calls, metadata } => {
                assert_eq!(content, Some(Content::Text("I'll help you".to_string())));
                assert!(tool_calls.is_empty());
                assert!(metadata.is_empty());
            }
            _ => panic!("Expected Assistant variant"),
        }

        let tool_call = ToolCall {
            id: "call_123".to_string(),
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                arguments: "{\"location\":\"San Francisco\"}".to_string(),
            },
        };
        
        let msg = Message::assistant_with_tool_calls(vec![tool_call]);
        match msg {
            Message::Assistant { content, tool_calls, metadata } => {
                assert_eq!(content, None);
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call_123");
                assert!(metadata.is_empty());
            }
            _ => panic!("Expected Assistant variant"),
        }
    }

    #[test]
    fn test_tool_message() {
        let msg = Message::tool("call_123", "The weather is sunny");
        match msg {
            Message::Tool { tool_call_id, content, metadata } => {
                assert_eq!(tool_call_id, "call_123");
                assert_eq!(content, "The weather is sunny");
                assert!(metadata.is_empty());
            }
            _ => panic!("Expected Tool variant"),
        }
    }
}