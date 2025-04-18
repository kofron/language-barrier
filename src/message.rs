use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents the role of a message in a conversation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// Message from the system (instructions)
    System,
    /// Message from the user
    User,
    /// Message from the assistant
    Assistant,
    /// Message containing a function call
    Function,
    /// Message from a tool
    Tool,
}

impl MessageRole {
    /// Returns a string representation of the role
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::MessageRole;
    ///
    /// let role = MessageRole::User;
    /// assert_eq!(role.as_str(), "user");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Function => "function",
            MessageRole::Tool => "tool",
        }
    }
}

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
    pub fn is_empty(&self) -> bool {
        match self {
            Content::Text(text) => text.is_empty(),
            Content::Parts(parts) => parts.is_empty() || parts.iter().all(|p| p.is_empty()),
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
    /// use language_barrier::message::{ContentPart, ImageUrl};
    ///
    /// let image_url = ImageUrl::new("https://example.com/image.jpg");
    /// let part = ContentPart::image_url(image_url);
    /// ```
    pub fn image_url(image_url: ImageUrl) -> Self {
        ContentPart::ImageUrl { image_url }
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
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// Represents a function call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The name of the function
    pub name: String,
    /// The arguments to the function
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
    /// The function call
    pub function: FunctionCall,
}

/// Represents a message in a conversation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender
    pub role: MessageRole,
    /// The content of the message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    /// The name of the sender (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The function call (for assistant messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    /// The tool calls (for assistant messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The ID of the tool call this message is responding to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Additional provider-specific metadata
    #[serde(flatten, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
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
        Self {
            role: MessageRole::System,
            content: Some(Content::Text(content.into())),
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
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
        Self {
            role: MessageRole::User,
            content: Some(Content::Text(content.into())),
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
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
        Self {
            role: MessageRole::Assistant,
            content: Some(Content::Text(content.into())),
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Creates a new function message
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Message;
    ///
    /// let msg = Message::function("get_weather", "The weather is sunny.");
    /// ```
    pub fn function(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Function,
            content: Some(Content::Text(content.into())),
            name: Some(name.into()),
            function_call: None,
            tool_calls: None,
            tool_call_id: None,
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
        Self {
            role: MessageRole::Tool,
            content: Some(Content::Text(content.into())),
            name: None,
            function_call: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            metadata: HashMap::new(),
        }
    }

    /// Sets the name and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::Message;
    ///
    /// let msg = Message::user("Hello")
    ///     .with_name("John");
    /// ```
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets multimodal content and returns self for method chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::message::{Message, ContentPart};
    ///
    /// let parts = vec![ContentPart::text("Hello")];
    /// let msg = Message::user("")
    ///     .with_content_parts(parts);
    /// ```
    pub fn with_content_parts(mut self, parts: Vec<ContentPart>) -> Self {
        self.content = Some(Content::Parts(parts));
        self
    }

    /// Adds metadata and returns self for method chaining
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
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_message_role_serialization() {
        let role = MessageRole::User;
        let serialized = serde_json::to_string(&role).unwrap();
        assert_eq!(serialized, "\"user\"");

        let deserialized: MessageRole = serde_json::from_str("\"assistant\"").unwrap();
        assert_eq!(deserialized, MessageRole::Assistant);
    }

    #[test]
    fn test_content_serialization() {
        let text_content = Content::text("Hello, world!");
        let serialized = serde_json::to_string(&text_content).unwrap();
        assert_eq!(serialized, "\"Hello, world!\"");

        let parts_content = Content::parts(vec![
            ContentPart::text("Hello"),
            ContentPart::image_url(ImageUrl::new("https://example.com/image.jpg")),
        ]);
        let serialized = serde_json::to_string(&parts_content).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("Hello, world!");
        let serialized = serde_json::to_string(&msg).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        assert_eq!(parsed["role"], "user");
        assert_eq!(parsed["content"], "Hello, world!");

        // Test with metadata
        let msg = Message::user("Hello")
            .with_metadata("priority", json!(5))
            .with_name("John");
        let serialized = serde_json::to_string(&msg).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        assert_eq!(parsed["role"], "user");
        assert_eq!(parsed["name"], "John");
        assert_eq!(parsed["priority"], 5);
    }
}
