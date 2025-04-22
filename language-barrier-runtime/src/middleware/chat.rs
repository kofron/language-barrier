use std::task::{Context, Poll};

use language_barrier_core::{
    Chat,
    error::{Error, Result},
    message::Message,
    model::ModelInfo,
    provider::HTTPProvider,
    tool::{ToolChoice, ToolDefinition},
};
use reqwest::Client;
use std::marker::PhantomData;
use tower_service::Service;
use tracing::{debug, error, info, trace};

use crate::ops::{LlmM, LlmOp};

use super::BoxFuture;

/// Send a chat request to an LLM provider and return the response message.
///
/// This is a stateless function that replicates the functionality of LLMService
/// without storing any state. It creates a fresh HTTP client for each request.
///
/// # Arguments
///
/// * `chat` - The Chat instance to send
/// * `model` - The model to use
/// * `provider` - The HTTPProvider implementation to use for converting the chat to a request
/// * `client` - Optional HTTP client to use. If None, a new client will be created.
///
/// # Returns
///
/// A Result containing the response Message on success, or an Error on failure.
pub async fn send_chat_request<M: ModelInfo>(
    chat: Chat,
    model: M,
    provider: &(dyn HTTPProvider<M> + Send + Sync),
    client: Option<Client>,
) -> Result<Message> {
    info!("Sending chat request with {} messages", chat.history.len());
    trace!("System prompt: {}", chat.system_prompt);

    // Convert chat to request using provider
    debug!("Converting chat to HTTP request");
    let request = match provider.accept(std::sync::Arc::new(model), std::sync::Arc::new(chat)) {
        Ok(req) => {
            debug!(
                "Request created successfully: {} {}",
                req.method(),
                req.url()
            );
            trace!("Request headers: {:#?}", req.headers());
            req
        }
        Err(e) => {
            error!("Failed to create request: {}", e);
            return Err(e);
        }
    };

    // Use provided client or create a new one
    let client = client.unwrap_or_default();

    // Send request and get response
    debug!("Sending HTTP request");
    let response = match client.execute(request).await {
        Ok(resp) => {
            info!("Received response with status: {}", resp.status());
            trace!("Response headers: {:#?}", resp.headers());
            resp
        }
        Err(e) => {
            error!("HTTP request failed: {}", e);
            return Err(e.into());
        }
    };

    // Get response text
    debug!("Reading response body");
    let response_text = match response.text().await {
        Ok(text) => {
            trace!("Response body: {}", text);
            text
        }
        Err(e) => {
            error!("Failed to read response body: {}", e);
            return Err(e.into());
        }
    };

    // Parse response using provider
    debug!("Parsing response");
    let message = match provider.parse(response_text) {
        Ok(msg) => {
            info!("Successfully parsed response into message");
            debug!("Message role: {}", msg.role_str());
            msg
        }
        Err(e) => {
            error!("Failed to parse response: {}", e);
            return Err(e);
        }
    };

    Ok(message)
}

/// Middleware that handles Chat operations
///
/// This middleware processes Chat operations in the request pipeline.
/// It stores a Chat instance, model and provider, creating a fresh HTTP client
/// for each request via the stateless send_chat_request function.
pub struct ChatMiddleware<S, M, P>
where
    M: ModelInfo + Clone,
    P: HTTPProvider<M> + Send + Sync + Clone + 'static,
{
    inner: S,
    provider: P,
    model: M,
    chat: Chat,
    _phantom: PhantomData<fn() -> LlmM<()>>,
}

// Required trait implementations for our middleware
unsafe impl<S: Send, M: ModelInfo + Clone + Send, P: HTTPProvider<M> + Send + Sync + Clone> Send
    for ChatMiddleware<S, M, P>
{
}
unsafe impl<S: Sync, M: ModelInfo + Clone + Sync, P: HTTPProvider<M> + Send + Sync + Clone> Sync
    for ChatMiddleware<S, M, P>
{
}

impl<S, M, P> ChatMiddleware<S, M, P>
where
    M: ModelInfo + Clone,
    P: HTTPProvider<M> + Send + Sync + Clone + 'static,
{
    /// Creates a new ChatMiddleware with a model, provider, and default Chat settings
    pub fn new(inner: S, model: M, provider: P) -> Self {
        Self {
            inner,
            provider,
            model: model.clone(),
            chat: Chat::new(),
            _phantom: PhantomData,
        }
    }

    /// Creates a new ChatMiddleware from an existing Chat instance
    pub fn with_chat(inner: S, chat: Chat, provider: P, model: M) -> Self {
        Self {
            inner,
            provider,
            model,
            chat,
            _phantom: PhantomData,
        }
    }

    /// Set the system prompt for this middleware
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        let prompt_string = prompt.into();
        self.chat = self.chat.with_system_prompt(prompt_string);
        self
    }

    /// Set the maximum output tokens
    pub fn with_max_output_tokens(mut self, tokens: usize) -> Self {
        self.chat = self.chat.with_max_output_tokens(tokens);
        self
    }

    /// Add a tool to the chat
    pub fn with_tool(mut self, tool: impl ToolDefinition) -> Result<Self> {
        self.chat = self.chat.with_tool(tool)?;
        Ok(self)
    }

    /// Set the tool choice strategy
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.chat = self.chat.with_tool_choice(choice);
        self
    }
}

impl<S, A, M, P> Service<LlmM<A>> for ChatMiddleware<S, M, P>
where
    S: Service<LlmM<A>, Response = A, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
    A: Send + 'static,
    M: ModelInfo + Clone + Send + Sync + Default + 'static,
    P: HTTPProvider<M> + Send + Sync + Clone + 'static,
{
    type Response = A;
    type Error = Error;
    type Future = BoxFuture<Result<Self::Response>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut program: LlmM<A>) -> Self::Future {
        debug!("Starting call in chat middleware");
        // Extract all the data we need outside the async block
        let mut inner = self.inner.clone();
        let operation = program.op.take();
        let result = program.result;

        // Clone everything we need for the async block
        let provider = self.provider.clone();

        // For Chat, we need to recreate it since it doesn't implement Clone
        let model = self.chat.model.clone();
        let system_prompt = self.chat.system_prompt.clone();
        let max_output_tokens = self.chat.max_output_tokens;
        let history = self.chat.history.clone();
        let tools = self.chat.tools.clone();
        let tool_choice = self.chat.tool_choice.clone();

        debug!("Executing");

        Box::pin(async move {
            match operation {
                Some(LlmOp::Chat {
                    messages,
                    tools: op_tools,
                    next,
                }) => {
                    debug!("Creating chat");
                    // Create a new chat with the same settings
                    let mut chat = Chat::new()
                        .with_system_prompt(system_prompt)
                        .with_max_output_tokens(max_output_tokens);

                    // Add existing history
                    for msg in history {
                        chat = chat.add_message(msg);
                    }

                    // Add new messages
                    for msg in messages {
                        chat = chat.add_message(msg);
                    }
                    debug!("Chat messages added");

                    // Add existing tools or new tools if provided
                    if let Some(tool_list) = tools {
                        chat = chat.with_tools(tool_list);
                    }

                    if let Some(tool_list) = op_tools {
                        chat = chat.with_tools(tool_list);
                    }

                    // Set tool choice if it exists
                    if let Some(choice) = tool_choice {
                        chat = chat.with_tool_choice(choice);
                    }
                    debug!("Sending chat request");
                    // Send the request using our stateless function
                    let response = send_chat_request(chat, model.clone(), &provider, None).await;
                    debug!("Done, delegating to next");

                    // Continue with the result
                    let next_program = next(response);
                    inner.call(next_program).await
                }
                Some(LlmOp::AddMessage {
                    chat,
                    message,
                    next,
                }) => {
                    debug!("adding message");
                    // Try to downcast the boxed chat to the specific type
                    let result = match chat.downcast::<Chat>() {
                        Ok(chat_box) => {
                            // Extract the Chat from the box
                            let chat = *chat_box;

                            // Immutably add the message to create a new Chat instance
                            let updated_chat = chat.add_message(message);
                            debug!("added message");

                            // Return the updated chat in a new box
                            let boxed_updated: Box<dyn std::any::Any + Send> =
                                Box::new(updated_chat);
                            Ok(boxed_updated)
                        }
                        Err(original_box) => {
                            // If we can't downcast, return an error with the original box
                            debug!("downcast didnt work");
                            Err(Error::Other(format!(
                                "ChatMiddleware couldn't handle Chat of unknown type. Expected Chat"
                            )))
                        }
                    };

                    // Continue with the result
                    let next_program = next(result);
                    inner.call(next_program).await
                }
                Some(op) => {
                    // Not our operation, repackage and pass through
                    let repackaged = LlmM::new(op);
                    inner.call(repackaged).await
                }
                None => {
                    // If the op is None, then there should be a result
                    if let Some(result) = result {
                        Ok(result)
                    } else {
                        Err(Error::Other(
                            "Invalid program state: both op and result are None".into(),
                        ))
                    }
                }
            }
        })
    }
}

impl<S, M, P> Clone for ChatMiddleware<S, M, P>
where
    S: Clone,
    M: ModelInfo + Clone,
    P: HTTPProvider<M> + Send + Sync + Clone,
{
    fn clone(&self) -> Self {
        // We need to manually recreate the chat since it doesn't implement Clone
        let system_prompt = self.chat.system_prompt.clone();
        let max_output_tokens = self.chat.max_output_tokens;
        let history = self.chat.history.clone();
        let tools = self.chat.tools.clone();
        let tool_choice = self.chat.tool_choice.clone();

        // Create a new chat with the same settings
        let mut chat = Chat::new()
            .with_system_prompt(system_prompt)
            .with_max_output_tokens(max_output_tokens);

        // Add history
        for msg in history {
            chat = chat.add_message(msg);
        }

        // Add tools if any
        if let Some(tool_list) = tools {
            chat = chat.with_tools(tool_list);
        }

        // Set tool choice if it exists
        if let Some(choice) = tool_choice {
            chat = chat.with_tool_choice(choice);
        }

        Self {
            inner: self.inner.clone(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            chat,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::FinalInterpreter;
    use crate::ops;
    use language_barrier_core::{
        message::{Content, Message},
        model::Claude,
    };
    use std::collections::HashMap;

    // Mock HTTPProvider for testing
    struct MockProvider;

    impl HTTPProvider<Claude> for MockProvider {
        fn accept(&self, _model: Arc<Claude>, _chat: Arc<Chat>) -> language_barrier_core::Result<reqwest::Request> {
            // Mock a request
            let client = reqwest::Client::new();
            let req = client.get("https://example.com").build().unwrap();
            Ok(req)
        }

        fn parse(&self, _raw_response_text: String) -> language_barrier_core::Result<Message> {
            // Mock a response message
            Ok(Message::Assistant {
                content: Some(Content::Text("Hello, world!".to_string())),
                metadata: HashMap::new(),
                tool_calls: vec![],
            })
        }
    }

    impl Clone for MockProvider {
        fn clone(&self) -> Self {
            MockProvider
        }
    }

    #[tokio::test]
    async fn test_chat_middleware() {
        // Create a model
        let model = Claude::Opus3;
        
        // Create a chat instance
        let chat = Chat::new().with_system_prompt("You are a helpful assistant.");

        // Create the middleware with the chat
        let middleware = ChatMiddleware::with_chat(FinalInterpreter::new(), chat, MockProvider, model);

        // Create a program with a Chat operation
        let program = ops::chat(
            vec![Message::User {
                content: Content::Text("Hello!".to_string()),
                metadata: HashMap::new(),
                name: None,
            }],
            None,
        );

        // Execute the program
        let mut service = middleware;
        let result = service.call(program).await;

        // Verify the result
        assert!(result.is_ok());
        if let Ok(message_result) = result {
            assert!(message_result.is_ok());
            if let Ok(message) = message_result {
                match message {
                    Message::Assistant { content, .. } => {
                        assert_eq!(content, Some(Content::Text("Hello, world!".to_string())));
                    }
                    _ => panic!("Expected assistant message"),
                }
            }
        }
    }

    #[test]
    fn test_chat_middleware_clone() {
        // Create a model
        let model = Claude::Opus3;
        
        // Create a chat instance with specific settings
        let chat = Chat::new()
            .with_system_prompt("You are a helpful assistant.")
            .with_max_output_tokens(1000)
            .add_message(Message::User {
                content: Content::Text("Hi there".to_string()),
                metadata: HashMap::new(),
                name: None,
            });

        // Create the middleware with the chat
        let middleware = ChatMiddleware::with_chat(FinalInterpreter::new(), chat, MockProvider, model)
            .with_tool_choice(ToolChoice::Auto);

        // Clone the middleware
        let cloned = middleware.clone();

        // Verify that the cloned middleware has the same settings
        assert_eq!(cloned.chat.system_prompt, "You are a helpful assistant.");
        assert_eq!(cloned.chat.max_output_tokens, 1000);
        assert_eq!(cloned.chat.history.len(), 1);
        assert_eq!(cloned.chat.tool_choice, Some(ToolChoice::Auto));
    }

    #[tokio::test]
    async fn test_add_message_operation() {
        // Create a model
        let model = Claude::Opus3;
        
        // Create a chat instance
        let chat = Chat::new().with_system_prompt("You are a helpful assistant.");

        // Create a message to add
        let message = Message::User {
            content: Content::Text("Hello, how can you help me?".to_string()),
            metadata: HashMap::new(),
            name: None,
        };

        // Create the middleware
        let mut middleware =
            ChatMiddleware::with_chat(FinalInterpreter::new(), chat.clone(), MockProvider, model);

        // Create the add_message program
        let program = ops::add_message(chat, message.clone());

        // Execute the program
        let result = middleware.call(program).await;

        // Verify the result
        assert!(result.is_ok());
        if let Ok(chat_result) = result {
            assert!(chat_result.is_ok());
            if let Ok(updated_chat) = chat_result {
                // Check that the message was added
                assert_eq!(updated_chat.history.len(), 1);
                assert!(matches!(updated_chat.history[0], Message::User { .. }));

                // Check it's the right message
                if let Message::User { content, .. } = &updated_chat.history[0] {
                    if let Content::Text(text) = content {
                        assert_eq!(text, "Hello, how can you help me?");
                    } else {
                        panic!("Expected text content");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_add_message_with_helper() {
        // Create a model
        let model = Claude::Opus3;
        
        // Create a chat instance
        let chat = Chat::new().with_system_prompt("You are a helpful assistant.");

        // Create the middleware
        let mut middleware =
            ChatMiddleware::with_chat(FinalInterpreter::new(), chat.clone(), MockProvider, model);

        // Create a user message using the helper function
        let message = ops::user_message("Hello, how can you help me?");

        // Create the add_message program
        let program = ops::add_message(chat, message);

        // Execute the program
        let result = middleware.call(program).await;

        // Verify the result
        assert!(result.is_ok());
        if let Ok(chat_result) = result {
            assert!(chat_result.is_ok());
            if let Ok(updated_chat) = chat_result {
                // Check that the message was added
                assert_eq!(updated_chat.history.len(), 1);

                // Check it's the right message
                if let Message::User { content, .. } = &updated_chat.history[0] {
                    if let Content::Text(text) = content {
                        assert_eq!(text, "Hello, how can you help me?");
                    } else {
                        panic!("Expected text content");
                    }
                } else {
                    panic!("Expected user message");
                }
            }
        }
    }
}
