use std::{
    sync::Arc,
    task::{Context, Poll},
};

use language_barrier_core::{
    ToolDefinition,
    error::{Error, Result},
    message::ToolCall,
};

use tower_service::Service;

use crate::ops::{LlmM, LlmOp, ToolResult};

use super::BoxFuture;

/// Middleware that executes tool calls using a ToolRegistry
pub struct ToolExecutorMiddleware<S, T: ToolDefinition> {
    inner: S,
    def: T,
    f: Arc<dyn Fn(T::Input) -> T::Output + Send + Sync>,
    auto_execute: bool,
}

impl<S, T> ToolExecutorMiddleware<S, T>
where
    T: ToolDefinition + Clone + Send + Sync + 'static,
{
    /// Creates a new ToolExecutorMiddleware with a ToolRegistry
    pub fn new(inner: S, def: T, f: Arc<dyn Fn(T::Input) -> T::Output + Send + Sync>) -> Self {
        Self { 
            inner, 
            def, 
            f,
            auto_execute: false 
        }
    }
    
    /// Creates a new ToolExecutorMiddleware with auto-execute mode enabled
    pub fn with_auto_execute(inner: S, def: T, f: Arc<dyn Fn(T::Input) -> T::Output + Send + Sync>) -> Self {
        Self { 
            inner, 
            def, 
            f,
            auto_execute: true 
        }
    }

    // Execute the tool with the given tool call
    fn execute_tool_call(_def: &T, f: &Arc<dyn Fn(T::Input) -> T::Output + Send + Sync>, tool_call: &ToolCall) -> Result<String> {
        tracing::debug!("Executing tool call: {:?}", tool_call);
        tracing::debug!("Tool arguments: {}", tool_call.function.arguments);
        
        let inp: T::Input = match serde_json::from_str(tool_call.function.arguments.as_str()) {
            Ok(inp) => {
                tracing::debug!("Deserialized tool input successfully");
                inp
            },
            Err(e) => {
                tracing::error!("Failed to deserialize tool input: {}", e);
                return Err(language_barrier_core::Error::Serialization(e));
            }
        };
        
        let result = f(inp);
        tracing::debug!("Tool execution completed, serializing result");
        
        serde_json::to_string(&result).map_err(|e| {
            tracing::error!("Failed to serialize tool output: {}", e);
            language_barrier_core::Error::from(
                language_barrier_core::ToolError::OutputTypeMismatch(
                    format!("Failed to serialize tool output to string: {}", e)
                ),
            )
        })
    }
}

impl<S, A, T> Service<LlmM<A>> for ToolExecutorMiddleware<S, T>
where
    S: Service<LlmM<A>, Response = A, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
    A: Send + 'static,
    T: ToolDefinition + Clone + Send + Sync + 'static,
{
    type Response = A;
    type Error = Error;
    type Future = BoxFuture<Result<Self::Response>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut program: LlmM<A>) -> Self::Future {
        // Clone things we need to move into the async block
        let mut inner = self.inner.clone();
        let def = self.def.clone();
        let f = self.f.clone();
        let auto_execute = self.auto_execute;

        // Extract the operation and result
        let operation = program.op.take();
        let result = program.result;

        Box::pin(async move {
            match operation {
                Some(LlmOp::GenerateNextMessage { chat, next }) => {
                    // If auto-execute is disabled, just pass through directly
                    if !auto_execute {
                        let pass_through = LlmM::new(LlmOp::GenerateNextMessage { 
                            chat, 
                            next 
                        });
                        return inner.call(pass_through).await;
                    }
                    
                    // With auto-execute, we intercept the message and potentially add our own processing
                    tracing::debug!("Auto-execute mode is enabled for GenerateNextMessage");
                    
                    // Try to detect if the last message is from the user (helps decide if we should arm)
                    let should_arm = if let Some(last_message) = chat.history.last() {
                        matches!(last_message, language_barrier_core::message::Message::User { .. })
                    } else {
                        false
                    };
                    
                    // Only enable auto-execution if the last message was from a user
                    if !should_arm {
                        tracing::debug!("Last message is not from user, not arming auto-execution");
                        let pass_through = LlmM::new(LlmOp::GenerateNextMessage { 
                            chat, 
                            next 
                        });
                        return inner.call(pass_through).await;
                    }
                    
                    // We're now armed for auto-execution
                    // First, create a modified GenerateNextMessage operation with a custom next function
                    let auto_next = Box::new(move |chat_result: Result<language_barrier_core::chat::Chat>| {
                        // This closure will run after the message is generated, but before the next continuation
                        if let Ok(updated_chat) = &chat_result {
                            // Check if the last message has tool calls for our tool
                            if let Some(last_message) = updated_chat.history.last() {
                                if let language_barrier_core::message::Message::Assistant { tool_calls, .. } = last_message {
                                    if !tool_calls.is_empty() {
                                        // There are tool calls in the response
                                        let our_tool_calls: Vec<_> = tool_calls.iter()
                                            .filter(|tc| tc.function.name == def.name())
                                            .collect();
                                            
                                        if !our_tool_calls.is_empty() {
                                            tracing::debug!("Found {} tool calls for our tool", our_tool_calls.len());
                                            
                                            // Create a modified chat with tool calls executed
                                            let mut chat_with_tools = updated_chat.clone();
                                            
                                            for tool_call in &our_tool_calls {
                                                if let Ok(content) = Self::execute_tool_call(&def, &f, tool_call) {
                                                    tracing::debug!("Executed tool '{}', adding result to chat", def.name());
                                                    
                                                    // Add the tool response to the chat
                                                    let tool_message = language_barrier_core::message::Message::Tool {
                                                        tool_call_id: tool_call.id.clone(),
                                                        content,
                                                        metadata: std::collections::HashMap::new(),
                                                    };
                                                    
                                                    chat_with_tools = chat_with_tools.add_message(tool_message);
                                                }
                                            }
                                            
                                            // Our next operation will be to generate another message
                                            // with the tool results included
                                            return LlmM::new(LlmOp::GenerateNextMessage {
                                                chat: chat_with_tools,
                                                next: Box::new(move |final_result| {
                                                    // Once we get the final result, pass it to the original next function
                                                    next(final_result)
                                                }),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                        
                        // If we didn't trigger auto-execution, just continue with the original result
                        next(chat_result)
                    });
                    
                    // Create the operation
                    let generate_op = LlmM::new(LlmOp::GenerateNextMessage {
                        chat,
                        next: auto_next,
                    });
                    
                    // Pass the operation to the inner service
                    inner.call(generate_op).await
                }
                Some(LlmOp::ExecuteTool { tool_call, next }) => {
                    if tool_call.function.name == def.name() {
                        let tool_call_clone = tool_call.clone();
                        // Call the static execute function
                        let result = Self::execute_tool_call(&def, &f, &tool_call_clone)
                            .map(|s| ToolResult {
                                content: s,
                                tool_call_id: tool_call.id,
                            });
                        // Continue with the result
                        let next_program = next(result);
                        inner.call(next_program).await
                    } else {
                        // Not our tool, pass through
                        let program = LlmM::new(LlmOp::ExecuteTool { tool_call, next });
                        inner.call(program).await
                    }
                }
                Some(op) => {
                    // Repackage the operation and pass through
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
