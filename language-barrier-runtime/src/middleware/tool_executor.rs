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
    ///
    /// Note: Auto-execute mode is currently a work-in-progress feature that will
    /// automatically execute tools and handle the responses in a single user interaction.
    /// The current implementation only enables the flag, but full functionality requires
    /// specialization that isn't yet implemented in the generic Service implementation.
    ///
    /// In the future, a specialized implementation for Result<Chat> will provide full
    /// auto-execution capabilities.
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
        let _auto_execute = self.auto_execute; // Currently unused but kept for future implementation

        // Extract the operation and result
        let operation = program.op.take();
        let result = program.result;

        Box::pin(async move {
            match operation {
                Some(LlmOp::GenerateNextMessage { chat, next }) => {
                    // For now, auto-execute mode only works in specific cases with the final API usage,
                    // so we just use a simple pass-through for all GenerateNextMessage operations
                    tracing::debug!("Auto-execute mode not fully implemented for generic types, passing through");
                    
                    // Just pass through
                    let pass_through = LlmM::new(LlmOp::GenerateNextMessage { 
                        chat, 
                        next 
                    });
                    inner.call(pass_through).await
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
