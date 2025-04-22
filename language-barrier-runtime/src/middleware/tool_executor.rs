use std::task::{Context, Poll};

use language_barrier_core::{
    ToolDefinition,
    error::{Error, Result},
    message::ToolCall,
};

use tower_service::Service;
use std::marker::PhantomData;

use crate::ops::{LlmM, LlmOp, ToolResult};

use super::BoxFuture;

/// Middleware that executes tool calls using a ToolRegistry
pub struct ToolExecutorMiddleware<S, T: ToolDefinition> {
    inner: S,
    def: T,
    _phantom: PhantomData<fn() -> LlmM<()>>,
}

// Required trait implementations for our middleware
unsafe impl<S: Send, T: ToolDefinition + Send> Send for ToolExecutorMiddleware<S, T> {}
unsafe impl<S: Sync, T: ToolDefinition + Sync> Sync for ToolExecutorMiddleware<S, T> {}

impl<S, T> ToolExecutorMiddleware<S, T>
where
    T: ToolDefinition + Clone + Send + Sync + 'static,
{
    /// Creates a new ToolExecutorMiddleware with a ToolRegistry
    pub fn new(inner: S, def: T) -> Self {
        Self { 
            inner, 
            def,
            _phantom: PhantomData,
        }
    }

    // Execute the tool with the given tool call
    pub async fn execute_tool_call(&self, tool_call: &ToolCall) -> Result<ToolResult> {
        Self::execute_tool_call_static(&self.def, tool_call).await
    }
    
    // Static version that doesn't need &self
    pub async fn execute_tool_call_static(_def: &T, tool_call: &ToolCall) -> Result<ToolResult> 
where
    T: Sync,
{
        // This is a minimal implementation to make it compile
        Ok(ToolResult {
            tool_call_id: tool_call.id.clone(),
            content: String::new(), // Placeholder, will be replaced later
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
        
        // Extract the operation and result
        let operation = program.op.take();
        let result = program.result;
        
        Box::pin(async move {
            match operation {
                Some(LlmOp::ExecuteTool { tool_call, next }) => {
                    if tool_call.function.name == def.name() {
                        let tool_call_clone = tool_call.clone();
                        
                        // Call our own execute_tool_call method
                        let result = Self::execute_tool_call_static(&def, &tool_call_clone).await;
                        // Continue with the result
                        let next_program = next(result);
                        inner.call(next_program).await
                    } else {
                        // Not our tool, pass through
                        let program = LlmM::new(LlmOp::ExecuteTool { tool_call, next });
                        inner.call(program).await
                    }
                },
                Some(op) => {
                    // Repackage the operation and pass through
                    let repackaged = LlmM::new(op);
                    inner.call(repackaged).await
                },
                None => {
                    // If the op is None, then there should be a result
                    if let Some(result) = result {
                        Ok(result)
                    } else {
                        Err(Error::Other("Invalid program state: both op and result are None".into()))
                    }
                }
            }
        })
    }
}

impl<S, T> Clone for ToolExecutorMiddleware<S, T>
where
    S: Clone,
    T: ToolDefinition + Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            def: self.def.clone(),
            _phantom: PhantomData,
        }
    }
}
