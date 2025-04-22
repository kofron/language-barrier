use std::task::{Context, Poll};

use language_barrier_core::{
    error::{Error, Result},
    message::Message,
};
use std::marker::PhantomData;
use tower_service::Service;

use crate::ops::{LlmM, LlmOp};

use super::BoxFuture;

/// Middleware that breaks execution on a specific tool being called
pub struct BreakOnToolMiddleware<S> {
    inner: S,
    target_tool_name: String,
    _phantom: PhantomData<fn() -> LlmM<()>>,
}

impl<S> BreakOnToolMiddleware<S> {
    /// Creates a new BreakOnToolMiddleware with a target tool name
    pub fn new(inner: S, target_tool_name: String) -> Self {
        Self {
            inner,
            target_tool_name,
            _phantom: PhantomData,
        }
    }
}

impl<S, A> Service<LlmM<A>> for BreakOnToolMiddleware<S>
where
    S: Service<LlmM<A>, Response = A, Error = Error> + Clone + Send + 'static,
    S::Future: Send + 'static,
    A: Send + 'static,
{
    type Response = A;
    type Error = Error;
    type Future = BoxFuture<Result<Self::Response>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut program: LlmM<A>) -> Self::Future {
        if let Some(LlmOp::Chat {
            messages,
            tools,
            next,
        }) = program.op.take()
        {
            let target_tool_name = self.target_tool_name.clone();
            let mut inner = self.inner.clone();
            
            // Create a handler function that is Send
            let handler = move |result: Result<Message>| -> LlmM<A> {
                // Check if the target tool was called
                if let Ok(response) = &result {
                    if let Message::Assistant { tool_calls, .. } = &response {
                        if tool_calls
                            .iter()
                            .any(|tc| tc.function.name == target_tool_name)
                        {
                            // If the target tool was called, stop and return the result
                            return LlmM::new(LlmOp::Done { result });
                        }
                    }
                }
                // Otherwise continue with the program
                next(result)
            };

            // Create a new program with the handler
            let next_program = LlmM::new(LlmOp::Chat {
                messages,
                tools,
                next: Box::new(handler),
            });

            // Execute the modified program
            Box::pin(async move {
                inner.call(next_program).await
            })
        } else {
            // Not our operation, pass through
            let future = self.inner.call(program);
            Box::pin(future)
        }
    }
}

impl<S> Clone for BreakOnToolMiddleware<S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            target_tool_name: self.target_tool_name.clone(),
            _phantom: PhantomData,
        }
    }
}
