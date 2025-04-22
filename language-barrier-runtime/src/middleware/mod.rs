use std::pin::Pin;

use futures::Future;
use language_barrier_core::error::{Error, Result};
use tower_service::Service;

mod break_on_tool;
mod chat;
mod tool_executor;

// pub use auto_tool::AutoToolMiddleware;
pub use break_on_tool::BreakOnToolMiddleware;
pub use chat::ChatMiddleware;
pub use tool_executor::ToolExecutorMiddleware;

// Re-export tower types for convenience
pub use tower::ServiceBuilder;

/// Type alias for a pinned future with static lifetime
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// Helper function to convert a future into a BoxFuture
pub fn boxed<F, T>(future: F) -> BoxFuture<T>
where
    F: Future<Output = T> + Send + 'static,
{
    Box::pin(future)
}

/// The final interpreter that resolves Done operations
#[derive(Clone)]
pub struct FinalInterpreter;

impl FinalInterpreter {
    /// Creates a new FinalInterpreter
    pub fn new() -> Self {
        Self
    }
}

impl Default for FinalInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> Service<crate::ops::LlmM<A>> for FinalInterpreter
where
    A: Send + 'static,
{
    type Response = A;
    type Error = Error;
    type Future = BoxFuture<Result<Self::Response>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, program: crate::ops::LlmM<A>) -> Self::Future {
        match (program.op, program.result) {
            (None, Some(result)) => boxed(async move { Ok(result) }),
            (Some(crate::ops::LlmOp::Done { result }), None) => {
                boxed(
                    async move { result.map(|_| panic!("Cannot extract value from Done operation")) },
                )
            }
            _ => boxed(async move { Err(Error::Other(format!("Invalid program state").into())) }),
        }
    }
}
