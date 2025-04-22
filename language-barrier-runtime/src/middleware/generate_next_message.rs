use std::{
    sync::Arc,
    task::{Context, Poll},
};

use language_barrier_core::{
    HTTPLlmService, LLMService,
    error::{Error, Result},
    model::ModelInfo,
    provider::HTTPProvider,
};

use tower_service::Service;
use tracing::debug;

use crate::ops::{LlmM, LlmOp};

use super::BoxFuture;

/// Middleware that handles Chat operations
///
/// This middleware processes Chat operations in the request pipeline.
/// It stores a Chat instance, model and provider, creating a fresh HTTP client
/// for each request via the stateless send_chat_request function.
pub struct GenerateNextMessageService<S, M, P>
where
    M: ModelInfo + Clone,
    P: HTTPProvider<M> + Send + Sync + Clone + 'static,
{
    inner: S,
    provider: Arc<P>,
    model: Arc<M>,
}

impl<S, M, P> GenerateNextMessageService<S, M, P>
where
    M: ModelInfo + Clone,
    P: HTTPProvider<M> + Send + Sync + Clone + 'static,
{
    /// Creates a new ChatMiddleware with a model, provider, and default Chat settings
    pub fn new(inner: S, model: Arc<M>, provider: Arc<P>) -> Self {
        Self {
            inner,
            provider: provider.clone(),
            model: model.clone(),
        }
    }
}

impl<S, A, M, P> Service<LlmM<A>> for GenerateNextMessageService<S, M, P>
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
        // Clone model and provider to avoid borrowing self
        let model = self.model.clone();
        let provider = self.provider.clone();

        debug!("Executing");

        Box::pin(async move {
            match operation {
                Some(LlmOp::GenerateNextMessage { chat, next }) => {
                    debug!("Creating chat");
                    let svc = HTTPLlmService::new(*model, provider);
                    let response = svc.generate_next_message(&chat).await;
                    debug!("Done, delegating to next");

                    // Continue with the result
                    let next_program = next(response.map(|m| chat.add_message(m)));
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
