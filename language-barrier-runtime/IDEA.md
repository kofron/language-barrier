# Toy Implementation Spec: LLM System with Free Monads and Tower

## 1. Overview

This document outlines a proof-of-concept implementation of an LLM system using free monads to model computations and Tower middleware to handle execution. The approach provides clean separation between defining what an LLM system should do (as data) versus how it should do it (via interpreters).

## 2. Key Components

### 2.1 Free Monad Operations (`src/ops.rs`)
```rust
pub enum LlmOp<Next> {
    // Query operation
    Query {
        input: String,
        next: Box<dyn FnOnce(String) -> Next>,
    },
    // Generate text with LLM
    Generate {
        prompt: String,
        next: Box<dyn FnOnce(String) -> Next>,
    },
    // Retrieve context (RAG)
    Retrieve {
        query: String,
        next: Box<dyn FnOnce(Vec<Document>) -> Next>,
    },
    // Done with final result
    Done {
        result: String,
    },
}

// The free monad wrapper
pub struct LlmM<A> {
    pub op: Option<LlmOp<LlmM<A>>>,
    pub result: Option<A>,
}
```

### 2.2 Tower Middleware (`src/middleware/`)
```rust
// Create middleware for each operation
pub struct GenerateMiddleware<S> {
    inner: S,
    llm_client: LlmClient,
}

pub struct RetrieveMiddleware<S> {
    inner: S,
    retrieval_engine: RetrievalEngine,
}

// All middleware implement the Tower Service trait
impl<S, A> Service<LlmM<A>> for GenerateMiddleware<S>
where
    S: Service<LlmM<A>, Response = A, Error = Error>,
{
    type Response = A;
    type Error = Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut program: LlmM<A>) -> Self::Future {
        // Check if this is a Generate operation we should handle
        if let Some(LlmOp::Generate { prompt, next }) = program.op.take() {
            let llm_client = self.llm_client.clone();
            let mut inner = self.inner.clone();

            Box::pin(async move {
                // Execute the operation
                let generated_text = llm_client.generate(prompt).await?;

                // Continue with the result
                let next_program = next(generated_text);
                inner.call(next_program).await
            })
        } else {
            // Not our operation, pass through
            let future = self.inner.call(program);
            Box::pin(future)
        }
    }
}
```

### 2.3 Program Construction Helpers (`src/dsl.rs`)
```rust
// Helper functions to build programs
pub fn query(input: String) -> LlmM<String> {
    LlmM {
        op: Some(LlmOp::Query {
            input,
            next: Box::new(|result| LlmM {
                op: None,
                result: Some(result),
            }),
        }),
        result: None,
    }
}

pub fn generate(prompt: String) -> LlmM<String> {
    // Similar implementation
}

// Monadic combinators
impl<A> LlmM<A> {
    pub fn and_then<B, F>(self, f: F) -> LlmM<B>
    where
        F: 'static + FnOnce(A) -> LlmM<B>,
    {
        match (self.op, self.result) {
            (None, Some(result)) => f(result),
            (Some(op), None) => {
                // Map the continuation
                match op {
                    LlmOp::Generate { prompt, next } => LlmM {
                        op: Some(LlmOp::Generate {
                            prompt,
                            next: Box::new(move |res| {
                                let next_program = next(res);
                                next_program.and_then(f)
                            }),
                        }),
                        result: None,
                    },
                    // Other operations similar...
                }
            }
            _ => panic!("Invalid state"),
        }
    }
}
```

## 3. Implementation Steps

1. **Core Data Types**
   - Define the `LlmOp` enum for all operations
   - Create the `LlmM` free monad wrapper
   - Implement monadic operations (`and_then`, `map`)

2. **Tower Middleware**
   - Implement `Service` trait for each operation type
   - Create test/mock versions of each service
   - Build middleware stack constructor

3. **Runtime Validation**
   - Create a validation middleware as discussed
   - Add tracing/logging middleware

4. **Example Programs**
   - Create utility functions for common program patterns
   - Demonstrate combining operations

## 4. Example Program

```rust
fn build_pipeline() -> impl Service<LlmM<String>, Response = String> {
    ServiceBuilder::new()
        .layer(TracingLayer::new())
        .layer_fn(|inner| GenerateMiddleware::new(inner, llm_client))
        .layer_fn(|inner| RetrieveMiddleware::new(inner, retrieval_engine))
        .service(FinalInterpreter::new())
}

fn create_rag_program(query: String) -> LlmM<String> {
    // This defines the flow of operations
    retrieve(query.clone()).and_then(|documents| {
        let context = documents.iter().map(|d| d.content.clone()).collect::<Vec<_>>().join("\n");
        let prompt = format!("Context: {}\n\nQuery: {}\n\nResponse:", context, query);
        generate(prompt)
    })
}

async fn execute() {
    let program = create_rag_program("How do free monads work?".to_string());
    let mut service = build_pipeline();

    // Optional validation
    validate_program(&program, &service)?;

    // Execute
    let result = service.call(program).await?;
    println!("Result: {}", result);
}
```

## 5. Validation

```rust
struct ValidationMiddleware<S> {
    inner: S,
    supported_ops: HashSet<TypeId>,
    missing_handlers: Vec<String>,
}

fn validate_program<A>(
    program: &LlmM<A>,
    service: &impl Service<LlmM<A>>
) -> Result<(), ValidationError> {
    // Create validation middleware
    let validation = ValidationMiddleware::new(service);

    // Clone program (it's just data) and run validation
    let result = validation.call(program.clone()).await;

    if validation.missing_handlers.is_empty() {
        Ok(())
    } else {
        Err(ValidationError {
            missing_handlers: validation.missing_handlers,
        })
    }
}
```

This spec provides a clear starting point for implementing the free monad + Tower architecture for an LLM system. The toy implementation should be functional enough to demonstrate the approach while being small enough to implement in a reasonable timeframe.
