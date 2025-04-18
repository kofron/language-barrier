# Language Barrier Design Document

## Overview

Language Barrier is a Rust library that provides abstractions for working with Large Language Models (LLMs) from different providers. The goal is to create a unified API that allows developers to easily switch between different LLM providers without changing their application code.

## Core Design Principles

1. **Provider-agnostic API**: Applications should be able to use any LLM provider with minimal code changes.
2. **Type safety**: Leverage Rust's type system to catch errors at compile time.
3. **Async-first**: All operations that involve network calls are async.
4. **Extensibility**: Easy to add support for new providers and models.
5. **Testability**: Components are designed to be easily testable.
6. **Error handling**: Comprehensive error handling that provides meaningful feedback.

## Architecture

The library is organized into the following components:

### Core Components

1. **Message Types**: The foundation of the library, representing the messages exchanged with LLMs.
2. **Models**: Representations of specific LLM models and their capabilities.
3. **Providers**: Abstractions for different LLM providers (OpenAI, Anthropic, etc.).
4. **Clients**: Provider-specific implementations of the communication with LLM APIs.
5. **Responses**: Structured representations of LLM responses.

### Design Decisions

#### 2024-04-17: Initial Design

1. **Message Format**: We've chosen to base our message format on the OpenAI chat completion API, as it has become a de-facto standard. This includes:
   - Message roles (system, user, assistant, function, tool)
   - Support for multimodal content (text, images)
   - Function calling capabilities

2. **Model Capabilities**: Models have explicit capabilities (like chat completion, vision, embeddings) to allow runtime checking of whether a model supports a particular feature.

3. **Provider Abstraction**: Each provider has its own module with client implementation that handles authentication, rate limiting, and other provider-specific details.

4. **Error Handling**: A centralized error type with variants for different kinds of errors (authentication, rate limiting, network, etc.).

5. **Async API**: Using `async-trait` to allow async methods in traits, enabling a clean interface for operations that require network I/O.

#### 2024-04-17: LlmProvider Trait Design

1. **LlmProvider Trait**: Created a core abstraction for interacting with LLMs:
   - Async methods for all operations to support network I/O
   - Clear contract that specifies message generation, model listing, and capability checking
   - Support for configuration options through the `GenerationOptions` struct
   - Function and tool calling support through the response format

2. **MockProvider Implementation**:
   - Simple mock implementation that returns predefined responses
   - Supports pattern matching to determine response based on message content
   - Fully tested with comprehensive unit tests
   - Designed for development and testing without external dependencies

3. **AdvancedMockProvider Implementation**:
   - Extension of the basic MockProvider that adds tool and function calling simulation
   - Enables testing of tool/function calling capabilities without real API usage
   - Uses composition pattern to extend the base implementation

4. **Generation Options**:
   - Flexible parameter passing through a builder pattern
   - Support for common LLM parameters like temperature and top_p
   - Extensible through extra_params for provider-specific parameters

## Future Directions

1. **Streaming**: Support for streaming responses from LLMs.
2. **Function Calling**: Improved support for function calling and tools.
3. **Embeddings**: Unified API for generating and working with embeddings.
4. **Caching**: Built-in caching to reduce API calls.
5. **Rate Limiting**: Intelligent rate limiting to avoid provider quotas.
6. **Middleware**: Pluggable middleware for logging, metrics, etc.
7. **Local Models**: Support for running LLMs locally.
8. **Real Provider Implementations**: Implementations for major providers like OpenAI, Anthropic, etc.

## Implementation Notes

The library uses the following Rust crates:
- `serde` and `serde_json` for serialization
- `async-trait` for async trait support
- `thiserror` for error handling
- `reqwest` for HTTP requests
- `tokio` for async runtime

Additional crates may be added as needed for specific features.