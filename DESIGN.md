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

#### 2024-04-17: Chat Interface Design

1. **Chat Interface**: Created a high-level, user-friendly abstraction for LLM interactions:
   - Provides a simple API for sending messages and receiving responses
   - Manages conversation history automatically
   - Supports function calling and tool usage workflows
   - Handles different message types (user, assistant, system, function, tool)

2. **Simplified Message Types**:
   - Introduced `ChatMessage` as a more user-friendly representation of messages
   - Automatic conversion between internal `Message` and user-facing `ChatMessage` types
   - Support for displaying messages in a human-readable format
   - Maintains compatibility with the underlying provider API

3. **Response Handling**:
   - `ChatResponse` type wraps the provider's response and adds convenience methods
   - Easy access to common metadata like token usage
   - Detection of function calls and tool calls
   - Standardized format regardless of provider implementation

4. **Configuration Options**:
   - Chat-specific options separate from generation parameters
   - Support for system messages to set conversation context
   - History management with optional size limits to prevent context window overflow
   - Ability to change models or parameters mid-conversation

5. **Testing Support**:
   - Built-in mock provider for testing chat functionality
   - Direct integration with the mock provider ecosystem
   - Comprehensive tests for all features including conversation management

#### 2025-04-17: Tool System Implementation

1. **Tool Trait**: Created a standardized interface for tools that can be used by LLMs:
   - Defined through the `Tool` trait, which requires methods for name, description, parameter schema, and execution
   - Supports both synchronous and asynchronous execution via the `async_trait` crate
   - Compatible with OpenAI's function calling format
   - Easily extensible with custom tools

2. **Calculator Tool**: Implemented a simple calculator tool as an example:
   - Supports basic arithmetic operations (add, subtract, multiply, divide)
   - Handles parameter validation and error cases
   - Demonstrates the pattern for creating new tools

3. **Provider Integration**:
   - Extended the `LlmProvider` trait with methods for registering and executing tools
   - Added tool support to the `GenerationOptions` struct
   - Updated the `MockProvider` implementation to support tools

4. **Chat Interface**:
   - Added tool support to the `ChatOptions` struct
   - Implemented `send_message_with_tools` method for automatic tool execution
   - Added methods to add tools and set tool choice strategy
   - Extended the conversation flow to handle tool calls and responses

5. **Error Handling**:
   - Added specific error types for tool-related operations
   - Comprehensive validation of tool parameters

6. **Tests and Documentation**:
   - Added unit tests for the Tool trait and Calculator implementation
   - Added integration tests for the full tool workflow
   - Added comprehensive doctests showing tool usage

The tool system design follows our core design principles:
- **Provider-agnostic**: Tools work with any LLM provider
- **Type safety**: Parameter schemas provide clear contracts
- **Async-first**: Tools support async execution
- **Extensibility**: Easy to add new tools
- **Testability**: Full test coverage for the tool system
- **Error handling**: Clear error types and validation

The implementation aligns with the OpenAI function calling format, making it compatible with existing LLM providers while maintaining our abstraction layer.

#### 2025-04-17: Anthropic Provider Implementation

1. **Anthropic API Integration**:
   - Implemented the Anthropic provider that connects to Anthropic's API
   - Handles authentication with API keys
   - Supports system messages as a separate parameter (following Anthropic's API design)
   - Includes proper error handling and response parsing

2. **Message Format Conversion**:
   - Created bidirectional conversion between our message format and Anthropic's format
   - System messages are handled specially (as required by Anthropic's API)
   - Tool calls and responses are properly mapped between formats
   - Handles multimodal content (text and images)

3. **Tool Support**:
   - Implemented tool calling for Claude 3 models
   - Mapped our Tool trait to Anthropic's tool format
   - Supports both tool execution and tool response handling
   - Handles tool choice strategies (auto, none, specific, any)

4. **Model Management**:
   - Added all Claude models (Claude 3 Opus, Sonnet, Haiku, Claude 2, etc.)
   - Correctly configured capabilities for each model
   - Implemented capability checking for the models
   - Added caching to improve performance

5. **Testing and Validation**:
   - Comprehensive unit tests for all key functionality
   - Message format conversion tests ensure correct mapping
   - Tool calling tests verify proper integration

The Anthropic provider implementation follows our core design principles:
- **Provider-agnostic API**: The provider seamlessly integrates with our existing abstractions
- **Type safety**: Strong types for all Anthropic-specific structures
- **Async-first**: All operations are async for efficient network I/O
- **Extensibility**: Easy to extend with additional Anthropic-specific features
- **Testability**: Well-tested implementation with mock responses
- **Error handling**: Comprehensive error handling specific to Anthropic's API

#### 2025-04-18: Google Provider Implementation

1. **Google Generative AI API Integration**:
   - Implemented the Google provider that connects to the Google Generative AI API
   - Configured HTTP client with proper authentication via API key
   - Supports both direct API key authentication and other Google Cloud credentials (via project ID)
   - Includes comprehensive error handling for various Google API error scenarios
   - Designed for compatibility with Google's specific Gemini model families

2. **Message Format Conversion**:
   - Created bidirectional mapping between our message format and Google's format
   - Managed role conversion (Google uses "model" instead of "assistant")
   - Properly handled multimodal content for both text and images
   - Implemented special handling for Google's unique content structure
   - Ensured proper conversion for function/tool calls and responses

3. **Tool Support**:
   - Implemented function calling support for Gemini models
   - Mapped our Tool trait to Google's function calling format
   - Supported tool choice strategies (auto, none, specific)
   - Converted between Google's function declaration format and our tool format
   - Implemented proper tool response handling

4. **Model Management**:
   - Added all current Gemini models (1.0 and 1.5 series)
   - Correctly configured capabilities for each model
   - Implemented context window settings based on Google's documentation
   - Added caching to improve performance
   - Ensured proper model validation

5. **Testing and Documentation**:
   - Comprehensive unit tests for all key functionality
   - Message format conversion tests
   - Tool calling tests
   - Detailed docstrings and examples

The Google provider implementation follows our core design principles:
- **Provider-agnostic API**: The provider seamlessly integrates with our abstractions
- **Type safety**: Strong types for all Google-specific structures
- **Async-first**: All operations are async for efficient network I/O
- **Extensibility**: Easy to extend with additional Google-specific features
- **Testability**: Well-tested implementation with mock responses
- **Error handling**: Comprehensive error handling specific to Google's API

Key differences from other providers:
- Google uses "model" instead of "assistant" for role names
- Google has a different structure for content parts
- Google's function calling format differs slightly from OpenAI's
- Google has specific safety settings that are unique to their API
- Google's API URL structure includes the model ID in the path

## Future Directions

1. **Streaming**: Support for streaming responses from LLMs.
2. **Embeddings**: Unified API for generating and working with embeddings.
3. **Caching**: Built-in caching to reduce API calls.
4. **Rate Limiting**: Intelligent rate limiting to avoid provider quotas.
5. **Middleware**: Pluggable middleware for logging, metrics, etc.
6. **Local Models**: Support for running LLMs locally.
7. **Additional Provider Implementations**: Implementations for other major providers like OpenAI.
8. **Additional Tools**: Implement more tools like web search, weather, etc.

## Implementation Notes

The library uses the following Rust crates:
- `serde` and `serde_json` for serialization
- `async-trait` for async trait support
- `thiserror` for error handling
- `reqwest` for HTTP requests
- `tokio` for async runtime
- `regex` for pattern matching in the mock provider
- `http` for testing HTTP responses

Additional crates may be added as needed for specific features.