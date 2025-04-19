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

#### 2025-04-18: Foundational Components Redesign and Tool Simplification

1. **Secret<T> Type**:
   - Created a dedicated wrapper type for sensitive information
   - Prevents accidental leakage in logs and debug output
   - Implements custom Debug and Display traits to hide secret values
   - Intentionally does not implement Serialize to prevent accidental serialization
   - Used for API keys and other credentials in provider configurations

2. **TokenCounter**:
   - Simple implementation for tracking token usage
   - Uses a naive whitespace-based tokenization for simplicity
   - Can be extended later with more sophisticated tokenizers
   - Provides methods for observing, subtracting, and checking against token budgets
   - Used with the ChatHistoryCompactor for conversation management

3. **ChatHistoryCompactor Trait**:
   - Strategy pattern for managing conversation history
   - Allows different algorithms for trimming history when token budget is exceeded
   - Maintains a clean separation of concerns between token counting and history management
   - `DropOldestCompactor` provided as a default implementation that removes oldest messages first
   - Designed to be extensible with other strategies (e.g., summarization)

4. **Tool System Simplification**:
   - Temporarily removed the tool system to simplify the refactoring process
   - Will be reintroduced later with a cleaner, more idiomatic design
   - Allowed us to focus on the core foundation components without dealing with object safety issues
   - Future implementation will better separate tool interfaces from provider-specific implementations

These new components follow our core design principles:
- **Type safety**: Strong types with clear semantics
- **Extensibility**: Easy to extend with additional implementations
- **Testability**: Comprehensive unit tests for all components
- **Error handling**: Clean handling of edge cases (like token budget exhaustion)
- **Modularity**: Components are designed to be used independently and integrated smoothly

#### 2025-04-18: Chat Interface Redesign

1. **Generic Chat Implementation**:
   - Completely redesigned Chat struct to be generic over model type and provider type
   - Implemented with clean separation of core functionality and provider-specific features
   - Simple fluent builder pattern for configuration
   - Internal token tracking and history management with automatic compaction

2. **Improved Provider Integration**:
   - Type-safe specialization for the LlmProvider trait 
   - Clear separation between generic chat functionality and provider-specific operations
   - Token-aware message handling that automatically tracks conversation size

3. **ChatBuilder Factory**:
   - Convenient factory methods for creating chats with specific providers
   - Handles provider creation and configuration with sensible defaults
   - Makes it easy to get started with common providers (Anthropic, Google)
   - Includes a mock builder for testing

4. **Stream-Lined User Interface**:
   - Simple methods for sending messages and getting responses
   - Support for different message types (user, function, tool)
   - Clean conversion between internal and external message formats
   - Well-designed response type that maintains important metadata

This redesign maintains compatibility with the existing provider implementations while providing a much more flexible and user-friendly interface. It better integrates the foundational components (TokenCounter, ChatHistoryCompactor) into the core functionality of the Chat client.

#### 2025-04-18: Model Types and Transport Separation

1. **Type-Safe Model Enums**:
   - Introduced `ModelInfo` trait to abstract common model functionality
   - Created type-safe enums for `AnthropicModel` and `GoogleModel`
   - Each enum variant represents a specific model with its properties
   - Maintained backwards compatibility with string-based model IDs
   - Improved type safety by preventing invalid model specifications

2. **Transport Visitor Pattern**:
   - Separated HTTP transport logic from provider-specific conversion logic
   - Created `TransportVisitor` trait as a base interface for transport implementations
   - Specialized with `AnthropicTransportVisitor` and `GoogleTransportVisitor` for provider-specific operations
   - Provider implementations accept visitors for actual API calls
   - Implemented both HTTP and mock transport implementations

3. **HTTP Transport Implementation**:
   - Implemented full HTTP transport using reqwest
   - Centralized error handling and response processing
   - Created proper header management
   - Reduced code duplication across providers

4. **Mock Transport for Testing**:
   - Implemented a fully functional mock transport
   - Allows testing provider code without actual API calls
   - Supports capturing requests for verification
   - Enables customizable responses for different test scenarios

5. **Key Benefits**:
   - **Improved Type Safety**: Models are now strongly typed
   - **Better Separation of Concerns**: Conversion logic separate from transport details
   - **Enhanced Testability**: Mock transport makes testing easier
   - **Reduced Code Duplication**: Common HTTP logic centralized
   - **More Flexible Architecture**: Providers can focus on format conversion

The introduction of the visitor pattern and type-safe model enums represents a significant architectural improvement. It maintains backward compatibility while paving the way for a cleaner, more maintainable, and more testable codebase.

#### 2025-04-18: Provider Model Management Refactoring

1. **Removed Available Models Lists**:
   - Eliminated the concept of a cached list of available models from providers
   - Removed the `list_models()` method from the `LlmProvider` trait
   - Replaced static model lists with direct model lookups using enums
   - Simplified provider implementations by removing model caching

2. **Direct Model Capability Checking**:
   - Modified `has_capability()` and `supports_tool_calling()` to use direct model mappings
   - Added pattern matching on model IDs to map them to appropriate `ModelInfo` implementations
   - Added fallback capability detection for unknown model IDs based on naming patterns
   - Enhanced error handling for unsupported models

3. **Benefits of the Refactoring**:
   - **Reduced Memory Usage**: No more caching of model lists
   - **Simplified Implementation**: Provider code is cleaner and more focused
   - **Direct Model Information Access**: More direct path to model capabilities via model enums
   - **Better Performance**: No need to build and search through model lists
   - **More Maintainable**: Easier to add new models without updating multiple locations

4. **Testing Improvements**:
   - Updated tests to work with the new model capability checking approach
   - Added comprehensive tests for both known and unknown model IDs
   - Verified capability detection for various model types
   - Ensured behavior is consistent with previous implementation

This refactoring simplifies the provider implementation while maintaining the same functionality. By leveraging the `ModelInfo` trait and model enums more directly, we've reduced code complexity and improved the architecture's clarity. The changes make the codebase more maintainable and set a foundation for future improvements in model management.

#### 2025-04-18: Provider Integration Testing

1. **Request Format Verification**:
   - Implemented integration tests to verify that providers correctly format requests according to their respective API specifications
   - Created tests that capture and analyze the exact JSON payloads sent to the API
   - Ensured that message format conversion follows each provider's requirements
   - Validated proper handling of system messages, user messages, and assistant messages
   - Confirmed correct implementation of content formatting, especially for text content

2. **Transport Visitor Pattern Testing**:
   - Used the mock transport to capture and verify request payloads
   - Tested the `prepare_*_request` methods to ensure they generate correctly formatted requests
   - Confirmed that model IDs, message content, and generation options are properly reflected in requests
   - Tested the visitor pattern implementation to ensure proper separation of concerns
   - Established a foundation for later testing of tool/function calling capabilities

3. **Provider-Specific Verification**:
   - For Anthropic: Verified proper message array format, content parts structure, and model specification
   - For Google: Verified contents array format, parts structure, and generation configuration
   - Ensured each provider correctly converts internal message format to provider-specific format
   - Tested edge cases like empty messages and system messages

4. **Foundation for Tool Testing**:
   - Created tests that can be extended to include tool and function calling
   - Established patterns for verifying tool specification in requests
   - Set up structure to verify tool response handling when implemented

These integration tests follow our core design principles:
- **Provider-agnostic API**: Tests verify that our common message format translates correctly to provider-specific formats
- **Type safety**: Leverage Rust's type system to catch format errors at compile time
- **Testability**: Mock transport enables verification without actual API calls
- **Error handling**: Tests verify proper error cases and response handling

The tests serve as executable documentation, showing exactly how our internal message format maps to each provider's expected request format. This establishes a foundation for implementing and testing more advanced features like tool calling while ensuring we don't break the basic message functionality.

#### 2025-04-18: Enhanced Anthropic Provider Implementation

1. **Message Type Conversions**:
   - Implemented bidirectional conversion between library Message types and Anthropic-specific types
   - Used From/Into trait implementations for clean, idiomatic conversion
   - Added support for multimodal content (text and images)
   - Created proper mappings for all message roles

2. **Request Payload Construction**:
   - Implemented creation of fully-formed Anthropic API request payloads
   - Properly handles system messages as separate parameter according to Anthropic's API requirements
   - Converts Chat settings like max_tokens to appropriate Anthropic parameters
   - Added support for temperature and sampling parameters
   - Structured foundation for tool support

3. **Response Handling**:
   - Created full conversion from Anthropic response format to library Message types
   - Properly captures token usage information as message metadata
   - Handles both text responses and tool use appropriately
   - Provides clean conversion for multipart responses

4. **Testing**:
   - Comprehensive unit tests for bidirectional conversion
   - Test coverage for message types, content parts, and response handling
   - Tests for complete request payload generation
   - Verification of proper system message handling

The implementation follows our core design principles:
- **Type safety**: Strong types for all Anthropic API format structures
- **Provider-agnostic API**: The provider seamlessly integrates with our generic abstraction
- **Extensibility**: Design patterns make it easy to add more capabilities
- **Testability**: Well-tested with high coverage

The From/Into trait pattern we've established provides a clean, idiomatic way to handle conversions between our library's format and provider-specific formats. This pattern can be applied to other providers like OpenAI and Google, ensuring consistent implementation across the library.

#### 2025-04-18: Model Parameters Update

1. **Context Window and Output Token Limits**:
   - Updated all model implementations with accurate context window sizes based on official documentation
   - Added proper max output token values for both Anthropic and Google models
   - Incorporated special case handling for Claude 3.7 Sonnet's extended thinking mode (128K tokens vs standard 64K)
   - Standardized the naming and structure of model enums

2. **Implementation Details**:
   - **Anthropic Models**:
     - Context Windows: All Claude 3 models have 200K token context windows, while older models have 100K
     - Output Tokens: Most Claude models have 4096 max output tokens
     - Special Case: Claude 3.7 Sonnet has 64K default output tokens, and 128K with extended thinking mode 
   - **Google Models**:
     - Context Windows: Gemini 1.5/2.0/2.5 models have massive 1M-2M token context windows
     - Gemini 1.0 models have smaller 32K token context windows
     - Output Tokens: Most Google models support 8192 output tokens
     - Exceptions: Gemini 1.0 Pro Vision (4096) and Gemini 2 Flash Lite (2048)

3. **Benefits of the Update**:
   - More accurate token usage predictions and constraints
   - Better handling of model-specific limitations
   - Improved user experience through appropriate context window management
   - Support for extended output capabilities where available

This update ensures our model representations accurately reflect the capabilities and constraints of each model according to the official documentation from Anthropic and Google. The implementation now properly handles edge cases like Claude 3.7's extended thinking mode and accurately represents the varying capabilities across different model versions.

Additionally, we've restructured the codebase to move client-related functionality to the provider module for better organization and to reflect the fact that providers are the main interaction point with LLM APIs. We've also:

1. Made ModelInfo extend the Model trait to ensure all models implement both the basic identification methods and the capability-specific methods
2. Added struct variants to model enums where needed, such as Sonnet35Version and the extended thinking flag for Sonnet37
3. Implemented a more flexible provider hierarchy with appropriate traits and implementations
4. Created stub implementations of mock providers to support testing

#### 2025-04-18: Typed Tool View Pattern Implementation

1. **Type-Safe Tool Call Handling**:
   - Implemented a "typed view" pattern for tool calls that provides type safety without making core types generic
   - Created a clear separation between untyped message storage and typed tool handling
   - Leveraged Rust's type system and pattern matching for safe tool call processing
   - Made tool integration fully optional and non-intrusive to the core API

2. **Implementation Details**:
   - **Core Untyped Toolbox**:
     - Simplified the `Toolbox` trait to be non-generic, working with untyped JSON values
     - Added `describe()` method to get tool descriptions for LLM prompt construction
     - Added `execute()` method for actual tool execution with string results
   
   - **Typed Layer**:
     - Created `TypedToolbox<T>` trait that extends the base `Toolbox` and adds type-safe methods
     - Added `parse_tool_call()` method to convert untyped ToolCall to typed representation
     - Added `execute_typed()` method for executing tools with strongly typed inputs
   
   - **View Pattern**:
     - Implemented `ToolCallView<T, TB>` struct that provides a typed view over messages
     - View takes a reference to both a message and a typed toolbox
     - Provides methods to extract typed tool calls and check for tool call presence
     - Leverages serde for type-safe JSON deserialization

3. **Chat Integration**:
   - Updated `Chat` struct to include an optional untyped `Toolbox`
   - Added methods for setting and managing toolboxes
   - Implemented `process_tool_calls()` to automatically handle tool execution
   - Added `tool_descriptions()` to get tool descriptions for LLM prompting

4. **Benefits of the Design**:
   - **Clean Separation of Concerns**: Core types remain simple and non-generic
   - **Type Safety Where Needed**: Tool implementations can be fully type-safe
   - **Optional Integration**: Tools are completely optional and non-intrusive
   - **Pattern Matching**: Enables exhaustive pattern matching on tool requests
   - **No Dynamic Type Checking**: Avoids runtime type errors through compile-time checking
   - **Extensibility**: Easy to add new tool types without changing core types

This design achieves the goal of providing type safety for tools without sacrificing the simplicity of the core API. It uses Rust's strong type system to ensure that tool inputs and outputs are handled correctly, while keeping the message storage layer simple and efficient. The view pattern provides a clean abstraction that bridges the gap between the untyped storage and the typed tool handling.

The implementation follows a principle of "pay for what you use" - users who don't need tools aren't burdened with the complexity, while those who do can opt into the type-safe layer. This balances flexibility, performance, and safety in an elegant way that leverages Rust's strengths.

## Future Directions

1. **Streaming**: Support for streaming responses from LLMs.
2. **Embeddings**: Unified API for generating and working with embeddings.
3. **Caching**: Built-in caching to reduce API calls.
4. **Rate Limiting**: Intelligent rate limiting to avoid provider quotas.
5. **Middleware**: Pluggable middleware for logging, metrics, etc.
6. **Local Models**: Support for running LLMs locally.
7. **Additional Provider Implementations**: Implementations for other major providers like OpenAI.
8. **Additional Tools**: Implement more tools like web search, weather, etc.
9. **Advanced History Compactors**: Implement more sophisticated history management strategies, such as summarization-based compaction.
10. **Proper Tokenization**: Replace the naive token counter with proper model-specific tokenizers.
11. **Full Transport Integration**: Complete the transport visitor pattern implementation for all providers.
12. **Comprehensive Integration Testing**: Expand integration tests to cover all features including tools, streaming, and vision capabilities.

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