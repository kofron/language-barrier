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

#### 2025-04-18: Model Types and HTTP Provider Architecture

1. **Type-Safe Model Enums**:
   - Introduced `ModelInfo` trait to abstract common model functionality
   - Created type-safe enums for `Claude`, `Gemini`, and `GPT` models
   - Each enum variant represents a specific model with its properties
   - Maintained backwards compatibility with string-based model IDs
   - Improved type safety by preventing invalid model specifications

2. **Provider and Executor Pattern**:
   - Separated provider-specific conversion logic from HTTP transport
   - Created `HTTPProvider<M>` trait that converts between chat objects and HTTP requests
   - Provider implementations handle format-specific conversions
   - Executor handles actual HTTP communication

3. **HTTP Provider Implementation**:
   - Each provider implements the `HTTPProvider<M>` trait:
   ```rust
   pub trait HTTPProvider<M: ModelInfo> {
       fn accept(&self, chat: Chat<M>) -> Result<Request>;
       fn parse(&self, raw_response_text: String) -> Result<Message>;
   }
   ```
   - `accept()` method converts a chat into a provider-specific HTTP request
   - `parse()` method converts a provider-specific response back into our message format
   - Full separation of format conversion from HTTP transport

4. **SingleRequestExecutor**:
   - Handles the actual HTTP communication 
   - Works with any `HTTPProvider` implementation
   - Manages the reqwest HTTP client
   - Standardizes error handling and logging
   ```rust
   pub struct SingleRequestExecutor<M: ModelInfo> {
       provider: Box<dyn HTTPProvider<M>>,
       client: Client,
   }
   
   impl<M: ModelInfo> SingleRequestExecutor<M> {
       pub fn new(provider: impl HTTPProvider<M> + 'static) -> Self { /*...*/ }
       pub async fn send(&self, chat: Chat<M>) -> Result<Message> { /*...*/ }
   }
   ```

5. **Testing Support**:
   - Easy to create mock providers that implement `HTTPProvider`
   - Test-specific providers can simulate specific behaviors
   - No actual HTTP calls needed for most tests
   - Response formats can be directly tested

6. **Key Benefits**:
   - **Improved Type Safety**: Models are strongly typed and checked at compile time
   - **Better Separation of Concerns**: Providers focus on format conversion, executor handles HTTP
   - **Enhanced Testability**: Mock providers make testing easier and more reliable
   - **Reduced Code Duplication**: Common HTTP logic centralized in the executor
   - **More Flexible Architecture**: Easy to add new providers

#### 2025-04-18: Understanding the HTTPProvider Pattern in Depth

The `HTTPProvider` pattern is a core architectural component of Language Barrier. It enables a clean separation between provider-specific message formatting and the actual HTTP transport logic.

##### The HTTPProvider Trait

The trait is defined as:

```rust
pub trait HTTPProvider<M: ModelInfo> {
    fn accept(&self, chat: Chat<M>) -> Result<Request>;
    fn parse(&self, raw_response_text: String) -> Result<Message>;
}
```

This simple interface hides significant complexity:

1. **Generic Over Model Type**:
   - Each provider implementation is generic over a specific model type (e.g., `AnthropicProvider` implements `HTTPProvider<Claude>`)
   - This ensures type safety when matching providers with compatible models
   - Prevents accidental use of a provider with an incompatible model

2. **The `accept` Method**:
   - Takes a `Chat<M>` object representing a complete conversation
   - Converts it to a provider-specific HTTP request
   - Handles all the format conversion details:
     - Message format conversion
     - System prompt placement
     - Tool/function integration
     - Generation parameters
   - Returns a fully-formed `reqwest::Request` ready to be sent

3. **The `parse` Method**:
   - Takes a raw response string from the HTTP response
   - Parses it according to the provider's specific response format
   - Extracts the model's response, tool calls, token usage, etc.
   - Converts to our unified `Message` format
   - Handles error conditions in the response

##### Provider Implementations

Each provider implements this trait for its specific model type:

```rust
impl HTTPProvider<Claude> for AnthropicProvider {
    fn accept(&self, chat: Chat<Claude>) -> Result<Request> {
        // Convert to Anthropic's format
        // Set Anthropic-specific headers
        // Create and return the request
    }
    
    fn parse(&self, raw_response_text: String) -> Result<Message> {
        // Parse Anthropic's response format
        // Convert to our Message type
        // Return the parsed message
    }
}
```

This pattern allows each provider to handle its unique requirements:

- **Anthropic**: System prompts are a separate field, tool calls use a specific format
- **Google**: Uses "model" instead of "assistant" for role names
- **OpenAI**: Has a different structure for function calling

##### The Executor's Role

The `SingleRequestExecutor` is responsible for:

1. **HTTP Transport**: Managing the reqwest client and sending requests
2. **Error Handling**: Standardizing error handling across providers
3. **Logging**: Consistent logging of requests and responses
4. **Response Management**: Converting HTTP responses back to domain objects

```rust
impl<M: ModelInfo> SingleRequestExecutor<M> {
    pub async fn send(&self, chat: Chat<M>) -> Result<Message> {
        // Convert chat to request using provider.accept()
        // Send the HTTP request
        // Get the response
        // Parse with provider.parse()
        // Return the parsed message
    }
}
```

##### Benefits of this Architecture

1. **Separation of Concerns**:
   - Providers focus only on format conversion
   - Executor handles HTTP communication
   - Chat manages conversation state

2. **Type Safety**:
   - Compile-time checks ensure providers are used with compatible models
   - Generic bounds prevent misuse

3. **Testability**:
   - Easy to test format conversion in isolation
   - Can mock responses without actual HTTP calls
   - Integration tests can verify end-to-end behavior

4. **Extensibility**:
   - New providers can be added by implementing the trait
   - Existing code remains unchanged
   - Common HTTP logic is reused

This architecture provides a solid foundation for the library, enabling clean abstractions and a consistent interface while handling the complexity of different provider formats behind the scenes.

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

#### 2025-04-18: Mistral Provider Implementation

1. **Mistral API Integration**:
   - Implemented the Mistral provider that connects to Mistral's API
   - Supports a range of models (Mistral Large, Small, Nemo, Codestral, and Embed)
   - Handles authentication via API key with Bearer token
   - Follows Mistral's chat completions API format
   - Maintains compatibility with the existing HTTPProvider architecture

2. **Message Format Conversion**:
   - Created bidirectional conversion between our message format and Mistral's format
   - System messages are included in the messages array, similar to OpenAI's approach
   - Text content is properly formatted according to Mistral's requirements
   - Tool calls and responses are mapped to Mistral's function calling format

3. **Model Management**:
   - Added all available Mistral models: Large, Small, Nemo, Codestral, and Embed
   - Configured correct context window sizes (131k for most models, 256k for Codestral, 8k for Embed)
   - Set appropriate maximum output tokens (4096 tokens across all models)
   - Implemented the MistralModelInfo trait for model ID mapping

4. **Testing and Validation**:
   - Comprehensive unit tests for message format conversion
   - Integration tests for request creation and payload format
   - Tool integration tests to verify proper tool specification formatting
   - Live API tests (conditional on API key availability) to verify end-to-end functionality

This implementation follows our core design principles:
- **Provider-agnostic API**: The Mistral provider seamlessly integrates with our existing abstractions
- **Type safety**: Strong types for all Mistral-specific structures and model enums
- **Async-first**: All operations are async for efficient network I/O
- **Extensibility**: Easy to extend with additional Mistral-specific features
- **Testability**: Comprehensive test coverage for both unit and integration aspects
- **Error handling**: Robust error handling for API responses and formatting issues

Key implementation notes:
- Mistral's API is very similar to OpenAI's in structure, using the chat/completions endpoint
- Mistral uses "system", "user", "assistant", "function", and "tool" roles like our internal format
- Context window sizes vary by model, with the newest models supporting larger contexts
- Tool/function calling support follows a similar pattern to OpenAI's implementation

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

#### 2025-04-18: Tool System Design: A Deep Dive

The tool system is one of the more sophisticated parts of the Language Barrier library. It's designed to provide type safety while maintaining flexibility and avoiding excessive generics throughout the codebase. This section offers a detailed explanation of the tool system architecture.

##### Key Components

1. **Tool Trait**:
   ```rust
   pub trait Tool where Self: JsonSchema {
       fn name(&self) -> &str;
       fn description(&self) -> &str;
   }
   ```
   - Implemented by concrete tool request types
   - Requires `JsonSchema` for automatic schema generation
   - Provides name and description for LLM prompting
   - Tools are *not* responsible for execution logic

2. **Toolbox Trait**:
   ```rust
   pub trait Toolbox {
       fn describe(&self) -> Vec<ToolDescription>;
       fn execute(&self, name: &str, arguments: Value) -> Result<String>;
   }
   ```
   - Non-generic, works with untyped JSON values
   - Used for core message handling infrastructure
   - Designed for easy serialization to LLM-specific formats
   - All providers work with this trait directly

3. **TypedToolbox Trait**:
   ```rust
   pub trait TypedToolbox<T: DeserializeOwned>: Toolbox {
       fn parse_tool_call(&self, tool_call: &ToolCall) -> Result<T>;
       fn execute_typed(&self, request: T) -> Result<String>;
   }
   ```
   - Generic extension of `Toolbox` for type safety
   - Type parameter `T` represents the union of all tool types (often an enum)
   - Parses untyped tool calls into strongly typed representations
   - Enables pattern matching on tool request types

4. **ToolCallView**:
   ```rust
   pub struct ToolCallView<'a, T, TB>
   where
       T: DeserializeOwned,
       TB: TypedToolbox<T>,
   {
       toolbox: &'a TB,
       message: &'a Message,
       _phantom: PhantomData<T>,
   }
   ```
   - Provides a typed view over an untyped message
   - Non-invasive - doesn't change core message structure
   - Extracts strongly typed tool calls from messages
   - Enables processing with pattern matching

##### Implementation Pattern

The typical implementation pattern follows these steps:

1. **Define Tool Request Types**:
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
   struct WeatherRequest {
       location: String,
       units: Option<String>,
   }
   
   impl Tool for WeatherRequest {
       fn name(&self) -> &str { "get_weather" }
       fn description(&self) -> &str { "Get weather for a location" }
   }
   ```

2. **Define a Union Type** (often an enum):
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   enum MyToolRequest {
       Weather(WeatherRequest),
       Calculator(CalculatorRequest),
   }
   ```

3. **Implement Toolbox**:
   ```rust
   impl Toolbox for MyToolbox {
       fn describe(&self) -> Vec<ToolDescription> {
           // Generate descriptions for all tools
       }
       
       fn execute(&self, name: &str, arguments: Value) -> Result<String> {
           // Deserialize and execute specific tools
       }
   }
   ```

4. **Implement TypedToolbox**:
   ```rust
   impl TypedToolbox<MyToolRequest> for MyToolbox {
       fn parse_tool_call(&self, tool_call: &ToolCall) -> Result<MyToolRequest> {
           // Convert untyped calls to typed enum variants
       }
       
       fn execute_typed(&self, request: MyToolRequest) -> Result<String> {
           // Handle each enum variant with pattern matching
       }
   }
   ```

5. **Use ToolCallView for Processing**:
   ```rust
   let view = ToolCallView::<MyToolRequest, _>::new(&toolbox, &message);
   if view.has_tool_calls() {
       let tool_requests = view.tool_calls()?;
       for request in tool_requests {
           match request {
               MyToolRequest::Weather(req) => { /* handle weather request */ },
               MyToolRequest::Calculator(req) => { /* handle calculator request */ },
           }
       }
   }
   ```

##### Architecture Benefits

This architecture provides several key benefits:

1. **Type Safety Without Pervasive Generics**:
   - Core message types remain non-generic
   - Type safety is achieved where needed without affecting other parts of the codebase
   - Avoids complex generic bounds throughout the library

2. **Separation of Concerns**:
   - Tool definitions are separate from execution logic
   - Untyped layer handles serialization/deserialization
   - Typed layer handles type safety and pattern matching

3. **Provider Independence**:
   - Provider implementations work with the untyped `Toolbox` trait
   - No provider-specific tool handling needed
   - Automatic conversion to each provider's tool format

4. **Extensibility**:
   - Easy to add new tools by implementing the `Tool` trait
   - Existing toolboxes can be extended with new tools
   - Union types can be expanded without changing existing code

5. **Compatibility with LLM APIs**:
   - Tool descriptions are structured to match LLM API expectations
   - Automatic schema generation using `schemars::JsonSchema`
   - Consistent handling across all supported providers

##### Integration with Chat

The Chat struct integrates tooling through several key methods:

```rust
impl<M> Chat<M> where M: ModelInfo {
    // Adding a toolbox at construction time
    pub fn with_toolbox<T: Toolbox + 'static>(mut self, toolbox: T) -> Self { /*...*/ }
    
    // Setting a toolbox at runtime
    pub fn set_toolbox<T: Toolbox + 'static>(&mut self, toolbox: T) { /*...*/ }
    
    // Getting tool descriptions
    pub fn tool_descriptions(&self) -> Vec<ToolDescription> { /*...*/ }
    
    // Processing tool calls from an assistant message
    pub fn process_tool_calls(&mut self, assistant_message: &Message) -> Result<()> { /*...*/ }
}
```

When a message is processed by the LLM and contains tool calls, the `process_tool_calls` method executes those tools and adds the results as new `tool` role messages to the conversation history.

##### Development and Testing Considerations

When using the tool system, consider these best practices:

1. Keep tool schemas simple and use primitive types where possible
2. Use enums to group related tools together
3. Implement both `Toolbox` and `TypedToolbox` for maximum flexibility
4. Use the view pattern for processing tool responses
5. Keep tool execution logic separate from tool definitions

This design creates a type-safe system that balances flexibility, safety, and simplicity, allowing developers to define strongly-typed tools without complicating the rest of the codebase with generics.

#### 2025-04-20: Enhanced Tool System Design

1. **ToolDefinition Trait**:
   - Implemented the `ToolDefinition` trait with associated `Input` and `Output` types
   - Provides stronger type safety through generics with associated types
   - Separates tool definitions from execution logic
   - Includes schema generation for the input type
   - Allows for a more flexible and extensible tool system

2. **ToolRegistry**:
   - Created a new `ToolRegistry` for managing tools
   - Supports registration of tools with the `ToolDefinition` trait
   - Generates LLM-friendly tool descriptions
   - Provides foundation for the future execution system in runtime crate
   - Supports querying for tools by name

3. **Compatibility Layer**:
   - Implemented `ToolRegistryAdapter` for backward compatibility
   - Allows using the new tool definitions with the existing execution system
   - Enables gradual migration to the new tool system
   - Preserves all existing functionality

4. **LlmToolInfo**:
   - Standardized structure for LLM-facing tool descriptions
   - Used by tool registry to represent tools to LLMs
   - Simplifies integration with different LLM providers

5. **Enhanced Error Handling**:
   - Created specialized `ToolError` type
   - More granular error categories (schema errors, argument parsing, execution errors)
   - Better diagnostics for debugging tool issues
   - Type safety errors for output mismatches

6. **Separation of Concerns**:
   - Clear separation between tool definitions and execution logic
   - Registry focuses on storing and describing tools
   - Execution logic will be moved to runtime crate in future

The new tool system design represents a significant improvement in type safety and flexibility, while maintaining backward compatibility with the existing system. The changes align with our overall architecture that separates core definitions from runtime execution. This prepares us for a future middleware-based execution model while preserving all current functionality.

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
13. **Tool Runtime Implementation**: Implement the execution system for tools in the runtime crate.

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