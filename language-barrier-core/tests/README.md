# Language Barrier Tests

This directory contains tests for the Language Barrier library, organized by functionality rather than by provider.

## Test Organization

Tests are organized based on functional areas:

1. **basic_chat_tests.rs** - Tests basic chat functionality across all providers
   - Request creation
   - Response parsing
   - Basic interactions

2. **weather_tool_tests.rs** - Tests the weather tool functionality
   - Tool request creation
   - Tool response handling
   - Multi-provider compatibility

3. **calculator_tool_tests.rs** - Tests the calculator tool functionality
   - Mathematical operations
   - Error handling (division by zero)
   - Response processing

4. **multi_turn_conversation_tests.rs** - Tests multi-turn conversations
   - Context retention
   - Tool result processing
   - Conversation history management

## Running Tests

Tests will use whatever API keys are available in the environment. Each test checks for the relevant environment variables and runs tests for providers that have credentials configured.

Required environment variables:
- `ANTHROPIC_API_KEY` - For Anthropic Claude models
- `OPENAI_API_KEY` - For OpenAI GPT models
- `GEMINI_API_KEY` - For Google Gemini models
- `MISTRAL_API_KEY` - For Mistral models

You can run all tests with:

```bash
cargo test
```

Or run a specific test suite with:

```bash
cargo test --test basic_chat_tests
```

Or a specific test within a suite:

```bash
cargo test --test calculator_tool_tests test_calculator_tool
```

Add `-- --nocapture` to see output during the test:

```bash
cargo test --test basic_chat_tests -- --nocapture
```