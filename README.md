# Language Barrier

A Rust library that provides abstractions for Large Language Models (LLMs).

## Overview

Language Barrier simplifies working with multiple LLM providers by offering a unified, type-safe API. It allows you to easily switch between different providers like OpenAI, Anthropic, Google, and more without changing your application code.

## Features

- Provider-agnostic API for working with LLMs
- Support for chat completions, text generation, and multimodal content
- Type-safe message structures based on the OpenAI standard format
- Detailed error handling for common LLM API issues
- Async-first design for efficient network operations
- Comprehensive model and provider management

## Installation

Add the library to your Cargo.toml:

```toml
[dependencies]
language-barrier = "0.1.0"
```

## Quick Start

```rust
use language_barrier::{
    message::{Message, MessageRole},
    model::{Model, ModelFamily, ModelCapability},
    provider::Provider,
};

// Example API usage will go here

```

## Documentation

For more details, see the [API documentation](https://docs.rs/language-barrier).

## Design

For information about the design and architecture of the library, see the [DESIGN.md](DESIGN.md) document.

## License

This project is licensed under the MIT License - see the LICENSE file for details.