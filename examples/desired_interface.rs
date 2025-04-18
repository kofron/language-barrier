//! Language Barrier – A miniature LLM runtime in a single, well‑commented file
//! ===================================================================================
//! This file stitches together **every design element** we’ve discussed so far:
//!
//!  1. `Secret<T>` – a new‑type that redacts API keys and other credentials.
//!  2. `Role`/`Message` – minimal chat‑history primitives.
//!  3. `TokenCounter` – running token budget plus `Default` impl.
//!  4. `ChatHistoryCompactor` trait – pluggable trimming strategy (`DropOldest` default).
//!  5. `Tool` trait – strongly‑typed function‑calling helpers (example: `Weather`).
//!  6. Provider/Model stubs – show separation of concerns.
//!  7. `Chat` – a mutable state object with builder‑ish helpers & runtime mutators.
//!  8. `SingleTurnChatExecutor` – stub executor emitting `tracing` spans.
//!
//! The code is intentionally **self‑contained** yet feels like a mini‑novel: every
//! block introduces the next, so you can read top‑to‑bottom and understand how
//! the parts snap together.
//!
//! -----------------------------------------------------------------------------
//! Crate dependencies (add these to Cargo.toml if you copy‑paste):
//! -----------------------------------------------------------------------------
//! async-trait = "0.1"
//! schemars    = "0.8"
//! serde       = { version = "1.0", features = ["derive"] }
//! thiserror   = "1.0"
//! tracing     = "0.1"
//! tracing-subscriber = "0.3"
//! tokio       = { version = "1", features = ["rt-multi-thread", "macros"] }
//!
//! -----------------------------------------------------------------------------
//! 0.  Prelude & `use` statements
//! -----------------------------------------------------------------------------

#![allow(dead_code)]
#![allow(unused_imports)]

use async_trait::async_trait;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use std::fmt;
use tracing::{info_span, Instrument};

/// --------------------------------------------------------------------------
/// 1.  Secret<T> – keeps secrets out of logs & JSON
/// --------------------------------------------------------------------------
#[derive(Clone)]
pub struct Secret<T>(pub T);

impl<T> fmt::Debug for Secret<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}
impl<T> fmt::Display for Secret<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("••••••")
    }
}
//  Secrets should *not* be serialized by default.
impl<T> serde::Serialize for Secret<T> {
    fn serialize<S>(&self, _: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unreachable!("Secret should never be serialized")
    }
}

/// --------------------------------------------------------------------------
/// 2.  Chat primitives: Role & Message
/// --------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Clone)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// --------------------------------------------------------------------------
/// 3.  TokenCounter – dirt‑simple, `Default`, and embeddable
/// --------------------------------------------------------------------------
#[derive(Default, Clone)]
pub struct TokenCounter {
    total: usize,
}

impl TokenCounter {
    /// Replace this with a real tokenizer such as `tiktoken-rs`.
    fn count_tokens(text: &str) -> usize {
        text.split_whitespace().count()
    }

    pub fn observe(&mut self, text: &str) {
        self.total += Self::count_tokens(text);
    }

    pub fn subtract(&mut self, text: &str) {
        self.total = self.total.saturating_sub(Self::count_tokens(text));
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn under_budget(&self, max: usize) -> bool {
        self.total <= max
    }
}

/// --------------------------------------------------------------------------
/// 4.  ChatHistoryCompactor – strategy pattern for trimming history
/// --------------------------------------------------------------------------
pub trait ChatHistoryCompactor: Send + Sync + 'static {
    /// Mutate `history` and `counter` in‑place so token total ≤ `max_tokens`.
    fn compact(
        &mut self,
        history: &mut Vec<Message>,
        counter: &mut TokenCounter,
        max_tokens: usize,
    );
}

/// Default strategy: drop messages from the front until under budget.
#[derive(Default)]
pub struct DropOldestCompactor;

impl ChatHistoryCompactor for DropOldestCompactor {
    fn compact(
        &mut self,
        history: &mut Vec<Message>,
        counter: &mut TokenCounter,
        max_tokens: usize,
    ) {
        while !counter.under_budget(max_tokens) && !history.is_empty() {
            let m = history.remove(0);
            counter.subtract(&m.content);
        }
    }
}

/// --------------------------------------------------------------------------
/// 5.  Tool abstraction + example Weather tool
/// --------------------------------------------------------------------------
#[derive(Deserialize, JsonSchema)]
pub struct WeatherInput {
    location: String,
}

#[derive(Serialize)]
pub struct WeatherData {
    temperature: f32,
    precip_chance: f32,
}

#[derive(Debug, thiserror::Error)]
#[error("Weather error")]
pub struct WeatherError;

#[derive(Deserialize, Serialize)]
pub struct Weather {
    api_key: Secret<&'static str>,
}

#[async_trait]
pub trait Tool: Send + Sync + 'static {
    type Input: for<'de> Deserialize<'de> + JsonSchema + Send;
    type Output: Serialize + Send;
    type Error: std::error::Error + Send + Sync;

    fn name() -> &'static str;
    fn description() -> &'static str;

    async fn call(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
}

#[async_trait]
impl Tool for Weather {
    type Input = WeatherInput;
    type Output = WeatherData;
    type Error = WeatherError;

    fn name() -> &'static str {
        "weather"
    }

    fn description() -> &'static str {
        "Get the latest forecast for a city or region"
    }

    async fn call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        // Tracing span shows up *without* leaking secret.
        let span = info_span!("tool_call", tool = %Self::name(), loc = %input.location);
        async move {
            // TODO: call real API.
            Err(WeatherError)
        }
        .instrument(span)
        .await
    }
}

/// --------------------------------------------------------------------------
/// 6.  Provider & Model – deliberately thin; focus on *transport* vs *capability*
/// --------------------------------------------------------------------------
mod language_barrier {
    use super::Secret;

    pub mod google {
        use super::Secret;

        /// Provider holds endpoint & auth.
        #[derive(Clone)]
        pub struct Google {
            pub api_key: Secret<&'static str>,
        }

        /// Model family – intrinsic capabilities.
        #[derive(Clone, Debug)]
        pub enum Gemini25 {
            Pro,
            Flash,
        }
    }
}

use language_barrier::google::*;

/// --------------------------------------------------------------------------
/// 7.  The Chat state object
/// --------------------------------------------------------------------------
pub struct Chat<M, P> {
    // Immutable after construction
    model: M,
    provider: P,

    // Tunable knobs / state
    system_prompt: String,
    max_output_tokens: usize,
    tools: Vec<Box<dyn Tool>>,

    history: Vec<Message>,
    token_counter: TokenCounter,
    compactor: Box<dyn ChatHistoryCompactor>,
}

impl<M, P> Chat<M, P>
where
    M: Clone + Send + Sync + 'static,
    P: Clone + Send + Sync + 'static,
{
    /// Bare‑bones new() keeps compile‑time guarantee that we always have
    /// both a model and provider.
    pub fn new(model: M, provider: P) -> Self {
        Self {
            model,
            provider,
            system_prompt: String::new(),
            max_output_tokens: 2048,
            tools: Vec::new(),
            history: Vec::new(),
            token_counter: TokenCounter::default(),
            compactor: Box::<DropOldestCompactor>::default(),
        }
    }

    /* ---------- fluent init helpers ---------- */

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.set_system_prompt(prompt);
        self
    }

    pub fn with_max_output_tokens(mut self, n: usize) -> Self {
        self.max_output_tokens = n;
        self
    }

    pub fn with_tool<T: Tool>(mut self, tool: T) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        // Recompute token counter from scratch.
        for m in &history {
            self.token_counter.observe(&m.content);
        }
        self.history = history;
        self
    }

    pub fn with_compactor<C: ChatHistoryCompactor>(mut self, comp: C) -> Self {
        self.compactor = Box::new(comp);
        self
    }

    /* ---------- runtime mutators ---------- */

    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        let p = prompt.into();
        self.token_counter.observe(&p);
        self.system_prompt = p;
        self.trim_to_context_window();
    }

    pub fn set_max_output_tokens(&mut self, n: usize) {
        self.max_output_tokens = n;
    }

    pub fn push_tool<T: Tool>(&mut self, tool: T) {
        self.tools.push(Box::new(tool));
    }

    pub fn push_message(&mut self, msg: Message) {
        self.token_counter.observe(&msg.content);
        self.history.push(msg);
        self.trim_to_context_window();
    }

    pub fn set_compactor<C: ChatHistoryCompactor>(&mut self, comp: C) {
        self.compactor = Box::new(comp);
        self.trim_to_context_window();
    }

    /* ---------- helpers ---------- */

    fn trim_to_context_window(&mut self) {
        const MAX_TOKENS: usize = 32_768; // could be model‑specific
        self.compactor
            .compact(&mut self.history, &mut self.token_counter, MAX_TOKENS);
    }

    pub fn tokens_used(&self) -> usize {
        self.token_counter.total()
    }
}

/// --------------------------------------------------------------------------
/// 8.  Executor – single‑turn stub with tracing
/// --------------------------------------------------------------------------
pub struct SingleTurnChatExecutor;

impl SingleTurnChatExecutor {
    pub async fn run<M, P>(chat: &Chat<M, P>) -> String {
        let span = info_span!(
            "chat_run",
            messages = chat.history.len(),
            tokens   = chat.tokens_used()
        );

        async move {
            // 1. Build provider‑specific HTTP request (omitted)
            // 2. Mock/real transport call
            // 3. Stream response & handle tool calls (omitted)
            format!(
                "(stub) executed chat with {} tokens across {} messages",
                chat.tokens_used(),
                chat.history.len()
            )
        }
        .instrument(span)
        .await
    }
}

/// --------------------------------------------------------------------------
/// 9.  Demonstration entry‑point
/// --------------------------------------------------------------------------
#[tokio::main]
async fn main() {
    // ------------- tracing setup -------------
    tracing_subscriber::fmt::init();

    let provider = Google {
        api_key: Secret("google‑key‑123"),
    };
    let gemini = Gemini25::Pro;

    // Build chat with a custom compactor that keeps only the last 5 messages.
    struct KeepLastFive;
    impl ChatHistoryCompactor for KeepLastFive {
        fn compact(
            &mut self,
            hist: &mut Vec<Message>,
            counter: &mut TokenCounter,
            _max: usize,
        ) {
            while hist.len() > 5 {
                let m = hist.remove(0);
                counter.subtract(&m.content);
            }
        }
    }

    let mut chat = Chat::new(gemini, provider)
        .with_system_prompt("You are a helpful assistant")
        .with_tool(Weather {
            api_key: Secret("wx‑key"),
        })
        .with_compactor(KeepLastFive);

    // Prime history with a couple of user messages.
    chat.push_message(Message {
        role: Role::User,
        content: "Hi, what's the weather in Paris?".into(),
    });

    chat.push_message(Message {
        role: Role::User,
        content: "And how about Tokyo?".into(),
    });

    let resp1 = SingleTurnChatExecutor::run(&chat).await;
    println!("Response‑1: {resp1}");

    // Mutate chat and run again.
    chat.set_system_prompt("Respond like a pirate, arrr!");
    chat.set_max_output_tokens(1024);

    let resp2 = SingleTurnChatExecutor::run(&chat).await;
    println!("Response‑2: {resp2}");
}

/// --------------------------------------------------------------------------
/// 10.  Testing notes (non‑executing, but compile‑time docs)
/// --------------------------------------------------------------------------
/// - **Provider unit‑tests** call an internal `build_request(&Chat)` helper and
///   assert the JSON matches a golden fixture.
/// - **Executor tests** inject a `MockTransport` implementing a `Transport`
///   trait (not shown) so we can feed predetermined responses and assert on
///   streaming, tool recursion, and span metadata.
/// - **Tracing** can be captured with `tracing::collect::with_default` in tests
///   to validate we emit `chat_run` and `tool_call` spans with correct fields.
