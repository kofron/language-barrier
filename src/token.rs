/// A simple token counter for tracking token usage in conversations
///
/// This is a very basic implementation that provides a minimal token counting
/// utility. In a real-world scenario, you would want to use a proper tokenizer
/// like `tiktoken-rs` for accurate token counting specific to the model being used.
///
/// # Examples
///
/// ```
/// use language_barrier::TokenCounter;
///
/// let mut counter = TokenCounter::default();
/// counter.observe("Hello, world!");
/// assert_eq!(counter.total(), 2); // "Hello," and "world!" as two tokens
///
/// counter.subtract("Hello,");
/// assert_eq!(counter.total(), 1); // "world!" as one token
///
/// assert!(counter.under_budget(10)); // 1 < 10
/// ```
#[derive(Default, Clone, Debug)]
pub struct TokenCounter {
    total: usize,
}

impl TokenCounter {
    /// Creates a new `TokenCounter` with zero tokens
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::TokenCounter;
    ///
    /// let counter = TokenCounter::new();
    /// assert_eq!(counter.total(), 0);
    /// ```
    pub fn new() -> Self {
        Self { total: 0 }
    }

    /// Counts the number of tokens in a string (naive implementation)
    ///
    /// This is a very simple whitespace-based tokenization. In a real-world
    /// implementation, you would want to use a proper tokenizer like `tiktoken-rs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::TokenCounter;
    ///
    /// assert_eq!(TokenCounter::count_tokens("Hello, world!"), 2);
    /// assert_eq!(TokenCounter::count_tokens("one two three four"), 4);
    /// ```
    pub fn count_tokens(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Adds the token count of the given text to the total
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::TokenCounter;
    ///
    /// let mut counter = TokenCounter::default();
    /// counter.observe("Hello, world!");
    /// assert_eq!(counter.total(), 2);
    /// ```
    pub fn observe(&mut self, text: &str) {
        self.total += Self::count_tokens(text);
    }

    /// Subtracts the token count of the given text from the total
    ///
    /// Will not go below zero (saturates at zero).
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::TokenCounter;
    ///
    /// let mut counter = TokenCounter::default();
    /// counter.observe("Hello, world!");
    /// counter.subtract("Hello,");
    /// assert_eq!(counter.total(), 1);
    ///
    /// // Won't go below zero
    /// counter.subtract("world! and more");
    /// assert_eq!(counter.total(), 0);
    /// ```
    pub fn subtract(&mut self, text: &str) {
        self.total = self.total.saturating_sub(Self::count_tokens(text));
    }

    /// Returns the current total token count
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::TokenCounter;
    ///
    /// let mut counter = TokenCounter::default();
    /// counter.observe("Hello, world!");
    /// assert_eq!(counter.total(), 2);
    /// ```
    pub fn total(&self) -> usize {
        self.total
    }

    /// Checks if the total token count is under the specified budget
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier::TokenCounter;
    ///
    /// let mut counter = TokenCounter::default();
    /// counter.observe("Hello, world!");
    /// assert!(counter.under_budget(5));
    /// assert!(!counter.under_budget(1));
    /// ```
    pub fn under_budget(&self, max: usize) -> bool {
        self.total <= max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_counter_new() {
        let counter = TokenCounter::new();
        assert_eq!(counter.total(), 0);
    }

    #[test]
    fn test_token_counting() {
        assert_eq!(TokenCounter::count_tokens("Hello world"), 2);
        assert_eq!(TokenCounter::count_tokens(""), 0);
        assert_eq!(TokenCounter::count_tokens("   "), 0);
        assert_eq!(TokenCounter::count_tokens("one two three"), 3);
    }

    #[test]
    fn test_observe_and_total() {
        let mut counter = TokenCounter::default();
        counter.observe("Hello world");
        assert_eq!(counter.total(), 2);

        counter.observe("another message");
        assert_eq!(counter.total(), 4); // 2 + 2 tokens
    }

    #[test]
    fn test_subtract() {
        let mut counter = TokenCounter::default();
        counter.observe("Hello world another message");
        assert_eq!(counter.total(), 4);

        counter.subtract("Hello world");
        assert_eq!(counter.total(), 2);

        // Test saturation (won't go below zero)
        counter.subtract("way more tokens than we have now");
        assert_eq!(counter.total(), 0);
    }

    #[test]
    fn test_under_budget() {
        let mut counter = TokenCounter::default();
        counter.observe("Hello world");
        
        assert!(counter.under_budget(2));
        assert!(counter.under_budget(3));
        assert!(!counter.under_budget(1));
    }
}