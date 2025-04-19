use serde::{Serialize, Serializer};
use std::fmt;

/// A wrapper type for sensitive information like API keys
///
/// `Secret<T>` hides the inner value in debug output and display implementations
/// to prevent accidental leakage of sensitive information in logs or error messages.
///
/// # Examples
///
/// ```
/// use language_barrier_core::Secret;
///
/// let api_key = Secret("my-secret-api-key");
/// println!("API Key: {}", api_key); // Prints "API Key: ••••••"
/// ```
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

// Intentionally do not implement Serialize to prevent accidental serialization
impl<T> Serialize for Secret<T> {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unreachable!("Secret should never be serialized")
    }
}

// Additional implementations for convenient access to the inner value
impl<T> Secret<T> {
    /// Creates a new Secret wrapper around a value
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier_core::Secret;
    ///
    /// let api_key = Secret::new("my-secret-api-key");
    /// ```
    pub fn new(value: T) -> Self {
        Secret(value)
    }

    /// Gets a reference to the inner value
    ///
    /// This method should be used carefully to avoid leaking the secret value.
    /// It is primarily intended for internal use when the secret value needs
    /// to be used in an API call.
    ///
    /// # Examples
    ///
    /// ```
    /// use language_barrier_core::Secret;
    ///
    /// let api_key = Secret::new("my-secret-api-key");
    /// let inner_ref = api_key.inner();
    /// assert_eq!(inner_ref, &"my-secret-api-key");
    /// ```
    pub fn inner(&self) -> &T {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_debug() {
        let secret = Secret("api-key-123");
        assert_eq!(format!("{:?}", secret), "[REDACTED]");
    }

    #[test]
    fn test_secret_display() {
        let secret = Secret("api-key-123");
        assert_eq!(format!("{}", secret), "••••••");
    }

    #[test]
    fn test_secret_inner() {
        let secret = Secret("api-key-123");
        assert_eq!(secret.inner(), &"api-key-123");
    }
}
