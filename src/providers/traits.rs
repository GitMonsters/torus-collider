//! Provider traits defining the interface for LLM providers.
//!
//! This module defines the core traits that all providers must implement.

use super::types::{
    ChatCompletion, CompletionOptions, Embedding, Message, ProviderResult, StreamChunk,
};
use std::future::Future;
use std::pin::Pin;

// =============================================================================
// CORE PROVIDER TRAIT
// =============================================================================

/// Core trait for LLM providers.
///
/// This trait defines the basic interface for chat completion.
/// All providers (local, Anthropic, OpenAI, etc.) implement this trait.
///
/// # Example
///
/// ```rust,ignore
/// use torus_attention::providers::{LLMProvider, Message, CompletionOptions};
///
/// async fn example(provider: &dyn LLMProvider) {
///     let messages = vec![Message::user("Hello!")];
///     let response = provider.complete(&messages, CompletionOptions::default()).await?;
///     println!("{}", response.content());
/// }
/// ```
pub trait LLMProvider: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &str;

    /// Get the model being used
    fn model(&self) -> &str;

    /// Check if the provider is available/configured
    fn is_available(&self) -> bool;

    /// Generate a completion for the given messages
    fn complete<'a>(
        &'a self,
        messages: &'a [Message],
        options: CompletionOptions,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ChatCompletion>> + Send + 'a>>;

    /// Count tokens for the given messages (approximate for remote providers)
    fn count_tokens(&self, messages: &[Message]) -> ProviderResult<usize> {
        // Default implementation: rough estimate
        let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
        Ok(total_chars / 4) // Rough estimate: 4 chars per token
    }

    /// Get the maximum context length for this model
    fn max_context_length(&self) -> usize {
        4096 // Conservative default
    }
}

// =============================================================================
// STREAMING PROVIDER TRAIT
// =============================================================================

/// Trait for providers that support streaming responses.
///
/// Not all providers support streaming, so this is a separate trait.
pub trait StreamingProvider: LLMProvider {
    /// Stream a completion for the given messages
    ///
    /// Returns a receiver that yields chunks as they arrive.
    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        options: CompletionOptions,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<StreamReceiver>> + Send + 'a>>;
}

/// Receiver for streaming chunks
pub struct StreamReceiver {
    receiver: tokio::sync::mpsc::Receiver<ProviderResult<StreamChunk>>,
}

impl StreamReceiver {
    /// Create a new stream receiver
    pub fn new(receiver: tokio::sync::mpsc::Receiver<ProviderResult<StreamChunk>>) -> Self {
        Self { receiver }
    }

    /// Receive the next chunk
    pub async fn recv(&mut self) -> Option<ProviderResult<StreamChunk>> {
        self.receiver.recv().await
    }

    /// Collect all chunks into a single string
    pub async fn collect_text(mut self) -> ProviderResult<String> {
        let mut text = String::new();
        while let Some(result) = self.recv().await {
            let chunk = result?;
            text.push_str(&chunk.delta);
            if chunk.is_final {
                break;
            }
        }
        Ok(text)
    }
}

// =============================================================================
// EMBEDDING PROVIDER TRAIT
// =============================================================================

/// Trait for providers that support embedding generation.
pub trait EmbeddingProvider: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &str;

    /// Get the embedding model being used
    fn model(&self) -> &str;

    /// Get the dimensionality of embeddings
    fn embedding_dim(&self) -> usize;

    /// Generate embeddings for the given texts
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
    ) -> Pin<Box<dyn Future<Output = ProviderResult<Vec<Embedding>>> + Send + 'a>>;

    /// Generate embedding for a single text
    fn embed_one<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<Embedding>> + Send + 'a>> {
        Box::pin(async move {
            let texts = vec![text.to_string()];
            let mut embeddings = self.embed(&texts).await?;
            embeddings
                .pop()
                .ok_or_else(|| super::types::ProviderError::Internal("No embedding returned".to_string()))
        })
    }
}

// =============================================================================
// PROVIDER FACTORY
// =============================================================================

/// Factory for creating providers from configuration
pub trait ProviderFactory {
    /// Create a provider from the given configuration
    fn create(config: &super::ProviderConfig) -> ProviderResult<Box<dyn LLMProvider>>;
}

// =============================================================================
// HELPER TYPES
// =============================================================================

/// A boxed provider for dynamic dispatch
pub type BoxedProvider = Box<dyn LLMProvider>;

/// A boxed streaming provider
pub type BoxedStreamingProvider = Box<dyn StreamingProvider>;

/// A boxed embedding provider
pub type BoxedEmbeddingProvider = Box<dyn EmbeddingProvider>;

// =============================================================================
// SAFETY-AWARE WRAPPER
// =============================================================================

use crate::safety::{ProposedAction, SafetyGuard, EthicsEnforcer};

/// A wrapper that validates completions against safety constraints.
///
/// This wraps any provider and validates responses against the Prime Directive.
pub struct SafeProvider<P: LLMProvider> {
    inner: P,
    guard: Box<dyn SafetyGuard>,
}

impl<P: LLMProvider> SafeProvider<P> {
    /// Create a new safe provider with default ethics enforcer
    pub fn new(provider: P) -> Self {
        Self {
            inner: provider,
            guard: Box::new(EthicsEnforcer::default()),
        }
    }

    /// Create with a custom safety guard
    pub fn with_guard(provider: P, guard: Box<dyn SafetyGuard>) -> Self {
        Self {
            inner: provider,
            guard,
        }
    }

    /// Get reference to inner provider
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Validate a proposed action
    fn validate_action(&self, description: &str) -> bool {
        let action = ProposedAction::new(description)
            .with_benefit_to_self(0.3)  // Completing request benefits self (learning)
            .with_benefit_to_other(0.7); // Providing answer benefits user
        
        self.guard.validate_action(&action).allowed
    }
}

impl<P: LLMProvider + 'static> LLMProvider for SafeProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    fn complete<'a>(
        &'a self,
        messages: &'a [Message],
        options: CompletionOptions,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ChatCompletion>> + Send + 'a>> {
        Box::pin(async move {
            // Validate the request
            let last_user_message = messages.iter().rev().find(|m| m.role == super::types::Role::User);
            if let Some(msg) = last_user_message {
                if !self.validate_action(&msg.content) {
                    return Err(super::types::ProviderError::ContentBlocked {
                        reason: "Request blocked by safety guard".to_string(),
                        categories: vec!["prime_directive".to_string()],
                    });
                }
            }

            // Forward to inner provider
            let response = self.inner.complete(messages, options).await?;

            // Validate the response
            if !self.validate_action(&response.message.content) {
                return Err(super::types::ProviderError::ContentBlocked {
                    reason: "Response blocked by safety guard".to_string(),
                    categories: vec!["prime_directive".to_string()],
                });
            }

            Ok(response)
        })
    }

    fn count_tokens(&self, messages: &[Message]) -> ProviderResult<usize> {
        self.inner.count_tokens(messages)
    }

    fn max_context_length(&self) -> usize {
        self.inner.max_context_length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock provider for testing
    struct MockProvider {
        name: String,
        model: String,
    }

    impl MockProvider {
        fn new() -> Self {
            Self {
                name: "mock".to_string(),
                model: "mock-model".to_string(),
            }
        }
    }

    impl LLMProvider for MockProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn model(&self) -> &str {
            &self.model
        }

        fn is_available(&self) -> bool {
            true
        }

        fn complete<'a>(
            &'a self,
            _messages: &'a [Message],
            _options: CompletionOptions,
        ) -> Pin<Box<dyn Future<Output = ProviderResult<ChatCompletion>> + Send + 'a>> {
            Box::pin(async move {
                Ok(ChatCompletion {
                    id: "test-123".to_string(),
                    model: self.model.clone(),
                    message: Message::assistant("Hello! I'm a mock response."),
                    finish_reason: super::super::types::FinishReason::Stop,
                    usage: super::super::types::Usage::new(10, 20),
                    metadata: std::collections::HashMap::new(),
                })
            })
        }
    }

    #[test]
    fn test_mock_provider() {
        let provider = MockProvider::new();
        assert_eq!(provider.name(), "mock");
        assert!(provider.is_available());
    }

    #[test]
    fn test_token_count_estimate() {
        let provider = MockProvider::new();
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello there!"),
        ];
        let count = provider.count_tokens(&messages).unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_safe_provider_creation() {
        let provider = MockProvider::new();
        let safe = SafeProvider::new(provider);
        assert_eq!(safe.name(), "mock");
    }
}
