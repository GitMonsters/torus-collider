//! # LLM Providers - Multi-Provider Abstraction
//!
//! This module provides a unified interface for interacting with various LLM providers,
//! including local models (TorusLLM) and remote API providers (Anthropic, OpenAI, etc.).
//!
//! ## Architecture
//!
//! ```text
//!                           LLM PROVIDER ABSTRACTION
//!     ┌────────────────────────────────────────────────────────────────┐
//!     │                                                                │
//!     │                    ┌─────────────────────┐                     │
//!     │                    │    LLMProvider      │                     │
//!     │                    │       trait         │                     │
//!     │                    └──────────┬──────────┘                     │
//!     │                               │                                │
//!     │           ┌───────────────────┼───────────────────┐            │
//!     │           │                   │                   │            │
//!     │           ▼                   ▼                   ▼            │
//!     │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
//!     │  │  LocalProvider  │ │AnthropicProvider│ │ OpenAIProvider  │  │
//!     │  │   (TorusLLM)    │ │   (Claude API)  │ │   (GPT API)     │  │
//!     │  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
//!     │                                                                │
//!     │  Future providers:                                             │
//!     │  • BedrockProvider (AWS)                                       │
//!     │  • OllamaProvider (local)                                      │
//!     │  • TogetherProvider                                            │
//!     │                                                                │
//!     └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use torus_attention::providers::{LLMProvider, AnthropicProvider, Message, Role};
//!
//! // Create a provider
//! let provider = AnthropicProvider::new("your-api-key")?;
//!
//! // Build messages
//! let messages = vec![
//!     Message::system("You are a helpful assistant."),
//!     Message::user("Hello, how are you?"),
//! ];
//!
//! // Generate completion
//! let response = provider.complete(&messages, Default::default()).await?;
//! println!("{}", response.content);
//! ```
//!
//! ## Safety Integration
//!
//! Providers can optionally integrate with the safety module:
//!
//! ```rust,ignore
//! use torus_attention::providers::SafeProvider;
//! use torus_attention::safety::EthicsEnforcer;
//!
//! let provider = AnthropicProvider::new("key")?;
//! let safe_provider = SafeProvider::new(provider, EthicsEnforcer::default());
//!
//! // All completions will be validated against Prime Directive
//! let response = safe_provider.complete(&messages, opts).await?;
//! ```

// Sub-modules
pub mod types;
pub mod traits;
pub mod local;
pub mod anthropic;
pub mod openai;

// Re-exports
pub use types::{
    Message, Role, ChatCompletion, CompletionOptions, Usage,
    ProviderError, ProviderResult, StreamChunk,
};
pub use traits::{LLMProvider, StreamingProvider, EmbeddingProvider};
pub use local::LocalProvider;
pub use anthropic::AnthropicProvider;
pub use openai::OpenAIProvider;

/// Provider configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProviderConfig {
    /// Provider type
    pub provider_type: ProviderType,
    /// API key (for remote providers)
    pub api_key: Option<String>,
    /// Base URL override
    pub base_url: Option<String>,
    /// Model name/ID
    pub model: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries
    pub max_retries: u32,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider_type: ProviderType::Local,
            api_key: None,
            base_url: None,
            model: "torus-small".to_string(),
            timeout_secs: 120,
            max_retries: 3,
        }
    }
}

impl ProviderConfig {
    /// Create config for Anthropic Claude
    pub fn anthropic(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::Anthropic,
            api_key: Some(api_key.into()),
            base_url: Some("https://api.anthropic.com".to_string()),
            model: model.into(),
            timeout_secs: 120,
            max_retries: 3,
        }
    }

    /// Create config for OpenAI
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::OpenAI,
            api_key: Some(api_key.into()),
            base_url: Some("https://api.openai.com".to_string()),
            model: model.into(),
            timeout_secs: 120,
            max_retries: 3,
        }
    }

    /// Create config for local TorusLLM
    pub fn local(model: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::Local,
            api_key: None,
            base_url: None,
            model: model.into(),
            timeout_secs: 300,
            max_retries: 1,
        }
    }
}

/// Supported provider types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProviderType {
    /// Local TorusLLM model
    Local,
    /// Anthropic Claude API
    Anthropic,
    /// OpenAI GPT API
    OpenAI,
    /// AWS Bedrock
    Bedrock,
    /// Ollama (local)
    Ollama,
    /// Together AI
    Together,
    /// Custom/other
    Custom,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => write!(f, "local"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenAI => write!(f, "openai"),
            Self::Bedrock => write!(f, "bedrock"),
            Self::Ollama => write!(f, "ollama"),
            Self::Together => write!(f, "together"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_config_defaults() {
        let config = ProviderConfig::default();
        assert_eq!(config.provider_type, ProviderType::Local);
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_provider_config_anthropic() {
        let config = ProviderConfig::anthropic("test-key", "claude-3-opus");
        assert_eq!(config.provider_type, ProviderType::Anthropic);
        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.model, "claude-3-opus");
    }

    #[test]
    fn test_provider_type_display() {
        assert_eq!(format!("{}", ProviderType::Anthropic), "anthropic");
        assert_eq!(format!("{}", ProviderType::OpenAI), "openai");
        assert_eq!(format!("{}", ProviderType::Local), "local");
    }
}
