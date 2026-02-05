//! Core types for LLM providers.
//!
//! This module defines the common types used across all providers:
//! messages, completions, errors, and streaming types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur when using LLM providers
#[derive(Debug, Clone)]
pub enum ProviderError {
    /// API authentication failed
    AuthenticationError(String),
    /// Rate limit exceeded
    RateLimitError {
        retry_after_secs: Option<u64>,
        message: String,
    },
    /// Invalid request parameters
    InvalidRequest(String),
    /// Model not found or not accessible
    ModelNotFound(String),
    /// Network/connection error
    NetworkError(String),
    /// Response parsing error
    ParseError(String),
    /// Timeout
    Timeout(String),
    /// Content blocked by safety filters
    ContentBlocked {
        reason: String,
        categories: Vec<String>,
    },
    /// Token limit exceeded
    TokenLimitExceeded { limit: usize, requested: usize },
    /// Provider-specific error
    ProviderSpecific { code: String, message: String },
    /// Internal error
    Internal(String),
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AuthenticationError(msg) => write!(f, "Authentication error: {}", msg),
            Self::RateLimitError {
                message,
                retry_after_secs,
            } => {
                if let Some(secs) = retry_after_secs {
                    write!(
                        f,
                        "Rate limit exceeded: {} (retry after {}s)",
                        message, secs
                    )
                } else {
                    write!(f, "Rate limit exceeded: {}", message)
                }
            }
            Self::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            Self::ModelNotFound(model) => write!(f, "Model not found: {}", model),
            Self::NetworkError(msg) => write!(f, "Network error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::Timeout(msg) => write!(f, "Timeout: {}", msg),
            Self::ContentBlocked { reason, categories } => {
                write!(
                    f,
                    "Content blocked: {} (categories: {:?})",
                    reason, categories
                )
            }
            Self::TokenLimitExceeded { limit, requested } => {
                write!(
                    f,
                    "Token limit exceeded: requested {} but limit is {}",
                    requested, limit
                )
            }
            Self::ProviderSpecific { code, message } => {
                write!(f, "Provider error [{}]: {}", code, message)
            }
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for ProviderError {}

/// Result type for provider operations
pub type ProviderResult<T> = Result<T, ProviderError>;

// =============================================================================
// MESSAGE TYPES
// =============================================================================

/// Role of a message in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message (instructions)
    System,
    /// User message
    User,
    /// Assistant (AI) message
    Assistant,
    /// Tool/function result
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message author
    pub role: Role,
    /// Text content of the message
    pub content: String,
    /// Optional name (for multi-user scenarios)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool call ID (for tool responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls made in this message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Additional metadata
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub metadata: HashMap<String, String>,
}

impl Message {
    /// Create a new message
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Create a tool response message
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a name to the message
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A tool call made by the assistant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// Name of the tool/function
    pub name: String,
    /// Arguments as JSON string
    pub arguments: String,
}

// =============================================================================
// COMPLETION TYPES
// =============================================================================

/// Options for completion requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionOptions {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature (0.0 = deterministic, 1.0+ = more random)
    pub temperature: f64,
    /// Top-p (nucleus) sampling
    pub top_p: f64,
    /// Stop sequences
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop: Vec<String>,
    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: f64,
    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: f64,
    /// Whether to stream the response
    pub stream: bool,
    /// Available tools
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolDefinition>,
    /// User ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl Default for CompletionOptions {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 1.0,
            stop: Vec::new(),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stream: false,
            tools: Vec::new(),
            user: None,
        }
    }
}

impl CompletionOptions {
    /// Create options for deterministic output
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            ..Default::default()
        }
    }

    /// Create options for creative output
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        }
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Add stop sequences
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }

    /// Enable streaming
    pub fn streaming(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Add a tool
    pub fn with_tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }
}

/// Definition of a tool that can be called
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// JSON schema for the tool's parameters
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// A chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletion {
    /// Unique ID for this completion
    pub id: String,
    /// Model used
    pub model: String,
    /// Generated message
    pub message: Message,
    /// Reason for stopping
    pub finish_reason: FinishReason,
    /// Token usage
    pub usage: Usage,
    /// Provider-specific metadata
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ChatCompletion {
    /// Get the text content
    pub fn content(&self) -> &str {
        &self.message.content
    }

    /// Check if the response was truncated
    pub fn is_truncated(&self) -> bool {
        self.finish_reason == FinishReason::Length
    }

    /// Check if tool calls were made
    pub fn has_tool_calls(&self) -> bool {
        self.message.tool_calls.is_some()
    }

    /// Get tool calls if any
    pub fn tool_calls(&self) -> Option<&Vec<ToolCall>> {
        self.message.tool_calls.as_ref()
    }
}

/// Reason why generation stopped
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of response
    Stop,
    /// Hit max tokens limit
    Length,
    /// Hit a stop sequence
    StopSequence,
    /// Tool call was made
    ToolCalls,
    /// Content was filtered
    ContentFilter,
    /// Other/unknown reason
    Other,
}

/// Token usage statistics
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Input/prompt tokens
    pub prompt_tokens: usize,
    /// Output/completion tokens
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
    /// Cached tokens (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
}

impl Usage {
    /// Create usage from counts
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cached_tokens: None,
        }
    }
}

// =============================================================================
// STREAMING TYPES
// =============================================================================

/// A chunk from a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Chunk ID
    pub id: String,
    /// Delta content (new text)
    pub delta: String,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Finish reason (only on final chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// Usage (only on final chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// =============================================================================
// EMBEDDING TYPES
// =============================================================================

/// An embedding vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The input text that was embedded
    pub input: String,
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Model used
    pub model: String,
    /// Token count
    pub tokens: usize,
}

impl Embedding {
    /// Get the dimensionality of the embedding
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot: f32 = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * b)
            .sum();
        let mag_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            dot / (mag_a * mag_b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello!");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn test_message_builders() {
        let sys = Message::system("You are helpful");
        let user = Message::user("Hi").with_name("Alice");
        let tool = Message::tool("call_123", r#"{"result": 42}"#);

        assert_eq!(sys.role, Role::System);
        assert_eq!(user.name, Some("Alice".to_string()));
        assert_eq!(tool.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_completion_options_defaults() {
        let opts = CompletionOptions::default();
        assert_eq!(opts.max_tokens, 4096);
        assert!((opts.temperature - 0.7).abs() < 0.01);
        assert!(!opts.stream);
    }

    #[test]
    fn test_completion_options_builders() {
        let opts = CompletionOptions::deterministic()
            .with_max_tokens(100)
            .streaming();

        assert!((opts.temperature - 0.0).abs() < 0.01);
        assert_eq!(opts.max_tokens, 100);
        assert!(opts.stream);
    }

    #[test]
    fn test_usage() {
        let usage = Usage::new(100, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_embedding_cosine_similarity() {
        let emb1 = Embedding {
            input: "hello".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            model: "test".to_string(),
            tokens: 1,
        };
        let emb2 = Embedding {
            input: "world".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            model: "test".to_string(),
            tokens: 1,
        };
        let emb3 = Embedding {
            input: "orthogonal".to_string(),
            vector: vec![0.0, 1.0, 0.0],
            model: "test".to_string(),
            tokens: 1,
        };

        assert!((emb1.cosine_similarity(&emb2) - 1.0).abs() < 0.001);
        assert!((emb1.cosine_similarity(&emb3) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_provider_error_display() {
        let err = ProviderError::RateLimitError {
            retry_after_secs: Some(60),
            message: "Too many requests".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("Rate limit"));
        assert!(display.contains("60"));
    }
}
