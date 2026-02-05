//! # Agent Types
//!
//! Core types for the agent system including errors, events, and messages.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::providers::types::{ProviderError, Role, ToolCall, Usage};
use crate::safety::EthicsViolationType;
use crate::tools::types::ToolError;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur during agent execution
#[derive(Debug, Clone)]
pub enum AgentError {
    /// LLM provider error
    Provider(ProviderError),
    /// Tool execution error
    Tool(ToolError),
    /// Safety/ethics violation
    SafetyViolation {
        /// Type of violation
        violation_type: EthicsViolationType,
        /// Description
        message: String,
        /// Suggestions for resolution
        suggestions: Vec<String>,
    },
    /// Maximum iterations exceeded
    MaxIterationsExceeded {
        /// Number of iterations reached
        iterations: usize,
        /// Configured limit
        limit: usize,
    },
    /// Maximum tokens exceeded
    MaxTokensExceeded {
        /// Tokens used
        tokens: usize,
        /// Configured limit
        limit: usize,
    },
    /// Context window exceeded
    ContextOverflow {
        /// Current context size
        current: usize,
        /// Maximum allowed
        max: usize,
    },
    /// Invalid tool call from LLM
    InvalidToolCall {
        /// Tool name
        tool_name: String,
        /// Error message
        message: String,
    },
    /// Configuration error
    Configuration(String),
    /// Cancelled by user/hook
    Cancelled(String),
    /// Timeout
    Timeout {
        /// Operation that timed out
        operation: String,
        /// Duration
        duration: Duration,
    },
    /// Internal error
    Internal(String),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(e) => write!(f, "Provider error: {}", e),
            Self::Tool(e) => write!(f, "Tool error: {}", e),
            Self::SafetyViolation {
                violation_type,
                message,
                suggestions,
            } => {
                write!(
                    f,
                    "Safety violation [{:?}]: {}. Suggestions: {:?}",
                    violation_type, message, suggestions
                )
            }
            Self::MaxIterationsExceeded { iterations, limit } => {
                write!(
                    f,
                    "Max iterations exceeded: {} (limit: {})",
                    iterations, limit
                )
            }
            Self::MaxTokensExceeded { tokens, limit } => {
                write!(f, "Max tokens exceeded: {} (limit: {})", tokens, limit)
            }
            Self::ContextOverflow { current, max } => {
                write!(
                    f,
                    "Context window overflow: {} tokens (max: {})",
                    current, max
                )
            }
            Self::InvalidToolCall { tool_name, message } => {
                write!(f, "Invalid tool call '{}': {}", tool_name, message)
            }
            Self::Configuration(msg) => write!(f, "Configuration error: {}", msg),
            Self::Cancelled(reason) => write!(f, "Cancelled: {}", reason),
            Self::Timeout {
                operation,
                duration,
            } => {
                write!(f, "Timeout on '{}' after {:?}", operation, duration)
            }
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(e) => Some(e),
            Self::Tool(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ProviderError> for AgentError {
    fn from(err: ProviderError) -> Self {
        Self::Provider(err)
    }
}

impl From<ToolError> for AgentError {
    fn from(err: ToolError) -> Self {
        Self::Tool(err)
    }
}

/// Result type for agent operations
pub type AgentResult<T> = Result<T, AgentError>;

// =============================================================================
// EVENT TYPES (for observability)
// =============================================================================

/// Events emitted during agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent started processing
    Started {
        /// Unique run ID
        run_id: String,
        /// Input message
        input: String,
        /// Timestamp (millis since epoch)
        timestamp_ms: u64,
    },
    /// LLM completion received
    LLMResponse {
        /// Run ID
        run_id: String,
        /// Content (may be partial)
        content: String,
        /// Whether tool calls were made
        has_tool_calls: bool,
        /// Token usage
        usage: Usage,
        /// Iteration number
        iteration: usize,
    },
    /// Tool call started
    ToolCallStarted {
        /// Run ID
        run_id: String,
        /// Tool call details
        tool_call: ToolCallInfo,
        /// Iteration number
        iteration: usize,
    },
    /// Tool call completed
    ToolCallCompleted {
        /// Run ID
        run_id: String,
        /// Tool call ID
        tool_call_id: String,
        /// Whether successful
        success: bool,
        /// Output (truncated if long)
        output: String,
        /// Execution time
        duration_ms: u64,
    },
    /// Safety check performed
    SafetyCheck {
        /// Run ID
        run_id: String,
        /// Action checked
        action: String,
        /// Whether allowed
        allowed: bool,
        /// Reason if blocked
        reason: Option<String>,
    },
    /// Thinking/chain-of-thought
    Thinking {
        /// Run ID
        run_id: String,
        /// Thought content
        thought: String,
        /// Iteration number
        iteration: usize,
    },
    /// Agent completed
    Completed {
        /// Run ID
        run_id: String,
        /// Final response
        response: String,
        /// Total iterations
        iterations: usize,
        /// Total token usage
        total_usage: Usage,
        /// Total duration
        duration_ms: u64,
    },
    /// Agent errored
    Error {
        /// Run ID
        run_id: String,
        /// Error message
        error: String,
        /// Iteration where error occurred
        iteration: usize,
    },
}

/// Simplified tool call info for events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    /// Tool call ID
    pub id: String,
    /// Tool name
    pub name: String,
    /// Arguments (as JSON string)
    pub arguments: String,
}

impl From<&ToolCall> for ToolCallInfo {
    fn from(call: &ToolCall) -> Self {
        Self {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        }
    }
}

impl AgentEvent {
    /// Get the run ID for this event
    pub fn run_id(&self) -> &str {
        match self {
            Self::Started { run_id, .. } => run_id,
            Self::LLMResponse { run_id, .. } => run_id,
            Self::ToolCallStarted { run_id, .. } => run_id,
            Self::ToolCallCompleted { run_id, .. } => run_id,
            Self::SafetyCheck { run_id, .. } => run_id,
            Self::Thinking { run_id, .. } => run_id,
            Self::Completed { run_id, .. } => run_id,
            Self::Error { run_id, .. } => run_id,
        }
    }

    /// Check if this is a terminal event
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed { .. } | Self::Error { .. })
    }
}

// =============================================================================
// MESSAGE TYPES (for agentic loop)
// =============================================================================

/// An agent message (for internal tracking, extends provider Message)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// Message role
    pub role: Role,
    /// Content
    pub content: String,
    /// Tool calls (if assistant with tool calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID (if tool result)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Iteration when this message was created
    pub iteration: usize,
    /// Timestamp (millis since epoch)
    pub timestamp_ms: u64,
    /// Metadata
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub metadata: HashMap<String, String>,
}

impl AgentMessage {
    /// Create a new agent message
    pub fn new(role: Role, content: impl Into<String>, iteration: usize) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            iteration,
            timestamp_ms: Self::now_ms(),
            metadata: HashMap::new(),
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>, iteration: usize) -> Self {
        Self::new(Role::User, content, iteration)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>, iteration: usize) -> Self {
        Self::new(Role::Assistant, content, iteration)
    }

    /// Create an assistant message with tool calls
    pub fn assistant_with_tools(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
        iteration: usize,
    ) -> Self {
        Self {
            tool_calls: Some(tool_calls),
            ..Self::new(Role::Assistant, content, iteration)
        }
    }

    /// Create a tool result message
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        iteration: usize,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            iteration,
            timestamp_ms: Self::now_ms(),
            metadata: HashMap::new(),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content, 0)
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Convert to provider Message
    pub fn to_provider_message(&self) -> crate::providers::types::Message {
        let mut msg = crate::providers::types::Message::new(self.role, &self.content);
        msg.tool_calls = self.tool_calls.clone();
        msg.tool_call_id = self.tool_call_id.clone();
        msg.metadata = self.metadata.clone();
        msg
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

// =============================================================================
// RESPONSE TYPES
// =============================================================================

/// Final response from an agent run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    /// Run ID
    pub run_id: String,
    /// Final text response
    pub content: String,
    /// All messages in the conversation
    pub messages: Vec<AgentMessage>,
    /// Total iterations
    pub iterations: usize,
    /// Total token usage
    pub usage: Usage,
    /// Total duration
    pub duration_ms: u64,
    /// Tool calls made
    pub tool_calls_made: Vec<ToolCallRecord>,
    /// Whether safety checks were applied
    pub safety_applied: bool,
    /// Metadata
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentResponse {
    /// Create a new agent response
    pub fn new(
        run_id: String,
        content: String,
        messages: Vec<AgentMessage>,
        iterations: usize,
        usage: Usage,
        duration_ms: u64,
    ) -> Self {
        Self {
            run_id,
            content,
            messages,
            iterations,
            usage,
            duration_ms,
            tool_calls_made: Vec::new(),
            safety_applied: false,
            metadata: HashMap::new(),
        }
    }
}

/// Record of a tool call made during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Tool call ID
    pub id: String,
    /// Tool name
    pub name: String,
    /// Arguments
    pub arguments: String,
    /// Whether successful
    pub success: bool,
    /// Output (may be truncated)
    pub output: String,
    /// Execution time
    pub duration_ms: u64,
    /// Iteration when called
    pub iteration: usize,
}

impl ToolCallRecord {
    /// Create from a tool call and result
    pub fn new(
        call: &ToolCall,
        success: bool,
        output: String,
        duration_ms: u64,
        iteration: usize,
    ) -> Self {
        Self {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            success,
            output,
            duration_ms,
            iteration,
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_error_display() {
        let err = AgentError::MaxIterationsExceeded {
            iterations: 10,
            limit: 5,
        };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn test_agent_error_from_provider() {
        let provider_err = ProviderError::NetworkError("connection failed".to_string());
        let agent_err: AgentError = provider_err.into();
        assert!(matches!(agent_err, AgentError::Provider(_)));
    }

    #[test]
    fn test_agent_event_run_id() {
        let event = AgentEvent::Started {
            run_id: "run-123".to_string(),
            input: "hello".to_string(),
            timestamp_ms: 0,
        };
        assert_eq!(event.run_id(), "run-123");
    }

    #[test]
    fn test_agent_event_is_terminal() {
        let started = AgentEvent::Started {
            run_id: "run-123".to_string(),
            input: "hello".to_string(),
            timestamp_ms: 0,
        };
        assert!(!started.is_terminal());

        let completed = AgentEvent::Completed {
            run_id: "run-123".to_string(),
            response: "done".to_string(),
            iterations: 1,
            total_usage: Usage::default(),
            duration_ms: 100,
        };
        assert!(completed.is_terminal());
    }

    #[test]
    fn test_agent_message_user() {
        let msg = AgentMessage::user("hello", 0);
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "hello");
        assert_eq!(msg.iteration, 0);
    }

    #[test]
    fn test_agent_message_tool_result() {
        let msg = AgentMessage::tool_result("call-123", "result data", 1);
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id, Some("call-123".to_string()));
    }

    #[test]
    fn test_agent_message_to_provider() {
        let msg = AgentMessage::assistant("hello", 0);
        let provider_msg = msg.to_provider_message();
        assert_eq!(provider_msg.role, Role::Assistant);
        assert_eq!(provider_msg.content, "hello");
    }

    #[test]
    fn test_tool_call_info_from() {
        let call = ToolCall {
            id: "call-1".to_string(),
            name: "read".to_string(),
            arguments: r#"{"path": "/tmp"}"#.to_string(),
        };
        let info = ToolCallInfo::from(&call);
        assert_eq!(info.id, "call-1");
        assert_eq!(info.name, "read");
    }

    #[test]
    fn test_agent_response_new() {
        let response = AgentResponse::new(
            "run-1".to_string(),
            "done".to_string(),
            vec![],
            2,
            Usage::default(),
            100,
        );
        assert_eq!(response.run_id, "run-1");
        assert_eq!(response.iterations, 2);
    }

    #[test]
    fn test_tool_call_record_new() {
        let call = ToolCall {
            id: "call-1".to_string(),
            name: "bash".to_string(),
            arguments: "{}".to_string(),
        };
        let record = ToolCallRecord::new(&call, true, "output".to_string(), 50, 1);
        assert_eq!(record.name, "bash");
        assert!(record.success);
    }
}
