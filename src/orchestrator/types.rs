//! Common types for the orchestrator module.
//!
//! Error types and result aliases used across orchestrator components.

use std::fmt;

/// Errors that can occur in the orchestrator.
#[derive(Debug, Clone)]
pub enum OrchestratorError {
    /// Memory operation failed
    MemoryError(String),

    /// Agent state transition error
    StateTransitionError {
        from: String,
        to: String,
        reason: String,
    },

    /// Provider communication failed
    ProviderError(String),

    /// Tool execution failed
    ToolError { tool_name: String, message: String },

    /// Safety validation failed
    SafetyViolation { action: String, reason: String },

    /// Configuration error
    ConfigError(String),

    /// Context limit exceeded
    ContextLimitExceeded { current: usize, limit: usize },

    /// Timeout during operation
    Timeout { operation: String, duration_ms: u64 },

    /// Session not found or expired
    SessionError(String),

    /// Internal orchestrator error
    Internal(String),
}

impl fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            Self::StateTransitionError { from, to, reason } => {
                write!(f, "Invalid state transition {} -> {}: {}", from, to, reason)
            }
            Self::ProviderError(msg) => write!(f, "Provider error: {}", msg),
            Self::ToolError { tool_name, message } => {
                write!(f, "Tool '{}' failed: {}", tool_name, message)
            }
            Self::SafetyViolation { action, reason } => {
                write!(f, "Safety violation for '{}': {}", action, reason)
            }
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            Self::ContextLimitExceeded { current, limit } => {
                write!(
                    f,
                    "Context limit exceeded: {} tokens (limit: {})",
                    current, limit
                )
            }
            Self::Timeout {
                operation,
                duration_ms,
            } => {
                write!(f, "Timeout during {}: {}ms", operation, duration_ms)
            }
            Self::SessionError(msg) => write!(f, "Session error: {}", msg),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for OrchestratorError {}

/// Result type for orchestrator operations.
pub type OrchestratorResult<T> = Result<T, OrchestratorError>;

/// Convert from provider errors
impl From<crate::providers::types::ProviderError> for OrchestratorError {
    fn from(err: crate::providers::types::ProviderError) -> Self {
        OrchestratorError::ProviderError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = OrchestratorError::StateTransitionError {
            from: "Idle".to_string(),
            to: "Acting".to_string(),
            reason: "Must think first".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("Invalid state transition"));
        assert!(display.contains("Idle"));
    }

    #[test]
    fn test_safety_violation_display() {
        let err = OrchestratorError::SafetyViolation {
            action: "harmful action".to_string(),
            reason: "Violates Prime Directive".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("Safety violation"));
        assert!(display.contains("Prime Directive"));
    }
}
