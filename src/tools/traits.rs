//! # Tool Executor Traits
//!
//! Defines the core traits for tool execution.

use std::future::Future;
use std::pin::Pin;

use super::types::{SandboxConfig, ToolError, ToolOutput, ToolResult, ToolSpec};
use crate::providers::types::ToolCall;

// =============================================================================
// CORE TRAIT
// =============================================================================

/// Trait for tools that can be executed.
///
/// Implementations should be thread-safe and handle their own error cases.
/// Tools are executed asynchronously to support IO-bound operations.
///
/// # Example
///
/// ```rust,ignore
/// use torus_attention::tools::{Tool, ToolCall, ToolOutput, ToolResult, ToolSpec};
///
/// struct EchoTool;
///
/// impl Tool for EchoTool {
///     fn spec(&self) -> ToolSpec {
///         ToolSpec::new("echo", "Echo back the input")
///             .with_param(ParameterSchema::required_string("message", "Message to echo"))
///     }
///
///     fn execute<'a>(
///         &'a self,
///         call: &'a ToolCall,
///     ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>> {
///         Box::pin(async move {
///             let args: serde_json::Value = serde_json::from_str(&call.arguments)?;
///             let message = args["message"].as_str().unwrap_or("(empty)");
///             Ok(ToolOutput::success(message))
///         })
///     }
/// }
/// ```
pub trait Tool: Send + Sync {
    /// Get the tool specification
    fn spec(&self) -> ToolSpec;

    /// Execute the tool with the given call
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>>;

    /// Get the tool name (convenience method)
    fn name(&self) -> &str {
        // Note: This returns a reference but ToolSpec is owned
        // Implementations should override this if they need efficient access
        ""
    }

    /// Validate arguments before execution
    fn validate_args(&self, args: &serde_json::Value) -> ToolResult<()> {
        let spec = self.spec();

        for param in &spec.parameters {
            if param.required {
                if args.get(&param.name).is_none() {
                    return Err(ToolError::invalid_args(
                        &spec.name,
                        format!("Missing required parameter: {}", param.name),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Check if the tool is available (e.g., required binaries exist)
    fn is_available(&self) -> bool {
        true
    }
}

/// A sandboxed tool that respects security constraints
pub trait SandboxedTool: Tool {
    /// Get the sandbox configuration
    fn sandbox_config(&self) -> &SandboxConfig;

    /// Set the sandbox configuration
    fn set_sandbox_config(&mut self, config: SandboxConfig);

    /// Check if an operation is allowed by the sandbox
    fn check_sandbox(&self, operation: &str, target: &str) -> ToolResult<()>;
}

// =============================================================================
// TOOL EXECUTOR (for running tools from ToolCall)
// =============================================================================

/// Executor that can run tools from LLM tool calls
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool call
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>>;

    /// Check if a tool exists
    fn has_tool(&self, name: &str) -> bool;

    /// Get available tool definitions (for LLM)
    fn tool_definitions(&self) -> Vec<crate::providers::types::ToolDefinition>;

    /// Get tool specs
    fn tool_specs(&self) -> Vec<ToolSpec>;
}

// =============================================================================
// TOOL HOOKS (for observability/interception)
// =============================================================================

/// Hook that is called before/after tool execution
pub trait ToolHook: Send + Sync {
    /// Called before a tool is executed
    fn before_execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolResult<()>> + Send + 'a>> {
        let _ = call;
        Box::pin(async move { Ok(()) })
    }

    /// Called after a tool is executed
    fn after_execute<'a>(
        &'a self,
        call: &'a ToolCall,
        result: &'a ToolResult<ToolOutput>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        let _ = (call, result);
        Box::pin(async move {})
    }

    /// Check if execution should proceed (for confirmation)
    fn should_execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>> {
        let _ = call;
        Box::pin(async move { true })
    }
}

/// A no-op hook that does nothing
pub struct NoOpHook;

impl ToolHook for NoOpHook {}

/// A logging hook that logs tool executions
pub struct LoggingHook {
    /// Log level
    pub level: LogLevel,
}

/// Log levels for the logging hook
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl Default for LoggingHook {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
        }
    }
}

impl ToolHook for LoggingHook {
    fn before_execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolResult<()>> + Send + 'a>> {
        let level = self.level;
        Box::pin(async move {
            match level {
                LogLevel::Debug => eprintln!("[DEBUG] Executing tool: {} ({})", call.name, call.id),
                LogLevel::Info => eprintln!("[INFO] Tool: {}", call.name),
                _ => {}
            }
            Ok(())
        })
    }

    fn after_execute<'a>(
        &'a self,
        call: &'a ToolCall,
        result: &'a ToolResult<ToolOutput>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        let level = self.level;
        let call_name = call.name.clone();
        let result_info = match result {
            Ok(output) => Some((output.success, output.execution_time_ms)),
            Err(_) => None,
        };
        let err_msg = match result {
            Err(e) => Some(e.to_string()),
            Ok(_) => None,
        };

        Box::pin(async move {
            match (level, result_info, err_msg) {
                (LogLevel::Debug, Some((success, time)), _) => {
                    eprintln!(
                        "[DEBUG] Tool {} completed: success={}, time={}ms",
                        call_name, success, time
                    );
                }
                (LogLevel::Info, Some((success, time)), _) => {
                    eprintln!(
                        "[INFO] Tool {} {} ({}ms)",
                        call_name,
                        if success { "succeeded" } else { "failed" },
                        time
                    );
                }
                (_, _, Some(err)) => {
                    eprintln!("[ERROR] Tool {} error: {}", call_name, err);
                }
                _ => {}
            }
        })
    }
}

// =============================================================================
// CONFIRMATION HANDLER
// =============================================================================

/// Handler for tool execution confirmation
pub trait ConfirmationHandler: Send + Sync {
    /// Request confirmation for a tool execution
    fn confirm<'a>(
        &'a self,
        call: &'a ToolCall,
        reason: &'a str,
    ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>>;
}

/// Auto-confirm handler (for non-interactive use)
pub struct AutoConfirm {
    /// Whether to confirm all (true) or deny all (false)
    pub confirm_all: bool,
}

impl Default for AutoConfirm {
    fn default() -> Self {
        Self { confirm_all: true }
    }
}

impl ConfirmationHandler for AutoConfirm {
    fn confirm<'a>(
        &'a self,
        _call: &'a ToolCall,
        _reason: &'a str,
    ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>> {
        let result = self.confirm_all;
        Box::pin(async move { result })
    }
}

/// Deny all confirmations
pub struct DenyAll;

impl ConfirmationHandler for DenyAll {
    fn confirm<'a>(
        &'a self,
        _call: &'a ToolCall,
        _reason: &'a str,
    ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>> {
        Box::pin(async move { false })
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::types::ParameterSchema;

    struct MockTool {
        name: String,
    }

    impl Tool for MockTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(&self.name, "A mock tool")
                .with_param(ParameterSchema::required_string("input", "Input value"))
        }

        fn execute<'a>(
            &'a self,
            call: &'a ToolCall,
        ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>> {
            Box::pin(async move { Ok(ToolOutput::success(format!("Executed: {}", call.name))) })
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[tokio::test]
    async fn test_mock_tool_execute() {
        let tool = MockTool {
            name: "mock".to_string(),
        };
        let call = ToolCall {
            id: "test-1".to_string(),
            name: "mock".to_string(),
            arguments: r#"{"input": "hello"}"#.to_string(),
        };

        let result = tool.execute(&call).await;
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }

    #[tokio::test]
    async fn test_validate_args_missing_required() {
        let tool = MockTool {
            name: "mock".to_string(),
        };

        let args = serde_json::json!({});
        let result = tool.validate_args(&args);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_args_success() {
        let tool = MockTool {
            name: "mock".to_string(),
        };

        let args = serde_json::json!({"input": "hello"});
        let result = tool.validate_args(&args);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_logging_hook() {
        let hook = LoggingHook::default();
        let call = ToolCall {
            id: "test-1".to_string(),
            name: "test_tool".to_string(),
            arguments: "{}".to_string(),
        };

        // Should not error
        let result = hook.before_execute(&call).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_auto_confirm() {
        let handler = AutoConfirm::default();
        let call = ToolCall {
            id: "test-1".to_string(),
            name: "dangerous_tool".to_string(),
            arguments: "{}".to_string(),
        };

        assert!(handler.confirm(&call, "This is destructive").await);
    }

    #[tokio::test]
    async fn test_deny_all() {
        let handler = DenyAll;
        let call = ToolCall {
            id: "test-1".to_string(),
            name: "any_tool".to_string(),
            arguments: "{}".to_string(),
        };

        assert!(!handler.confirm(&call, "Any reason").await);
    }
}
