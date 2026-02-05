//! # Bash Tool
//!
//! Sandboxed shell command execution.

use std::future::Future;
use std::pin::Pin;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

use super::traits::{SandboxedTool, Tool};
use super::types::{
    ParameterSchema, SandboxConfig, ToolCategory, ToolError, ToolOutput, ToolResult, ToolSpec,
};
use crate::providers::types::ToolCall;

// =============================================================================
// BASH TOOL
// =============================================================================

/// A sandboxed bash/shell command executor
pub struct BashTool {
    /// Sandbox configuration
    sandbox: SandboxConfig,
    /// Working directory
    workdir: Option<String>,
    /// Shell to use (default: /bin/sh)
    shell: String,
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BashTool {
    /// Create a new bash tool with default sandbox
    pub fn new() -> Self {
        Self {
            sandbox: SandboxConfig::default(),
            workdir: None,
            shell: "/bin/sh".to_string(),
        }
    }

    /// Create with a specific sandbox configuration
    pub fn with_sandbox(sandbox: SandboxConfig) -> Self {
        Self {
            sandbox,
            workdir: None,
            shell: "/bin/sh".to_string(),
        }
    }

    /// Create a permissive bash tool (for trusted environments)
    pub fn permissive() -> Self {
        Self {
            sandbox: SandboxConfig::permissive(),
            workdir: None,
            shell: "/bin/sh".to_string(),
        }
    }

    /// Set working directory
    pub fn with_workdir(mut self, workdir: impl Into<String>) -> Self {
        self.workdir = Some(workdir.into());
        self
    }

    /// Set shell
    pub fn with_shell(mut self, shell: impl Into<String>) -> Self {
        self.shell = shell.into();
        self
    }

    /// Execute a raw command (internal, no sandbox check)
    async fn execute_raw(&self, command: &str, timeout_ms: u64) -> ToolResult<ToolOutput> {
        let start = Instant::now();
        let timeout_duration = Duration::from_millis(timeout_ms);

        // Build command
        let mut cmd = Command::new(&self.shell);
        cmd.arg("-c").arg(command);

        if let Some(ref workdir) = self.workdir {
            cmd.current_dir(workdir);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::null());

        // Spawn process
        let mut child = cmd
            .spawn()
            .map_err(|e| ToolError::execution_failed(format!("Failed to spawn shell: {}", e)))?;

        // Wait with timeout
        let result = timeout(timeout_duration, async {
            let status = child
                .wait()
                .await
                .map_err(|e| ToolError::execution_failed(format!("Failed to wait for process: {}", e)))?;

            // Read stdout
            let mut stdout_content = String::new();
            if let Some(mut stdout) = child.stdout.take() {
                stdout.read_to_string(&mut stdout_content).await.map_err(|e| {
                    ToolError::io_error(format!("Failed to read stdout: {}", e))
                })?;
            }

            // Read stderr
            let mut stderr_content = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                stderr.read_to_string(&mut stderr_content).await.map_err(|e| {
                    ToolError::io_error(format!("Failed to read stderr: {}", e))
                })?;
            }

            // Truncate if too large
            let max_size = self.sandbox.max_output_size;
            if stdout_content.len() > max_size {
                stdout_content.truncate(max_size);
                stdout_content.push_str("\n... (output truncated)");
            }
            if stderr_content.len() > max_size {
                stderr_content.truncate(max_size);
                stderr_content.push_str("\n... (output truncated)");
            }

            let exit_code = status.code().unwrap_or(-1);
            let execution_time = start.elapsed().as_millis() as u64;

            Ok(ToolOutput {
                success: status.success(),
                output: stdout_content,
                error: if stderr_content.is_empty() {
                    None
                } else {
                    Some(stderr_content)
                },
                exit_code: Some(exit_code),
                execution_time_ms: execution_time,
                metadata: std::collections::HashMap::new(),
            })
        })
        .await;

        match result {
            Ok(output) => output,
            Err(_) => {
                // Timeout - try to kill the process
                let _ = child.kill().await;
                Err(ToolError::timeout(timeout_ms))
            }
        }
    }

    /// Parse arguments from a tool call
    fn parse_args(&self, call: &ToolCall) -> ToolResult<BashArgs> {
        serde_json::from_str(&call.arguments)
            .map_err(|e| ToolError::invalid_args(&call.name, format!("JSON parse error: {}", e)))
    }

    /// Internal execute implementation
    async fn execute_impl(&self, call: &ToolCall) -> ToolResult<ToolOutput> {
        let args = self.parse_args(call)?;

        // Check if shell commands are allowed
        if !self.sandbox.allow_shell {
            return Err(ToolError::permission_denied("Shell commands are disabled"));
        }

        // Check if the command is allowed
        if !self.sandbox.is_command_allowed(&args.command) {
            return Err(ToolError::command_not_allowed(&args.command));
        }

        // Use provided timeout or default
        let timeout_ms = args.timeout.unwrap_or(self.sandbox.default_timeout_ms);

        // Execute with optional workdir override
        let mut tool = self.clone();
        if let Some(workdir) = args.workdir {
            // Check if workdir is allowed
            if !self.sandbox.is_path_allowed(&workdir) {
                return Err(ToolError::path_not_allowed(&workdir));
            }
            tool.workdir = Some(workdir);
        }

        tool.execute_raw(&args.command, timeout_ms).await
    }
}

/// Arguments for the bash tool
#[derive(Debug, serde::Deserialize)]
struct BashArgs {
    /// Command to execute
    command: String,
    /// Optional working directory
    workdir: Option<String>,
    /// Optional timeout in milliseconds
    timeout: Option<u64>,
    /// Optional description (for logging)
    #[serde(default)]
    #[allow(dead_code)]
    description: Option<String>,
}

impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("bash", "Execute a shell command")
            .with_param(ParameterSchema::required_string(
                "command",
                "The shell command to execute",
            ))
            .with_param(ParameterSchema::optional_string(
                "workdir",
                "Working directory for the command",
                None,
            ))
            .with_param(ParameterSchema::optional_number(
                "timeout",
                "Timeout in milliseconds (default: 30000)",
                Some(30000.0),
            ))
            .with_param(ParameterSchema::optional_string(
                "description",
                "Description of what the command does",
                None,
            ))
            .with_category(ToolCategory::Shell)
            .destructive()
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>> {
        Box::pin(async move { self.execute_impl(call).await })
    }

    fn name(&self) -> &str {
        "bash"
    }

    fn is_available(&self) -> bool {
        // Check if shell exists
        std::path::Path::new(&self.shell).exists()
    }
}

impl Clone for BashTool {
    fn clone(&self) -> Self {
        Self {
            sandbox: self.sandbox.clone(),
            workdir: self.workdir.clone(),
            shell: self.shell.clone(),
        }
    }
}

impl SandboxedTool for BashTool {
    fn sandbox_config(&self) -> &SandboxConfig {
        &self.sandbox
    }

    fn set_sandbox_config(&mut self, config: SandboxConfig) {
        self.sandbox = config;
    }

    fn check_sandbox(&self, operation: &str, target: &str) -> ToolResult<()> {
        match operation {
            "execute" => {
                if !self.sandbox.allow_shell {
                    return Err(ToolError::permission_denied("Shell execution disabled"));
                }
                if !self.sandbox.is_command_allowed(target) {
                    return Err(ToolError::command_not_allowed(target));
                }
            }
            "workdir" => {
                if !self.sandbox.is_path_allowed(target) {
                    return Err(ToolError::path_not_allowed(target));
                }
            }
            _ => {
                return Err(ToolError::permission_denied(format!(
                    "Unknown operation: {}",
                    operation
                )));
            }
        }
        Ok(())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_call(command: &str) -> ToolCall {
        ToolCall {
            id: "test-1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({
                "command": command
            })
            .to_string(),
        }
    }

    #[tokio::test]
    async fn test_bash_echo() {
        let tool = BashTool::permissive();
        let call = make_call("echo hello");

        let result = tool.execute(&call).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.success);
        assert!(output.output.contains("hello"));
    }

    #[tokio::test]
    async fn test_bash_with_stderr() {
        let tool = BashTool::permissive();
        let call = make_call("echo error >&2");

        let result = tool.execute(&call).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.error.is_some());
        assert!(output.error.unwrap().contains("error"));
    }

    #[tokio::test]
    async fn test_bash_exit_code() {
        let tool = BashTool::permissive();
        let call = make_call("exit 42");

        let result = tool.execute(&call).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.success);
        assert_eq!(output.exit_code, Some(42));
    }

    #[tokio::test]
    async fn test_bash_sandbox_blocks_command() {
        let tool = BashTool::new(); // Default sandbox
        let call = make_call("rm -rf /");

        let result = tool.execute(&call).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            ToolError::CommandNotAllowed { .. } => {}
            other => panic!("Expected CommandNotAllowed, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bash_sandbox_allows_safe_command() {
        let tool = BashTool::new(); // Default sandbox allows ls
        let call = make_call("ls");

        let result = tool.execute(&call).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_bash_timeout() {
        let tool = BashTool::permissive();
        let call = ToolCall {
            id: "test-1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({
                "command": "sleep 10",
                "timeout": 100  // 100ms timeout
            })
            .to_string(),
        };

        let result = tool.execute(&call).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            ToolError::Timeout { timeout_ms } => {
                assert_eq!(timeout_ms, 100);
            }
            other => panic!("Expected Timeout, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bash_spec() {
        let tool = BashTool::new();
        let spec = tool.spec();

        assert_eq!(spec.name, "bash");
        assert_eq!(spec.category, ToolCategory::Shell);
        assert!(spec.is_destructive);
    }

    #[test]
    fn test_bash_is_available() {
        let tool = BashTool::new();
        // /bin/sh should exist on most Unix systems
        assert!(tool.is_available());
    }

    #[tokio::test]
    async fn test_bash_workdir() {
        let tool = BashTool::permissive().with_workdir("/tmp");
        let call = make_call("pwd");

        let result = tool.execute(&call).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.output.contains("/tmp"));
    }

    #[test]
    fn test_sandbox_check() {
        let tool = BashTool::new();

        // Should allow safe commands
        assert!(tool.check_sandbox("execute", "ls -la").is_ok());
        assert!(tool.check_sandbox("execute", "grep pattern file").is_ok());

        // Should block dangerous commands
        assert!(tool.check_sandbox("execute", "rm file").is_err());
        assert!(tool.check_sandbox("execute", "sudo ls").is_err());
    }
}
