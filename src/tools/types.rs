//! # Tool Types
//!
//! Core types for the tool execution system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur during tool execution
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum ToolError {
    /// Tool not found in registry
    #[error("Tool not found: {name}")]
    NotFound { name: String },

    /// Invalid arguments provided
    #[error("Invalid arguments for tool '{tool}': {reason}")]
    InvalidArguments { tool: String, reason: String },

    /// Tool execution failed
    #[error("Tool execution failed: {message}")]
    ExecutionFailed { message: String },

    /// Tool execution timed out
    #[error("Tool execution timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Permission denied
    #[error("Permission denied: {reason}")]
    PermissionDenied { reason: String },

    /// Sandboxing error
    #[error("Sandbox error: {message}")]
    SandboxError { message: String },

    /// Path not allowed
    #[error("Path not allowed: {path}")]
    PathNotAllowed { path: String },

    /// Command not allowed
    #[error("Command not allowed: {command}")]
    CommandNotAllowed { command: String },

    /// Network error
    #[error("Network error: {message}")]
    NetworkError { message: String },

    /// IO error
    #[error("IO error: {message}")]
    IoError { message: String },

    /// Serialization/deserialization error
    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    /// Tool was cancelled
    #[error("Tool execution cancelled")]
    Cancelled,
}

impl ToolError {
    pub fn not_found(name: impl Into<String>) -> Self {
        Self::NotFound { name: name.into() }
    }

    pub fn invalid_args(tool: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidArguments {
            tool: tool.into(),
            reason: reason.into(),
        }
    }

    pub fn execution_failed(message: impl Into<String>) -> Self {
        Self::ExecutionFailed {
            message: message.into(),
        }
    }

    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }

    pub fn permission_denied(reason: impl Into<String>) -> Self {
        Self::PermissionDenied {
            reason: reason.into(),
        }
    }

    pub fn path_not_allowed(path: impl Into<String>) -> Self {
        Self::PathNotAllowed { path: path.into() }
    }

    pub fn command_not_allowed(command: impl Into<String>) -> Self {
        Self::CommandNotAllowed {
            command: command.into(),
        }
    }

    pub fn io_error(message: impl Into<String>) -> Self {
        Self::IoError {
            message: message.into(),
        }
    }
}

/// Result type for tool operations
pub type ToolResult<T> = Result<T, ToolError>;

// =============================================================================
// TOOL EXECUTION RESULT
// =============================================================================

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Whether the tool succeeded
    pub success: bool,
    /// Output content (stdout for bash, file content for read, etc.)
    pub output: String,
    /// Error output if any (stderr for bash)
    pub error: Option<String>,
    /// Exit code if applicable
    pub exit_code: Option<i32>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Additional metadata
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ToolOutput {
    /// Create a successful output
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
            exit_code: Some(0),
            execution_time_ms: 0,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed output
    pub fn failure(output: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: output.into(),
            error: Some(error.into()),
            exit_code: Some(1),
            execution_time_ms: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set execution time
    pub fn with_execution_time(mut self, ms: u64) -> Self {
        self.execution_time_ms = ms;
        self
    }

    /// Set exit code
    pub fn with_exit_code(mut self, code: i32) -> Self {
        self.exit_code = Some(code);
        self.success = code == 0;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl std::fmt::Display for ToolOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.success {
            write!(f, "{}", self.output)
        } else {
            write!(
                f,
                "Error: {}\nOutput: {}",
                self.error.as_deref().unwrap_or("unknown"),
                self.output
            )
        }
    }
}

// =============================================================================
// TOOL DEFINITION (extended from providers::types)
// =============================================================================

/// Parameter schema for a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    /// Parameter name
    pub name: String,
    /// Parameter type (string, number, boolean, object, array)
    #[serde(rename = "type")]
    pub param_type: String,
    /// Description
    pub description: String,
    /// Whether the parameter is required
    pub required: bool,
    /// Default value if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    /// Enum values if this is an enum
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

impl ParameterSchema {
    /// Create a required string parameter
    pub fn required_string(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: "string".to_string(),
            description: description.into(),
            required: true,
            default: None,
            enum_values: None,
        }
    }

    /// Create an optional string parameter
    pub fn optional_string(
        name: impl Into<String>,
        description: impl Into<String>,
        default: Option<&str>,
    ) -> Self {
        Self {
            name: name.into(),
            param_type: "string".to_string(),
            description: description.into(),
            required: false,
            default: default.map(|s| serde_json::Value::String(s.to_string())),
            enum_values: None,
        }
    }

    /// Create a required number parameter
    pub fn required_number(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: "number".to_string(),
            description: description.into(),
            required: true,
            default: None,
            enum_values: None,
        }
    }

    /// Create an optional number parameter
    pub fn optional_number(
        name: impl Into<String>,
        description: impl Into<String>,
        default: Option<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            param_type: "number".to_string(),
            description: description.into(),
            required: false,
            default: default.map(|n| serde_json::json!(n)),
            enum_values: None,
        }
    }

    /// Create a boolean parameter
    pub fn boolean(name: impl Into<String>, description: impl Into<String>, default: bool) -> Self {
        Self {
            name: name.into(),
            param_type: "boolean".to_string(),
            description: description.into(),
            required: false,
            default: Some(serde_json::Value::Bool(default)),
            enum_values: None,
        }
    }

    /// Create an enum parameter
    pub fn enum_param(
        name: impl Into<String>,
        description: impl Into<String>,
        values: Vec<String>,
        required: bool,
    ) -> Self {
        Self {
            name: name.into(),
            param_type: "string".to_string(),
            description: description.into(),
            required,
            default: None,
            enum_values: Some(values),
        }
    }
}

/// Extended tool definition with parameter schemas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name (must be unique)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Parameter schemas
    pub parameters: Vec<ParameterSchema>,
    /// Category for grouping
    pub category: ToolCategory,
    /// Whether this tool requires confirmation before execution
    pub requires_confirmation: bool,
    /// Whether this tool can modify state (files, network, etc.)
    pub is_destructive: bool,
    /// Timeout in milliseconds (0 = use default)
    pub timeout_ms: u64,
}

impl ToolSpec {
    /// Create a new tool spec
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
            category: ToolCategory::Utility,
            requires_confirmation: false,
            is_destructive: false,
            timeout_ms: 0,
        }
    }

    /// Add a parameter
    pub fn with_param(mut self, param: ParameterSchema) -> Self {
        self.parameters.push(param);
        self
    }

    /// Set category
    pub fn with_category(mut self, category: ToolCategory) -> Self {
        self.category = category;
        self
    }

    /// Mark as requiring confirmation
    pub fn requires_confirmation(mut self) -> Self {
        self.requires_confirmation = true;
        self
    }

    /// Mark as destructive
    pub fn destructive(mut self) -> Self {
        self.is_destructive = true;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Convert to JSON schema for LLM tool definitions
    pub fn to_json_schema(&self) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            let mut prop = serde_json::Map::new();
            prop.insert("type".to_string(), serde_json::json!(param.param_type));
            prop.insert(
                "description".to_string(),
                serde_json::json!(param.description),
            );

            if let Some(ref values) = param.enum_values {
                prop.insert("enum".to_string(), serde_json::json!(values));
            }

            if let Some(ref default) = param.default {
                prop.insert("default".to_string(), default.clone());
            }

            properties.insert(param.name.clone(), serde_json::Value::Object(prop));

            if param.required {
                required.push(param.name.clone());
            }
        }

        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required
        })
    }

    /// Convert to providers::types::ToolDefinition
    pub fn to_tool_definition(&self) -> crate::providers::types::ToolDefinition {
        crate::providers::types::ToolDefinition::new(
            &self.name,
            &self.description,
            self.to_json_schema(),
        )
    }
}

/// Tool categories for organization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCategory {
    /// File system operations
    FileSystem,
    /// Shell/command execution
    Shell,
    /// Network operations
    Network,
    /// Search and exploration
    Search,
    /// Code analysis and modification
    Code,
    /// System information
    System,
    /// General utilities
    Utility,
}

impl std::fmt::Display for ToolCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileSystem => write!(f, "File System"),
            Self::Shell => write!(f, "Shell"),
            Self::Network => write!(f, "Network"),
            Self::Search => write!(f, "Search"),
            Self::Code => write!(f, "Code"),
            Self::System => write!(f, "System"),
            Self::Utility => write!(f, "Utility"),
        }
    }
}

// =============================================================================
// SANDBOX CONFIGURATION
// =============================================================================

/// Sandbox configuration for tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Allowed directories for file operations
    pub allowed_paths: Vec<String>,
    /// Blocked directories (takes precedence over allowed)
    pub blocked_paths: Vec<String>,
    /// Allowed commands for shell execution
    pub allowed_commands: Vec<String>,
    /// Blocked commands (takes precedence over allowed)
    pub blocked_commands: Vec<String>,
    /// Allowed network hosts
    pub allowed_hosts: Vec<String>,
    /// Blocked network hosts
    pub blocked_hosts: Vec<String>,
    /// Maximum file size to read (bytes)
    pub max_file_size: usize,
    /// Maximum output size (bytes)
    pub max_output_size: usize,
    /// Default timeout for commands (ms)
    pub default_timeout_ms: u64,
    /// Whether to allow writes
    pub allow_writes: bool,
    /// Whether to allow deletes
    pub allow_deletes: bool,
    /// Whether to allow network access
    pub allow_network: bool,
    /// Whether to allow shell commands
    pub allow_shell: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            allowed_paths: vec![".".to_string()], // Current directory only by default
            blocked_paths: vec![
                "/etc".to_string(),
                "/var".to_string(),
                "/root".to_string(),
                "~/.ssh".to_string(),
                "~/.gnupg".to_string(),
                "~/.aws".to_string(),
                ".env".to_string(),
                ".git/config".to_string(),
            ],
            allowed_commands: vec![
                "ls".to_string(),
                "cat".to_string(),
                "head".to_string(),
                "tail".to_string(),
                "grep".to_string(),
                "find".to_string(),
                "wc".to_string(),
                "sort".to_string(),
                "uniq".to_string(),
                "echo".to_string(),
                "pwd".to_string(),
                "date".to_string(),
                "whoami".to_string(),
            ],
            blocked_commands: vec![
                "rm".to_string(),
                "rmdir".to_string(),
                "mv".to_string(),
                "cp".to_string(),
                "chmod".to_string(),
                "chown".to_string(),
                "sudo".to_string(),
                "su".to_string(),
                "curl".to_string(),
                "wget".to_string(),
                "ssh".to_string(),
                "scp".to_string(),
                "rsync".to_string(),
            ],
            allowed_hosts: vec!["localhost".to_string(), "127.0.0.1".to_string()],
            blocked_hosts: Vec::new(),
            max_file_size: 10 * 1024 * 1024, // 10MB
            max_output_size: 1024 * 1024,    // 1MB
            default_timeout_ms: 30_000,      // 30 seconds
            allow_writes: false,
            allow_deletes: false,
            allow_network: false,
            allow_shell: true,
        }
    }
}

impl SandboxConfig {
    /// Create a permissive sandbox (for trusted environments)
    pub fn permissive() -> Self {
        Self {
            allowed_paths: vec!["/".to_string()],
            blocked_paths: vec!["/etc/shadow".to_string(), "/etc/passwd".to_string()],
            allowed_commands: vec!["*".to_string()], // All commands
            blocked_commands: vec!["sudo".to_string(), "su".to_string()],
            allowed_hosts: vec!["*".to_string()], // All hosts
            blocked_hosts: Vec::new(),
            max_file_size: 100 * 1024 * 1024,  // 100MB
            max_output_size: 10 * 1024 * 1024, // 10MB
            default_timeout_ms: 120_000,       // 2 minutes
            allow_writes: true,
            allow_deletes: true,
            allow_network: true,
            allow_shell: true,
        }
    }

    /// Create a read-only sandbox
    pub fn read_only() -> Self {
        Self {
            allow_writes: false,
            allow_deletes: false,
            allow_network: false,
            ..Self::default()
        }
    }

    /// Check if a path is allowed
    pub fn is_path_allowed(&self, path: &str) -> bool {
        // Check blocked paths first
        for blocked in &self.blocked_paths {
            if path.starts_with(blocked) || path.contains(blocked) {
                return false;
            }
        }

        // Check allowed paths
        if self.allowed_paths.iter().any(|p| p == "*") {
            return true;
        }

        for allowed in &self.allowed_paths {
            if path.starts_with(allowed) {
                return true;
            }
        }

        false
    }

    /// Check if a command is allowed
    pub fn is_command_allowed(&self, command: &str) -> bool {
        // Extract the base command (first word)
        let base_cmd = command.split_whitespace().next().unwrap_or(command);

        // Check blocked commands first
        for blocked in &self.blocked_commands {
            if base_cmd == blocked || command.contains(blocked) {
                return false;
            }
        }

        // Check allowed commands
        if self.allowed_commands.iter().any(|c| c == "*") {
            return true;
        }

        self.allowed_commands.iter().any(|c| c == base_cmd)
    }

    /// Check if a host is allowed
    pub fn is_host_allowed(&self, host: &str) -> bool {
        // Check blocked hosts first
        for blocked in &self.blocked_hosts {
            if host == blocked || host.ends_with(blocked) {
                return false;
            }
        }

        // Check allowed hosts
        if self.allowed_hosts.iter().any(|h| h == "*") {
            return true;
        }

        self.allowed_hosts
            .iter()
            .any(|h| h == host || host.ends_with(h))
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_error_display() {
        let err = ToolError::not_found("bash");
        assert!(err.to_string().contains("bash"));

        let err = ToolError::timeout(5000);
        assert!(err.to_string().contains("5000"));
    }

    #[test]
    fn test_tool_output_success() {
        let output = ToolOutput::success("hello world");
        assert!(output.success);
        assert_eq!(output.output, "hello world");
        assert!(output.error.is_none());
    }

    #[test]
    fn test_tool_output_failure() {
        let output = ToolOutput::failure("partial output", "command failed");
        assert!(!output.success);
        assert_eq!(output.error, Some("command failed".to_string()));
    }

    #[test]
    fn test_parameter_schema() {
        let param = ParameterSchema::required_string("path", "File path to read");
        assert_eq!(param.name, "path");
        assert!(param.required);
        assert_eq!(param.param_type, "string");
    }

    #[test]
    fn test_tool_spec_to_json_schema() {
        let spec = ToolSpec::new("read_file", "Read a file")
            .with_param(ParameterSchema::required_string("path", "File path"))
            .with_param(ParameterSchema::optional_number(
                "limit",
                "Max lines",
                Some(100.0),
            ));

        let schema = spec.to_json_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema.get("required").is_some());
    }

    #[test]
    fn test_sandbox_path_allowed() {
        let sandbox = SandboxConfig::default();

        assert!(sandbox.is_path_allowed("./src/main.rs"));
        assert!(!sandbox.is_path_allowed("/etc/passwd"));
        assert!(!sandbox.is_path_allowed("~/.ssh/id_rsa"));
    }

    #[test]
    fn test_sandbox_command_allowed() {
        let sandbox = SandboxConfig::default();

        assert!(sandbox.is_command_allowed("ls -la"));
        assert!(sandbox.is_command_allowed("grep pattern file.txt"));
        assert!(!sandbox.is_command_allowed("rm -rf /"));
        assert!(!sandbox.is_command_allowed("sudo ls"));
    }

    #[test]
    fn test_sandbox_permissive() {
        let sandbox = SandboxConfig::permissive();

        assert!(sandbox.allow_writes);
        assert!(sandbox.allow_deletes);
        assert!(sandbox.allow_network);
    }

    #[test]
    fn test_sandbox_read_only() {
        let sandbox = SandboxConfig::read_only();

        assert!(!sandbox.allow_writes);
        assert!(!sandbox.allow_deletes);
        assert!(!sandbox.allow_network);
    }

    #[test]
    fn test_tool_category_display() {
        assert_eq!(format!("{}", ToolCategory::FileSystem), "File System");
        assert_eq!(format!("{}", ToolCategory::Shell), "Shell");
    }
}
