//! # Tools Module
//!
//! Sandboxed tool execution system for AI agents.
//!
//! This module provides a comprehensive tool execution framework with:
//! - Sandbox configuration for security
//! - Shell command execution (bash)
//! - File system operations (read, write, list, search)
//! - Tool registry for managing multiple tools
//! - Hooks for observability and interception
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use torus_attention::tools::{ToolRegistry, ToolExecutor};
//! use torus_attention::providers::types::ToolCall;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create a registry with default tools
//!     let registry = ToolRegistry::with_defaults();
//!
//!     // Execute a tool
//!     let call = ToolCall {
//!         id: "1".to_string(),
//!         name: "bash".to_string(),
//!         arguments: r#"{"command": "echo hello"}"#.to_string(),
//!     };
//!
//!     let result = registry.execute(&call).await;
//!     println!("{:?}", result);
//! }
//! ```
//!
//! ## Available Tools
//!
//! - `bash` - Execute shell commands
//! - `read` - Read file contents
//! - `write` - Write to files
//! - `glob` - List directory contents
//! - `grep` - Search files for content
//!
//! ## Sandboxing
//!
//! All tools respect `SandboxConfig` which controls:
//! - Allowed/blocked paths
//! - Allowed/blocked commands
//! - Network access
//! - Write/delete permissions
//! - Size limits
//!
//! ```rust,ignore
//! use torus_attention::tools::{ToolRegistry, SandboxConfig};
//!
//! // Create a read-only sandbox
//! let sandbox = SandboxConfig::read_only();
//! let registry = ToolRegistry::with_sandbox(sandbox);
//!
//! // Create a permissive sandbox (for trusted environments)
//! let registry = ToolRegistry::permissive();
//! ```

pub mod bash;
pub mod file;
pub mod registry;
pub mod traits;
pub mod types;

// Re-export core types
pub use types::{
    ParameterSchema, SandboxConfig, ToolCategory, ToolError, ToolOutput, ToolResult, ToolSpec,
};

// Re-export traits
pub use traits::{
    AutoConfirm, ConfirmationHandler, DenyAll, LogLevel, LoggingHook, NoOpHook, SandboxedTool,
    Tool, ToolExecutor, ToolHook,
};

// Re-export tools
pub use bash::BashTool;
pub use file::{FileTool, ListDirectoryTool, ReadFileTool, SearchFilesTool, WriteFileTool};

// Re-export registry
pub use registry::{RegistrySummary, ToolRegistry, ToolRegistryBuilder};
