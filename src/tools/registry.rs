//! # Tool Registry
//!
//! A registry for managing and executing tools.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::bash::BashTool;
use super::file::{ListDirectoryTool, ReadFileTool, SearchFilesTool, WriteFileTool};
use super::traits::{ConfirmationHandler, Tool, ToolExecutor, ToolHook};
use super::types::{SandboxConfig, ToolCategory, ToolError, ToolOutput, ToolResult, ToolSpec};
use crate::providers::types::{ToolCall, ToolDefinition};

// =============================================================================
// TOOL REGISTRY
// =============================================================================

/// A registry that manages multiple tools and executes them by name
pub struct ToolRegistry {
    /// Registered tools by name
    tools: HashMap<String, Arc<dyn Tool>>,
    /// Execution hooks
    hooks: Vec<Arc<dyn ToolHook>>,
    /// Confirmation handler for destructive operations
    confirmation_handler: Option<Arc<dyn ConfirmationHandler>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            hooks: Vec::new(),
            confirmation_handler: None,
        }
    }

    /// Create a registry with default tools (bash, read, write, glob, grep)
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register_default_tools();
        registry
    }

    /// Create a registry with permissive sandbox defaults
    pub fn permissive() -> Self {
        let mut registry = Self::new();
        registry.register(Arc::new(BashTool::permissive()));
        registry.register(Arc::new(ReadFileTool::permissive()));
        registry.register(Arc::new(WriteFileTool::permissive()));
        registry.register(Arc::new(ListDirectoryTool::permissive()));
        registry.register(Arc::new(SearchFilesTool::permissive()));
        registry
    }

    /// Create a registry with a specific sandbox configuration
    pub fn with_sandbox(sandbox: SandboxConfig) -> Self {
        let mut registry = Self::new();
        registry.register(Arc::new(BashTool::with_sandbox(sandbox.clone())));
        registry.register(Arc::new(ReadFileTool::with_sandbox(sandbox.clone())));
        registry.register(Arc::new(WriteFileTool::with_sandbox(sandbox.clone())));
        registry.register(Arc::new(ListDirectoryTool::with_sandbox(sandbox.clone())));
        registry.register(Arc::new(SearchFilesTool::with_sandbox(sandbox)));
        registry
    }

    /// Register default tools with default sandbox
    fn register_default_tools(&mut self) {
        self.register(Arc::new(BashTool::new()));
        self.register(Arc::new(ReadFileTool::new()));
        self.register(Arc::new(WriteFileTool::new()));
        self.register(Arc::new(ListDirectoryTool::new()));
        self.register(Arc::new(SearchFilesTool::new()));
    }

    /// Register a tool
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> &mut Self {
        let name = tool.spec().name.clone();
        self.tools.insert(name, tool);
        self
    }

    /// Unregister a tool by name
    pub fn unregister(&mut self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.remove(name)
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Check if a tool exists
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get all tool names
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Get tools by category
    pub fn tools_by_category(&self, category: ToolCategory) -> Vec<&Arc<dyn Tool>> {
        self.tools
            .values()
            .filter(|t| t.spec().category == category)
            .collect()
    }

    /// Add an execution hook
    pub fn add_hook(&mut self, hook: Arc<dyn ToolHook>) -> &mut Self {
        self.hooks.push(hook);
        self
    }

    /// Set confirmation handler
    pub fn set_confirmation_handler(
        &mut self,
        handler: Arc<dyn ConfirmationHandler>,
    ) -> &mut Self {
        self.confirmation_handler = Some(handler);
        self
    }

    /// Execute a tool call with hooks
    async fn execute_with_hooks(&self, call: &ToolCall) -> ToolResult<ToolOutput> {
        // Get the tool
        let tool = self
            .tools
            .get(&call.name)
            .ok_or_else(|| ToolError::not_found(&call.name))?;

        // Check if we should execute (hook veto)
        for hook in &self.hooks {
            if !hook.should_execute(call).await {
                return Err(ToolError::Cancelled);
            }
        }

        // Check for confirmation on destructive operations
        let spec = tool.spec();
        if spec.requires_confirmation {
            if let Some(ref handler) = self.confirmation_handler {
                let reason = format!(
                    "Tool '{}' is marked as destructive and requires confirmation",
                    call.name
                );
                if !handler.confirm(call, &reason).await {
                    return Err(ToolError::Cancelled);
                }
            }
        }

        // Run before hooks
        for hook in &self.hooks {
            hook.before_execute(call).await?;
        }

        // Execute the tool
        let result = tool.execute(call).await;

        // Run after hooks
        for hook in &self.hooks {
            hook.after_execute(call, &result).await;
        }

        result
    }

    /// Get the number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Get a summary of registered tools
    pub fn summary(&self) -> RegistrySummary {
        let mut by_category: HashMap<ToolCategory, Vec<String>> = HashMap::new();

        for tool in self.tools.values() {
            let spec = tool.spec();
            by_category
                .entry(spec.category)
                .or_default()
                .push(spec.name.clone());
        }

        RegistrySummary {
            total_tools: self.tools.len(),
            tools_by_category: by_category,
        }
    }
}

impl ToolExecutor for ToolRegistry {
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>> {
        Box::pin(async move { self.execute_with_hooks(call).await })
    }

    fn has_tool(&self, name: &str) -> bool {
        self.has(name)
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|t| t.spec().to_tool_definition())
            .collect()
    }

    fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec()).collect()
    }
}

// =============================================================================
// REGISTRY SUMMARY
// =============================================================================

/// Summary of registered tools
#[derive(Debug, Clone)]
pub struct RegistrySummary {
    /// Total number of tools
    pub total_tools: usize,
    /// Tools grouped by category
    pub tools_by_category: HashMap<ToolCategory, Vec<String>>,
}

impl std::fmt::Display for RegistrySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tool Registry ({} tools):", self.total_tools)?;
        for (category, tools) in &self.tools_by_category {
            writeln!(f, "  {}:", category)?;
            for tool in tools {
                writeln!(f, "    - {}", tool)?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// BUILDER PATTERN
// =============================================================================

/// Builder for creating a ToolRegistry with custom configuration
pub struct ToolRegistryBuilder {
    sandbox: Option<SandboxConfig>,
    include_bash: bool,
    include_file_tools: bool,
    custom_tools: Vec<Arc<dyn Tool>>,
    hooks: Vec<Arc<dyn ToolHook>>,
    confirmation_handler: Option<Arc<dyn ConfirmationHandler>>,
}

impl Default for ToolRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistryBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            sandbox: None,
            include_bash: true,
            include_file_tools: true,
            custom_tools: Vec::new(),
            hooks: Vec::new(),
            confirmation_handler: None,
        }
    }

    /// Set sandbox configuration
    pub fn with_sandbox(mut self, sandbox: SandboxConfig) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// Exclude bash tool
    pub fn without_bash(mut self) -> Self {
        self.include_bash = false;
        self
    }

    /// Exclude file tools
    pub fn without_file_tools(mut self) -> Self {
        self.include_file_tools = false;
        self
    }

    /// Add a custom tool
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.custom_tools.push(tool);
        self
    }

    /// Add an execution hook
    pub fn with_hook(mut self, hook: Arc<dyn ToolHook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Set confirmation handler
    pub fn with_confirmation_handler(mut self, handler: Arc<dyn ConfirmationHandler>) -> Self {
        self.confirmation_handler = Some(handler);
        self
    }

    /// Build the registry
    pub fn build(self) -> ToolRegistry {
        let mut registry = ToolRegistry::new();

        let sandbox = self.sandbox.unwrap_or_default();

        if self.include_bash {
            registry.register(Arc::new(BashTool::with_sandbox(sandbox.clone())));
        }

        if self.include_file_tools {
            registry.register(Arc::new(ReadFileTool::with_sandbox(sandbox.clone())));
            registry.register(Arc::new(WriteFileTool::with_sandbox(sandbox.clone())));
            registry.register(Arc::new(ListDirectoryTool::with_sandbox(sandbox.clone())));
            registry.register(Arc::new(SearchFilesTool::with_sandbox(sandbox)));
        }

        for tool in self.custom_tools {
            registry.register(tool);
        }

        for hook in self.hooks {
            registry.add_hook(hook);
        }

        if let Some(handler) = self.confirmation_handler {
            registry.set_confirmation_handler(handler);
        }

        registry
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::traits::{AutoConfirm, DenyAll};

    fn make_call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "test-1".to_string(),
            name: name.to_string(),
            arguments: args.to_string(),
        }
    }

    #[test]
    fn test_registry_new() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = ToolRegistry::with_defaults();
        assert!(!registry.is_empty());
        assert!(registry.has("bash"));
        assert!(registry.has("read"));
        assert!(registry.has("write"));
        assert!(registry.has("glob"));
        assert!(registry.has("grep"));
    }

    #[test]
    fn test_registry_permissive() {
        let registry = ToolRegistry::permissive();
        assert_eq!(registry.len(), 5);
    }

    #[test]
    fn test_register_unregister() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(BashTool::new()));

        assert!(registry.has("bash"));
        assert_eq!(registry.len(), 1);

        let removed = registry.unregister("bash");
        assert!(removed.is_some());
        assert!(!registry.has("bash"));
    }

    #[test]
    fn test_tool_names() {
        let registry = ToolRegistry::with_defaults();
        let names = registry.tool_names();

        assert!(names.contains(&"bash".to_string()));
        assert!(names.contains(&"read".to_string()));
    }

    #[test]
    fn test_tools_by_category() {
        let registry = ToolRegistry::with_defaults();

        let shell_tools = registry.tools_by_category(ToolCategory::Shell);
        assert_eq!(shell_tools.len(), 1);

        let fs_tools = registry.tools_by_category(ToolCategory::FileSystem);
        assert_eq!(fs_tools.len(), 3); // read, write, glob
    }

    #[test]
    fn test_tool_definitions() {
        let registry = ToolRegistry::with_defaults();
        let definitions = registry.tool_definitions();

        assert_eq!(definitions.len(), 5);
        assert!(definitions.iter().any(|d| d.name == "bash"));
    }

    #[tokio::test]
    async fn test_execute_tool() {
        let registry = ToolRegistry::permissive();
        let call = make_call("bash", serde_json::json!({"command": "echo hello"}));

        let result = registry.execute(&call).await;
        assert!(result.is_ok());
        assert!(result.unwrap().output.contains("hello"));
    }

    #[tokio::test]
    async fn test_execute_nonexistent_tool() {
        let registry = ToolRegistry::new();
        let call = make_call("nonexistent", serde_json::json!({}));

        let result = registry.execute(&call).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            ToolError::NotFound { name } => assert_eq!(name, "nonexistent"),
            other => panic!("Expected NotFound, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_confirmation_handler_deny() {
        let mut registry = ToolRegistry::permissive();
        registry.set_confirmation_handler(Arc::new(DenyAll));

        // Write tool requires confirmation
        let call = make_call(
            "write",
            serde_json::json!({
                "filePath": "/tmp/test.txt",
                "content": "hello"
            }),
        );

        let result = registry.execute(&call).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            ToolError::Cancelled => {}
            other => panic!("Expected Cancelled, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_confirmation_handler_allow() {
        let mut registry = ToolRegistry::permissive();
        registry.set_confirmation_handler(Arc::new(AutoConfirm::default()));

        // Bash tool with echo should work
        let call = make_call("bash", serde_json::json!({"command": "echo test"}));

        let result = registry.execute(&call).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_registry_summary() {
        let registry = ToolRegistry::with_defaults();
        let summary = registry.summary();

        assert_eq!(summary.total_tools, 5);
        assert!(summary.tools_by_category.contains_key(&ToolCategory::Shell));
        assert!(summary
            .tools_by_category
            .contains_key(&ToolCategory::FileSystem));
    }

    #[test]
    fn test_builder_default() {
        let registry = ToolRegistryBuilder::new().build();
        assert_eq!(registry.len(), 5);
    }

    #[test]
    fn test_builder_without_bash() {
        let registry = ToolRegistryBuilder::new().without_bash().build();
        assert!(!registry.has("bash"));
        assert!(registry.has("read"));
    }

    #[test]
    fn test_builder_without_file_tools() {
        let registry = ToolRegistryBuilder::new().without_file_tools().build();
        assert!(registry.has("bash"));
        assert!(!registry.has("read"));
        assert!(!registry.has("write"));
    }

    #[test]
    fn test_builder_with_sandbox() {
        let sandbox = SandboxConfig::read_only();
        let registry = ToolRegistryBuilder::new().with_sandbox(sandbox).build();

        assert_eq!(registry.len(), 5);
    }

    #[test]
    fn test_has_tool_trait() {
        let registry = ToolRegistry::with_defaults();
        assert!(registry.has_tool("bash"));
        assert!(!registry.has_tool("nonexistent"));
    }
}
