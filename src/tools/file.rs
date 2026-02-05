//! # File Tool
//!
//! Sandboxed file system operations.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Instant;
use tokio::fs;

use super::traits::{SandboxedTool, Tool};
use super::types::{
    ParameterSchema, SandboxConfig, ToolCategory, ToolError, ToolOutput, ToolResult, ToolSpec,
};
use crate::providers::types::ToolCall;

// =============================================================================
// FILE TOOL
// =============================================================================

/// A sandboxed file system tool for reading, writing, and searching files
pub struct FileTool {
    /// Sandbox configuration
    pub(crate) sandbox: SandboxConfig,
    /// Base directory for relative paths
    base_dir: PathBuf,
}

impl Default for FileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl FileTool {
    /// Create a new file tool with default sandbox
    pub fn new() -> Self {
        Self {
            sandbox: SandboxConfig::default(),
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    /// Create with a specific sandbox configuration
    pub fn with_sandbox(sandbox: SandboxConfig) -> Self {
        Self {
            sandbox,
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    /// Create a permissive file tool (for trusted environments)
    pub fn permissive() -> Self {
        Self {
            sandbox: SandboxConfig::permissive(),
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    /// Set base directory for relative paths
    pub fn with_base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.base_dir = base_dir.into();
        self
    }

    /// Resolve a path, making it absolute if relative
    pub(crate) fn resolve_path(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.base_dir.join(p)
        }
    }

    /// Check if a path is allowed by the sandbox
    fn check_path_allowed(&self, path: &Path) -> ToolResult<()> {
        let path_str = path.to_string_lossy();
        if !self.sandbox.is_path_allowed(&path_str) {
            return Err(ToolError::path_not_allowed(path_str));
        }
        Ok(())
    }

    /// Read a file with optional offset and limit
    pub(crate) async fn read_file_impl(
        &self,
        path: &Path,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> ToolResult<ToolOutput> {
        let start = Instant::now();

        // Check path is allowed
        self.check_path_allowed(path)?;

        // Check file exists
        if !path.exists() {
            return Err(ToolError::io_error(format!(
                "File not found: {}",
                path.display()
            )));
        }

        // Check file size
        let metadata = fs::metadata(path)
            .await
            .map_err(|e| ToolError::io_error(format!("Failed to get file metadata: {}", e)))?;

        if metadata.len() as usize > self.sandbox.max_file_size {
            return Err(ToolError::io_error(format!(
                "File too large: {} bytes (max: {} bytes)",
                metadata.len(),
                self.sandbox.max_file_size
            )));
        }

        // Read file
        let content = fs::read_to_string(path)
            .await
            .map_err(|e| ToolError::io_error(format!("Failed to read file: {}", e)))?;

        // Apply offset and limit (line-based)
        let lines: Vec<&str> = content.lines().collect();
        let offset = offset.unwrap_or(0);
        let limit = limit.unwrap_or(2000);

        let selected_lines: Vec<String> = lines
            .iter()
            .skip(offset)
            .take(limit)
            .enumerate()
            .map(|(i, line)| format!("{:5}\t{}", offset + i + 1, line))
            .collect();

        let output = selected_lines.join("\n");
        let execution_time = start.elapsed().as_millis() as u64;

        Ok(ToolOutput::success(output)
            .with_execution_time(execution_time)
            .with_metadata("lines_total", serde_json::json!(lines.len()))
            .with_metadata("lines_returned", serde_json::json!(selected_lines.len()))
            .with_metadata("offset", serde_json::json!(offset))
            .with_metadata("path", serde_json::json!(path.to_string_lossy())))
    }

    /// Write content to a file
    pub(crate) async fn write_file_impl(&self, path: &Path, content: &str) -> ToolResult<ToolOutput> {
        let start = Instant::now();

        // Check writes are allowed
        if !self.sandbox.allow_writes {
            return Err(ToolError::permission_denied("File writes are disabled"));
        }

        // Check path is allowed
        self.check_path_allowed(path)?;

        // Check content size
        if content.len() > self.sandbox.max_file_size {
            return Err(ToolError::io_error(format!(
                "Content too large: {} bytes (max: {} bytes)",
                content.len(),
                self.sandbox.max_file_size
            )));
        }

        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)
                    .await
                    .map_err(|e| ToolError::io_error(format!("Failed to create parent directories: {}", e)))?;
            }
        }

        // Write file
        fs::write(path, content)
            .await
            .map_err(|e| ToolError::io_error(format!("Failed to write file: {}", e)))?;

        let execution_time = start.elapsed().as_millis() as u64;

        Ok(
            ToolOutput::success(format!("Wrote {} bytes to {}", content.len(), path.display()))
                .with_execution_time(execution_time)
                .with_metadata("bytes_written", serde_json::json!(content.len()))
                .with_metadata("path", serde_json::json!(path.to_string_lossy())),
        )
    }

    /// List directory contents
    pub(crate) async fn list_directory_impl(
        &self,
        path: &Path,
        pattern: Option<&str>,
    ) -> ToolResult<ToolOutput> {
        let start = Instant::now();

        // Check path is allowed
        self.check_path_allowed(path)?;

        // Check directory exists
        if !path.exists() {
            return Err(ToolError::io_error(format!(
                "Directory not found: {}",
                path.display()
            )));
        }

        if !path.is_dir() {
            return Err(ToolError::io_error(format!(
                "Not a directory: {}",
                path.display()
            )));
        }

        // Read directory
        let mut entries = fs::read_dir(path)
            .await
            .map_err(|e| ToolError::io_error(format!("Failed to read directory: {}", e)))?;

        let mut items = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| ToolError::io_error(format!("Failed to read entry: {}", e)))?
        {
            let name = entry.file_name().to_string_lossy().to_string();

            // Apply pattern filter if provided
            if let Some(pat) = pattern {
                if !glob_match(pat, &name) {
                    continue;
                }
            }

            let file_type = entry.file_type().await.ok();
            let metadata = entry.metadata().await.ok();

            let type_indicator = match file_type {
                Some(ft) if ft.is_dir() => "/",
                Some(ft) if ft.is_symlink() => "@",
                _ => "",
            };

            let size = metadata.map(|m| m.len()).unwrap_or(0);

            items.push(format!("{}{}\t{} bytes", name, type_indicator, size));
        }

        items.sort();
        let output = items.join("\n");
        let execution_time = start.elapsed().as_millis() as u64;

        Ok(ToolOutput::success(output)
            .with_execution_time(execution_time)
            .with_metadata("count", serde_json::json!(items.len()))
            .with_metadata("path", serde_json::json!(path.to_string_lossy())))
    }

    /// Search for files matching a pattern
    pub(crate) async fn search_files_impl(
        &self,
        path: &Path,
        file_pattern: Option<&str>,
        content_pattern: Option<&str>,
        max_results: usize,
    ) -> ToolResult<ToolOutput> {
        let start = Instant::now();

        // Check path is allowed
        self.check_path_allowed(path)?;

        // Check directory exists
        if !path.exists() {
            return Err(ToolError::io_error(format!(
                "Path not found: {}",
                path.display()
            )));
        }

        let mut results = Vec::new();
        self.search_recursive(path, file_pattern, content_pattern, max_results, &mut results)
            .await?;

        let output = results.join("\n");
        let execution_time = start.elapsed().as_millis() as u64;

        Ok(ToolOutput::success(output)
            .with_execution_time(execution_time)
            .with_metadata("matches", serde_json::json!(results.len()))
            .with_metadata("path", serde_json::json!(path.to_string_lossy())))
    }

    /// Recursive search helper
    async fn search_recursive(
        &self,
        path: &Path,
        file_pattern: Option<&str>,
        content_pattern: Option<&str>,
        max_results: usize,
        results: &mut Vec<String>,
    ) -> ToolResult<()> {
        if results.len() >= max_results {
            return Ok(());
        }

        // Check path is allowed
        if !self.sandbox.is_path_allowed(&path.to_string_lossy()) {
            return Ok(()); // Silently skip disallowed paths
        }

        if path.is_file() {
            let name = path.file_name().map(|n| n.to_string_lossy().to_string());

            // Check file pattern
            let matches_pattern = match (file_pattern, &name) {
                (Some(pat), Some(n)) => glob_match(pat, n),
                (None, _) => true,
                (_, None) => false,
            };

            if matches_pattern {
                // Check content pattern if provided
                if let Some(content_pat) = content_pattern {
                    if let Ok(content) = fs::read_to_string(path).await {
                        if content.contains(content_pat) {
                            // Find matching lines
                            for (i, line) in content.lines().enumerate() {
                                if line.contains(content_pat) && results.len() < max_results {
                                    results.push(format!(
                                        "{}:{}: {}",
                                        path.display(),
                                        i + 1,
                                        line.trim()
                                    ));
                                }
                            }
                        }
                    }
                } else {
                    results.push(path.display().to_string());
                }
            }
        } else if path.is_dir() {
            if let Ok(mut entries) = fs::read_dir(path).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    if results.len() >= max_results {
                        break;
                    }
                    // Skip hidden files/directories
                    let name = entry.file_name();
                    if name.to_string_lossy().starts_with('.') {
                        continue;
                    }
                    Box::pin(self.search_recursive(
                        &entry.path(),
                        file_pattern,
                        content_pattern,
                        max_results,
                        results,
                    ))
                    .await?;
                }
            }
        }

        Ok(())
    }

    /// Parse arguments for read operation
    pub(crate) fn parse_read_args(&self, call: &ToolCall) -> ToolResult<ReadArgs> {
        serde_json::from_str(&call.arguments)
            .map_err(|e| ToolError::invalid_args(&call.name, format!("JSON parse error: {}", e)))
    }

    /// Parse arguments for write operation
    pub(crate) fn parse_write_args(&self, call: &ToolCall) -> ToolResult<WriteArgs> {
        serde_json::from_str(&call.arguments)
            .map_err(|e| ToolError::invalid_args(&call.name, format!("JSON parse error: {}", e)))
    }

    /// Parse arguments for list operation
    pub(crate) fn parse_list_args(&self, call: &ToolCall) -> ToolResult<ListArgs> {
        serde_json::from_str(&call.arguments)
            .map_err(|e| ToolError::invalid_args(&call.name, format!("JSON parse error: {}", e)))
    }

    /// Parse arguments for search operation
    pub(crate) fn parse_search_args(&self, call: &ToolCall) -> ToolResult<SearchArgs> {
        serde_json::from_str(&call.arguments)
            .map_err(|e| ToolError::invalid_args(&call.name, format!("JSON parse error: {}", e)))
    }
}

/// Simple glob pattern matching (supports * and ?)
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();

    fn matches(p: &[char], t: &[char]) -> bool {
        match (p.first(), t.first()) {
            (None, None) => true,
            (Some('*'), _) => {
                // Try matching zero or more characters
                matches(&p[1..], t) || (!t.is_empty() && matches(p, &t[1..]))
            }
            (Some('?'), Some(_)) => matches(&p[1..], &t[1..]),
            (Some(pc), Some(tc)) if *pc == *tc => matches(&p[1..], &t[1..]),
            (Some(_), None) => p.iter().all(|c| *c == '*'),
            _ => false,
        }
    }

    matches(&pattern_chars, &text_chars)
}

// =============================================================================
// ARGUMENT TYPES
// =============================================================================

#[derive(Debug, serde::Deserialize)]
pub(crate) struct ReadArgs {
    #[serde(rename = "filePath")]
    pub file_path: String,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize)]
pub(crate) struct WriteArgs {
    #[serde(rename = "filePath")]
    pub file_path: String,
    pub content: String,
}

#[derive(Debug, serde::Deserialize)]
pub(crate) struct ListArgs {
    pub path: String,
    pub pattern: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub(crate) struct SearchArgs {
    pub path: String,
    #[serde(rename = "filePattern")]
    pub file_pattern: Option<String>,
    #[serde(rename = "contentPattern")]
    pub content_pattern: Option<String>,
    #[serde(rename = "maxResults")]
    pub max_results: Option<usize>,
}

// =============================================================================
// INDIVIDUAL TOOLS (split for cleaner LLM tool definitions)
// =============================================================================

/// Read file tool
pub struct ReadFileTool {
    file_tool: FileTool,
}

impl ReadFileTool {
    pub fn new() -> Self {
        Self {
            file_tool: FileTool::new(),
        }
    }

    pub fn with_sandbox(sandbox: SandboxConfig) -> Self {
        Self {
            file_tool: FileTool::with_sandbox(sandbox),
        }
    }

    pub fn permissive() -> Self {
        Self {
            file_tool: FileTool::permissive(),
        }
    }

    async fn execute_impl(&self, call: &ToolCall) -> ToolResult<ToolOutput> {
        let args = self.file_tool.parse_read_args(call)?;
        let path = self.file_tool.resolve_path(&args.file_path);
        self.file_tool
            .read_file_impl(&path, args.offset, args.limit)
            .await
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("read", "Read contents of a file")
            .with_param(ParameterSchema::required_string(
                "filePath",
                "Absolute path to the file to read",
            ))
            .with_param(ParameterSchema::optional_number(
                "offset",
                "Line number to start reading from (0-based)",
                None,
            ))
            .with_param(ParameterSchema::optional_number(
                "limit",
                "Maximum number of lines to read (default: 2000)",
                Some(2000.0),
            ))
            .with_category(ToolCategory::FileSystem)
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>> {
        Box::pin(async move { self.execute_impl(call).await })
    }

    fn name(&self) -> &str {
        "read"
    }
}

impl SandboxedTool for ReadFileTool {
    fn sandbox_config(&self) -> &SandboxConfig {
        &self.file_tool.sandbox
    }

    fn set_sandbox_config(&mut self, config: SandboxConfig) {
        self.file_tool.sandbox = config;
    }

    fn check_sandbox(&self, operation: &str, target: &str) -> ToolResult<()> {
        match operation {
            "read" => {
                if !self.file_tool.sandbox.is_path_allowed(target) {
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

/// Write file tool
pub struct WriteFileTool {
    file_tool: FileTool,
}

impl WriteFileTool {
    pub fn new() -> Self {
        Self {
            file_tool: FileTool::new(),
        }
    }

    pub fn with_sandbox(sandbox: SandboxConfig) -> Self {
        Self {
            file_tool: FileTool::with_sandbox(sandbox),
        }
    }

    pub fn permissive() -> Self {
        Self {
            file_tool: FileTool::permissive(),
        }
    }

    async fn execute_impl(&self, call: &ToolCall) -> ToolResult<ToolOutput> {
        let args = self.file_tool.parse_write_args(call)?;
        let path = self.file_tool.resolve_path(&args.file_path);
        self.file_tool.write_file_impl(&path, &args.content).await
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WriteFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("write", "Write content to a file")
            .with_param(ParameterSchema::required_string(
                "filePath",
                "Absolute path to the file to write",
            ))
            .with_param(ParameterSchema::required_string(
                "content",
                "Content to write to the file",
            ))
            .with_category(ToolCategory::FileSystem)
            .destructive()
            .requires_confirmation()
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>> {
        Box::pin(async move { self.execute_impl(call).await })
    }

    fn name(&self) -> &str {
        "write"
    }
}

impl SandboxedTool for WriteFileTool {
    fn sandbox_config(&self) -> &SandboxConfig {
        &self.file_tool.sandbox
    }

    fn set_sandbox_config(&mut self, config: SandboxConfig) {
        self.file_tool.sandbox = config;
    }

    fn check_sandbox(&self, operation: &str, target: &str) -> ToolResult<()> {
        match operation {
            "write" => {
                if !self.file_tool.sandbox.allow_writes {
                    return Err(ToolError::permission_denied("Writes are disabled"));
                }
                if !self.file_tool.sandbox.is_path_allowed(target) {
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

/// List directory tool
pub struct ListDirectoryTool {
    file_tool: FileTool,
}

impl ListDirectoryTool {
    pub fn new() -> Self {
        Self {
            file_tool: FileTool::new(),
        }
    }

    pub fn with_sandbox(sandbox: SandboxConfig) -> Self {
        Self {
            file_tool: FileTool::with_sandbox(sandbox),
        }
    }

    pub fn permissive() -> Self {
        Self {
            file_tool: FileTool::permissive(),
        }
    }

    async fn execute_impl(&self, call: &ToolCall) -> ToolResult<ToolOutput> {
        let args = self.file_tool.parse_list_args(call)?;
        let path = self.file_tool.resolve_path(&args.path);
        self.file_tool
            .list_directory_impl(&path, args.pattern.as_deref())
            .await
    }
}

impl Default for ListDirectoryTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ListDirectoryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("glob", "List directory contents matching a pattern")
            .with_param(ParameterSchema::required_string(
                "path",
                "Directory path to list",
            ))
            .with_param(ParameterSchema::optional_string(
                "pattern",
                "Glob pattern to filter files (e.g., *.rs)",
                None,
            ))
            .with_category(ToolCategory::FileSystem)
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>> {
        Box::pin(async move { self.execute_impl(call).await })
    }

    fn name(&self) -> &str {
        "glob"
    }
}

/// Search files tool
pub struct SearchFilesTool {
    file_tool: FileTool,
}

impl SearchFilesTool {
    pub fn new() -> Self {
        Self {
            file_tool: FileTool::new(),
        }
    }

    pub fn with_sandbox(sandbox: SandboxConfig) -> Self {
        Self {
            file_tool: FileTool::with_sandbox(sandbox),
        }
    }

    pub fn permissive() -> Self {
        Self {
            file_tool: FileTool::permissive(),
        }
    }

    async fn execute_impl(&self, call: &ToolCall) -> ToolResult<ToolOutput> {
        let args = self.file_tool.parse_search_args(call)?;
        let path = self.file_tool.resolve_path(&args.path);
        let max_results = args.max_results.unwrap_or(100);
        self.file_tool
            .search_files_impl(
                &path,
                args.file_pattern.as_deref(),
                args.content_pattern.as_deref(),
                max_results,
            )
            .await
    }
}

impl Default for SearchFilesTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for SearchFilesTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("grep", "Search for files and content matching patterns")
            .with_param(ParameterSchema::required_string(
                "path",
                "Directory path to search in",
            ))
            .with_param(ParameterSchema::optional_string(
                "filePattern",
                "Glob pattern to filter files (e.g., *.rs)",
                None,
            ))
            .with_param(ParameterSchema::optional_string(
                "contentPattern",
                "Text pattern to search for in file contents",
                None,
            ))
            .with_param(ParameterSchema::optional_number(
                "maxResults",
                "Maximum number of results to return (default: 100)",
                Some(100.0),
            ))
            .with_category(ToolCategory::Search)
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>> {
        Box::pin(async move { self.execute_impl(call).await })
    }

    fn name(&self) -> &str {
        "grep"
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_read_call(path: &str) -> ToolCall {
        ToolCall {
            id: "test-1".to_string(),
            name: "read".to_string(),
            arguments: serde_json::json!({
                "filePath": path
            })
            .to_string(),
        }
    }

    fn make_write_call(path: &str, content: &str) -> ToolCall {
        ToolCall {
            id: "test-1".to_string(),
            name: "write".to_string(),
            arguments: serde_json::json!({
                "filePath": path,
                "content": content
            })
            .to_string(),
        }
    }

    fn make_list_call(path: &str) -> ToolCall {
        ToolCall {
            id: "test-1".to_string(),
            name: "glob".to_string(),
            arguments: serde_json::json!({
                "path": path
            })
            .to_string(),
        }
    }

    fn make_search_call(path: &str, content_pattern: &str) -> ToolCall {
        ToolCall {
            id: "test-1".to_string(),
            name: "grep".to_string(),
            arguments: serde_json::json!({
                "path": path,
                "contentPattern": content_pattern
            })
            .to_string(),
        }
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(glob_match("*.rs", "test.rs"));
        assert!(!glob_match("*.rs", "main.py"));
        assert!(glob_match("test_*", "test_foo"));
        assert!(glob_match("test_*", "test_"));
        assert!(glob_match("?.rs", "a.rs"));
        assert!(!glob_match("?.rs", "ab.rs"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("foo*bar", "foobar"));
        assert!(glob_match("foo*bar", "foo123bar"));
    }

    #[tokio::test]
    async fn test_read_file() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("test.txt");
        std::fs::write(&file_path, "line 1\nline 2\nline 3").unwrap();

        let tool = ReadFileTool::permissive();
        let call = make_read_call(file_path.to_str().unwrap());

        let result = tool.execute(&call).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.success);
        assert!(output.output.contains("line 1"));
        assert!(output.output.contains("line 2"));
    }

    #[tokio::test]
    async fn test_read_file_with_offset() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("test.txt");
        std::fs::write(&file_path, "line 1\nline 2\nline 3\nline 4\nline 5").unwrap();

        let tool = ReadFileTool::permissive();
        let call = ToolCall {
            id: "test-1".to_string(),
            name: "read".to_string(),
            arguments: serde_json::json!({
                "filePath": file_path.to_str().unwrap(),
                "offset": 2,
                "limit": 2
            })
            .to_string(),
        };

        let result = tool.execute(&call).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.output.contains("line 3"));
        assert!(output.output.contains("line 4"));
        assert!(!output.output.contains("line 1"));
        assert!(!output.output.contains("line 5"));
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let tool = ReadFileTool::permissive();
        let call = make_read_call("/nonexistent/file.txt");

        let result = tool.execute(&call).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_write_file() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("new_file.txt");

        let tool = WriteFileTool::permissive();
        let call = make_write_call(file_path.to_str().unwrap(), "hello world");

        let result = tool.execute(&call).await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "hello world");
    }

    #[tokio::test]
    async fn test_write_file_blocked_by_sandbox() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("blocked.txt");

        let tool = WriteFileTool::new(); // Default sandbox disables writes
        let call = make_write_call(file_path.to_str().unwrap(), "content");

        let result = tool.execute(&call).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            ToolError::PermissionDenied { .. } => {}
            other => panic!("Expected PermissionDenied, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_list_directory() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("file1.txt"), "").unwrap();
        std::fs::write(tmp.path().join("file2.rs"), "").unwrap();
        std::fs::create_dir(tmp.path().join("subdir")).unwrap();

        let tool = ListDirectoryTool::permissive();
        let call = make_list_call(tmp.path().to_str().unwrap());

        let result = tool.execute(&call).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.output.contains("file1.txt"));
        assert!(output.output.contains("file2.rs"));
        assert!(output.output.contains("subdir/"));
    }

    #[tokio::test]
    async fn test_list_directory_with_pattern() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("file1.txt"), "").unwrap();
        std::fs::write(tmp.path().join("file2.rs"), "").unwrap();

        let tool = ListDirectoryTool::permissive();
        let call = ToolCall {
            id: "test-1".to_string(),
            name: "glob".to_string(),
            arguments: serde_json::json!({
                "path": tmp.path().to_str().unwrap(),
                "pattern": "*.rs"
            })
            .to_string(),
        };

        let result = tool.execute(&call).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.output.contains("file1.txt"));
        assert!(output.output.contains("file2.rs"));
    }

    #[tokio::test]
    async fn test_search_files_content() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("has_pattern.txt"), "foo bar baz").unwrap();
        std::fs::write(tmp.path().join("no_pattern.txt"), "nothing here").unwrap();

        let tool = SearchFilesTool::permissive();
        let call = make_search_call(tmp.path().to_str().unwrap(), "bar");

        let result = tool.execute(&call).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.output.contains("has_pattern.txt"));
        assert!(!output.output.contains("no_pattern.txt"));
    }

    #[tokio::test]
    async fn test_tool_specs() {
        let read_tool = ReadFileTool::new();
        assert_eq!(read_tool.spec().name, "read");
        assert_eq!(read_tool.spec().category, ToolCategory::FileSystem);

        let write_tool = WriteFileTool::new();
        assert_eq!(write_tool.spec().name, "write");
        assert!(write_tool.spec().is_destructive);
        assert!(write_tool.spec().requires_confirmation);

        let list_tool = ListDirectoryTool::new();
        assert_eq!(list_tool.spec().name, "glob");

        let search_tool = SearchFilesTool::new();
        assert_eq!(search_tool.spec().name, "grep");
        assert_eq!(search_tool.spec().category, ToolCategory::Search);
    }
}
