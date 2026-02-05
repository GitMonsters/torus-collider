//! # Agent Context
//!
//! Maintains state across the agentic loop including conversation history,
//! tool call records, and accumulated metrics.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::types::{AgentMessage, ToolCallRecord};
use crate::providers::types::{Message, Role, Usage};

// =============================================================================
// CONTEXT CONFIGURATION
// =============================================================================

/// Configuration for context management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum messages to keep in history
    pub max_messages: usize,
    /// Maximum tokens to accumulate before warning
    pub max_tokens: usize,
    /// Maximum context window size (for overflow prevention)
    pub max_context_window: usize,
    /// Whether to include system message in context
    pub include_system: bool,
    /// Maximum tool output length before truncation
    pub max_tool_output_length: usize,
    /// Whether to track chain-of-thought separately
    pub track_thinking: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_messages: 100,
            max_tokens: 100_000,
            max_context_window: 128_000,
            include_system: true,
            max_tool_output_length: 10_000,
            track_thinking: true,
        }
    }
}

impl ContextConfig {
    /// Create a minimal context config (for testing)
    pub fn minimal() -> Self {
        Self {
            max_messages: 10,
            max_tokens: 10_000,
            max_context_window: 16_000,
            include_system: true,
            max_tool_output_length: 1_000,
            track_thinking: false,
        }
    }

    /// Create a large context config
    pub fn large() -> Self {
        Self {
            max_messages: 500,
            max_tokens: 500_000,
            max_context_window: 200_000,
            include_system: true,
            max_tool_output_length: 50_000,
            track_thinking: true,
        }
    }
}

// =============================================================================
// AGENT CONTEXT
// =============================================================================

/// Context for an agent run, tracking conversation and tool state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    /// Unique run ID
    pub run_id: String,
    /// System prompt (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Conversation history
    pub messages: Vec<AgentMessage>,
    /// Tool call records
    pub tool_calls: Vec<ToolCallRecord>,
    /// Current iteration (0-based)
    pub iteration: usize,
    /// Accumulated token usage
    pub total_usage: Usage,
    /// Configuration
    pub config: ContextConfig,
    /// Chain-of-thought entries (if tracking)
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub thinking: Vec<ThinkingEntry>,
    /// Custom metadata
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Start timestamp
    pub started_at_ms: u64,
}

/// A thinking/chain-of-thought entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingEntry {
    /// Iteration when thought occurred
    pub iteration: usize,
    /// The thought content
    pub content: String,
    /// Timestamp
    pub timestamp_ms: u64,
}

impl AgentContext {
    /// Create a new context with a unique run ID
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            system_prompt: None,
            messages: Vec::new(),
            tool_calls: Vec::new(),
            iteration: 0,
            total_usage: Usage::default(),
            config: ContextConfig::default(),
            thinking: Vec::new(),
            metadata: HashMap::new(),
            started_at_ms: Self::now_ms(),
        }
    }

    /// Create with a system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Create with custom config
    pub fn with_config(mut self, config: ContextConfig) -> Self {
        self.config = config;
        self
    }

    /// Generate a unique run ID
    pub fn generate_run_id() -> String {
        let timestamp = Self::now_ms();
        let random: u32 = rand_simple();
        format!("run-{}-{:08x}", timestamp, random)
    }

    // =========================================================================
    // MESSAGE MANAGEMENT
    // =========================================================================

    /// Add a user message
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages
            .push(AgentMessage::user(content, self.iteration));
        self.maybe_trim_messages();
    }

    /// Add an assistant message
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages
            .push(AgentMessage::assistant(content, self.iteration));
        self.maybe_trim_messages();
    }

    /// Add an assistant message with tool calls
    pub fn add_assistant_with_tools(
        &mut self,
        content: impl Into<String>,
        tool_calls: Vec<crate::providers::types::ToolCall>,
    ) {
        self.messages.push(AgentMessage::assistant_with_tools(
            content,
            tool_calls,
            self.iteration,
        ));
        self.maybe_trim_messages();
    }

    /// Add a tool result message
    pub fn add_tool_result(&mut self, tool_call_id: impl Into<String>, content: impl Into<String>) {
        let content_str = content.into();
        // Truncate if too long
        let truncated = if content_str.len() > self.config.max_tool_output_length {
            format!(
                "{}...[truncated, {} chars total]",
                &content_str[..self.config.max_tool_output_length],
                content_str.len()
            )
        } else {
            content_str
        };
        self.messages.push(AgentMessage::tool_result(
            tool_call_id,
            truncated,
            self.iteration,
        ));
        self.maybe_trim_messages();
    }

    /// Add a tool call record
    pub fn add_tool_call_record(&mut self, record: ToolCallRecord) {
        self.tool_calls.push(record);
    }

    /// Get messages for LLM (converted to provider format)
    pub fn get_messages_for_llm(&self) -> Vec<Message> {
        let mut messages = Vec::new();

        // Add system prompt if configured
        if self.config.include_system {
            if let Some(ref system) = self.system_prompt {
                messages.push(Message::system(system));
            }
        }

        // Add conversation messages
        for msg in &self.messages {
            messages.push(msg.to_provider_message());
        }

        messages
    }

    /// Trim messages if exceeding limit (keeps system + recent)
    fn maybe_trim_messages(&mut self) {
        if self.messages.len() > self.config.max_messages {
            let to_remove = self.messages.len() - self.config.max_messages;
            // Skip system messages (Role::System) when trimming
            let mut removed = 0;
            self.messages.retain(|msg| {
                if removed >= to_remove || msg.role == Role::System {
                    true
                } else {
                    removed += 1;
                    false
                }
            });
        }
    }

    // =========================================================================
    // ITERATION MANAGEMENT
    // =========================================================================

    /// Increment iteration counter
    pub fn next_iteration(&mut self) {
        self.iteration += 1;
    }

    /// Add token usage from a completion
    pub fn add_usage(&mut self, usage: Usage) {
        self.total_usage.prompt_tokens += usage.prompt_tokens;
        self.total_usage.completion_tokens += usage.completion_tokens;
        self.total_usage.total_tokens += usage.total_tokens;
    }

    /// Check if we've exceeded max tokens
    pub fn is_over_token_limit(&self) -> bool {
        self.total_usage.total_tokens > self.config.max_tokens
    }

    /// Estimate current context size (rough)
    pub fn estimated_context_size(&self) -> usize {
        // Rough estimate: 4 chars per token
        let char_count: usize = self.messages.iter().map(|m| m.content.len()).sum();
        if let Some(ref system) = self.system_prompt {
            (char_count + system.len()) / 4
        } else {
            char_count / 4
        }
    }

    /// Check if context is approaching overflow
    pub fn is_near_context_limit(&self) -> bool {
        self.estimated_context_size() > (self.config.max_context_window * 80 / 100)
    }

    // =========================================================================
    // THINKING/CHAIN-OF-THOUGHT
    // =========================================================================

    /// Add a thinking entry
    pub fn add_thinking(&mut self, content: impl Into<String>) {
        if self.config.track_thinking {
            self.thinking.push(ThinkingEntry {
                iteration: self.iteration,
                content: content.into(),
                timestamp_ms: Self::now_ms(),
            });
        }
    }

    /// Get thinking for current iteration
    pub fn current_thinking(&self) -> Vec<&ThinkingEntry> {
        self.thinking
            .iter()
            .filter(|t| t.iteration == self.iteration)
            .collect()
    }

    // =========================================================================
    // METADATA
    // =========================================================================

    /// Set metadata
    pub fn set_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Get elapsed time since start
    pub fn elapsed_ms(&self) -> u64 {
        Self::now_ms().saturating_sub(self.started_at_ms)
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================

    /// Get a summary of the context
    pub fn summary(&self) -> ContextSummary {
        ContextSummary {
            run_id: self.run_id.clone(),
            iterations: self.iteration,
            message_count: self.messages.len(),
            tool_call_count: self.tool_calls.len(),
            total_tokens: self.total_usage.total_tokens,
            elapsed_ms: self.elapsed_ms(),
            has_system_prompt: self.system_prompt.is_some(),
            thinking_count: self.thinking.len(),
        }
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// Simple pseudo-random number (not cryptographic)
fn rand_simple() -> u32 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(12345);
    // Simple xorshift
    let mut x = nanos;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

/// Summary of context state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSummary {
    /// Run ID
    pub run_id: String,
    /// Number of iterations
    pub iterations: usize,
    /// Number of messages
    pub message_count: usize,
    /// Number of tool calls
    pub tool_call_count: usize,
    /// Total tokens used
    pub total_tokens: usize,
    /// Elapsed time
    pub elapsed_ms: u64,
    /// Whether system prompt is set
    pub has_system_prompt: bool,
    /// Number of thinking entries
    pub thinking_count: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_new() {
        let ctx = AgentContext::new("run-123");
        assert_eq!(ctx.run_id, "run-123");
        assert_eq!(ctx.iteration, 0);
        assert!(ctx.messages.is_empty());
    }

    #[test]
    fn test_context_with_system_prompt() {
        let ctx = AgentContext::new("run-1").with_system_prompt("You are helpful.");
        assert_eq!(ctx.system_prompt, Some("You are helpful.".to_string()));
    }

    #[test]
    fn test_add_messages() {
        let mut ctx = AgentContext::new("run-1");
        ctx.add_user_message("Hello");
        ctx.add_assistant_message("Hi there!");
        assert_eq!(ctx.messages.len(), 2);
        assert_eq!(ctx.messages[0].role, Role::User);
        assert_eq!(ctx.messages[1].role, Role::Assistant);
    }

    #[test]
    fn test_add_tool_result() {
        let mut ctx = AgentContext::new("run-1");
        ctx.add_tool_result("call-1", "tool output");
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.messages[0].role, Role::Tool);
        assert_eq!(ctx.messages[0].tool_call_id, Some("call-1".to_string()));
    }

    #[test]
    fn test_tool_result_truncation() {
        let mut ctx = AgentContext::new("run-1").with_config(ContextConfig {
            max_tool_output_length: 10,
            ..ContextConfig::default()
        });
        ctx.add_tool_result(
            "call-1",
            "this is a very long output that should be truncated",
        );
        assert!(ctx.messages[0].content.contains("truncated"));
    }

    #[test]
    fn test_get_messages_for_llm() {
        let mut ctx = AgentContext::new("run-1").with_system_prompt("System prompt");
        ctx.add_user_message("Hello");
        ctx.add_assistant_message("Hi!");

        let messages = ctx.get_messages_for_llm();
        assert_eq!(messages.len(), 3); // system + user + assistant
        assert_eq!(messages[0].role, Role::System);
    }

    #[test]
    fn test_next_iteration() {
        let mut ctx = AgentContext::new("run-1");
        assert_eq!(ctx.iteration, 0);
        ctx.next_iteration();
        assert_eq!(ctx.iteration, 1);
        ctx.next_iteration();
        assert_eq!(ctx.iteration, 2);
    }

    #[test]
    fn test_add_usage() {
        let mut ctx = AgentContext::new("run-1");
        ctx.add_usage(Usage::new(100, 50));
        ctx.add_usage(Usage::new(100, 50));
        assert_eq!(ctx.total_usage.prompt_tokens, 200);
        assert_eq!(ctx.total_usage.completion_tokens, 100);
        assert_eq!(ctx.total_usage.total_tokens, 300);
    }

    #[test]
    fn test_is_over_token_limit() {
        let mut ctx = AgentContext::new("run-1").with_config(ContextConfig {
            max_tokens: 1000,
            ..ContextConfig::default()
        });
        assert!(!ctx.is_over_token_limit());
        ctx.add_usage(Usage::new(500, 600)); // 1100 total
        assert!(ctx.is_over_token_limit());
    }

    #[test]
    fn test_add_thinking() {
        let mut ctx = AgentContext::new("run-1");
        ctx.add_thinking("I should use the read tool");
        assert_eq!(ctx.thinking.len(), 1);
        assert_eq!(ctx.thinking[0].content, "I should use the read tool");
    }

    #[test]
    fn test_thinking_disabled() {
        let mut ctx = AgentContext::new("run-1").with_config(ContextConfig {
            track_thinking: false,
            ..ContextConfig::default()
        });
        ctx.add_thinking("This should not be recorded");
        assert!(ctx.thinking.is_empty());
    }

    #[test]
    fn test_context_summary() {
        let mut ctx = AgentContext::new("run-1").with_system_prompt("System");
        ctx.add_user_message("Hello");
        ctx.next_iteration();
        ctx.add_thinking("Thinking...");

        let summary = ctx.summary();
        assert_eq!(summary.run_id, "run-1");
        assert_eq!(summary.iterations, 1);
        assert_eq!(summary.message_count, 1);
        assert!(summary.has_system_prompt);
        assert_eq!(summary.thinking_count, 1);
    }

    #[test]
    fn test_generate_run_id() {
        let id1 = AgentContext::generate_run_id();
        let id2 = AgentContext::generate_run_id();
        assert!(id1.starts_with("run-"));
        // IDs should be unique (with high probability)
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_metadata() {
        let mut ctx = AgentContext::new("run-1");
        ctx.set_metadata("key1", serde_json::json!("value1"));
        assert_eq!(ctx.get_metadata("key1"), Some(&serde_json::json!("value1")));
        assert_eq!(ctx.get_metadata("nonexistent"), None);
    }
}
