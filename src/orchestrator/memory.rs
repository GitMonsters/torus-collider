//! Memory Manager for hierarchical message storage.
//!
//! Inspired by Confucius' CfMemoryManager, this provides:
//! - Hierarchical memory scopes (Conversation, Task, Persistent)
//! - Thread-safe message storage with ordering
//! - Parent-child memory relationships for context inheritance
//! - Optional SQLite persistence for long-term storage

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::{OrchestratorError, OrchestratorResult};

// =============================================================================
// MESSAGE TYPES
// =============================================================================

/// Type of message in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageType {
    /// System instructions
    System,
    /// User input
    User,
    /// AI assistant response
    Assistant,
    /// Tool call request
    ToolCall,
    /// Tool result
    ToolResult,
    /// Internal thought/reasoning
    Thought,
    /// Error message
    Error,
}

impl MessageType {
    /// Convert to provider Role
    pub fn to_role(&self) -> crate::providers::types::Role {
        match self {
            Self::System => crate::providers::types::Role::System,
            Self::User => crate::providers::types::Role::User,
            Self::Assistant => crate::providers::types::Role::Assistant,
            Self::ToolCall => crate::providers::types::Role::Assistant,
            Self::ToolResult => crate::providers::types::Role::Tool,
            Self::Thought => crate::providers::types::Role::Assistant,
            Self::Error => crate::providers::types::Role::System,
        }
    }
}

/// Visibility scope for retrieving history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryVisibility {
    /// All messages in the current session
    Session,
    /// Messages from the current entry/task only
    Entry,
    /// Messages from the current analect scope only
    Analect,
    /// Messages from a specific runnable/component
    Runnable,
}

/// Memory scope levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryScope {
    /// Conversation-level memory (cleared between sessions)
    Conversation,
    /// Task-level memory (persists for task duration)
    Task,
    /// Persistent memory (stored in database)
    Persistent,
}

// Global message counter for ordering
static MESSAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_message_id() -> u64 {
    MESSAGE_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// A message stored in memory.
#[derive(Debug, Clone)]
pub struct MemoryMessage {
    /// Unique sequence ID for ordering
    pub sequence_id: u64,
    /// Message type
    pub msg_type: MessageType,
    /// Message content
    pub content: String,
    /// Entry name (task/component that created this)
    pub entry_name: Option<String>,
    /// Runnable name (specific runnable that created this)
    pub runnable_name: Option<String>,
    /// Hierarchical path in the analect tree
    pub path: Vec<String>,
    /// Timestamp (millis since epoch)
    pub timestamp: u64,
    /// Optional tool call ID (for tool messages)
    pub tool_call_id: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl MemoryMessage {
    /// Create a new message.
    pub fn new(msg_type: MessageType, content: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            sequence_id: next_message_id(),
            msg_type,
            content: content.into(),
            entry_name: None,
            runnable_name: None,
            path: Vec::new(),
            timestamp: now,
            tool_call_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageType::System, content)
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageType::User, content)
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageType::Assistant, content)
    }

    /// Create a thought message (internal reasoning).
    pub fn thought(content: impl Into<String>) -> Self {
        Self::new(MessageType::Thought, content)
    }

    /// Create a tool result message.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        let mut msg = Self::new(MessageType::ToolResult, content);
        msg.tool_call_id = Some(tool_call_id.into());
        msg
    }

    /// Set the entry name.
    pub fn with_entry(mut self, entry: impl Into<String>) -> Self {
        self.entry_name = Some(entry.into());
        self
    }

    /// Set the runnable name.
    pub fn with_runnable(mut self, runnable: impl Into<String>) -> Self {
        self.runnable_name = Some(runnable.into());
        self
    }

    /// Set the path.
    pub fn with_path(mut self, path: Vec<String>) -> Self {
        self.path = path;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Convert to provider Message.
    pub fn to_provider_message(&self) -> crate::providers::types::Message {
        let mut msg = crate::providers::types::Message::new(self.msg_type.to_role(), &self.content);

        if let Some(ref id) = self.tool_call_id {
            msg.tool_call_id = Some(id.clone());
        }

        for (k, v) in &self.metadata {
            msg = msg.with_metadata(k, v);
        }

        msg
    }
}

impl PartialOrd for MemoryMessage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MemoryMessage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sequence_id.cmp(&other.sequence_id)
    }
}

impl PartialEq for MemoryMessage {
    fn eq(&self, other: &Self) -> bool {
        self.sequence_id == other.sequence_id
    }
}

impl Eq for MemoryMessage {}

// =============================================================================
// MEMORY STORAGE
// =============================================================================

/// In-memory message storage.
#[derive(Debug, Default)]
struct MemoryStorage {
    messages: Vec<MemoryMessage>,
}

impl MemoryStorage {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    fn add(&mut self, message: MemoryMessage) {
        self.messages.push(message);
        self.messages.sort();
    }

    fn add_all(&mut self, messages: impl IntoIterator<Item = MemoryMessage>) {
        self.messages.extend(messages);
        self.messages.sort();
    }

    fn clear(&mut self) {
        self.messages.clear();
    }

    fn len(&self) -> usize {
        self.messages.len()
    }

    fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    fn filter_by_entry(&self, entry_name: &str) -> Vec<MemoryMessage> {
        self.messages
            .iter()
            .filter(|m| m.entry_name.as_deref() == Some(entry_name))
            .cloned()
            .collect()
    }

    fn filter_by_runnable(&self, runnable_name: &str) -> Vec<MemoryMessage> {
        self.messages
            .iter()
            .filter(|m| m.runnable_name.as_deref() == Some(runnable_name))
            .cloned()
            .collect()
    }

    fn filter_by_type(&self, types: &[MessageType]) -> Vec<MemoryMessage> {
        self.messages
            .iter()
            .filter(|m| types.contains(&m.msg_type))
            .cloned()
            .collect()
    }
}

// =============================================================================
// MEMORY CONFIG
// =============================================================================

/// Configuration for memory manager.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum messages to retain (0 = unlimited)
    pub max_messages: usize,
    /// Maximum tokens to retain (0 = unlimited)
    pub max_tokens: usize,
    /// Whether to enable summarization when limits are reached
    pub enable_summarization: bool,
    /// Message types to include in child contexts
    pub child_included_types: Vec<MessageType>,
    /// Path to SQLite database for persistent storage (None = in-memory only)
    pub db_path: Option<String>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_messages: 1000,
            max_tokens: 100_000,
            enable_summarization: true,
            child_included_types: vec![
                MessageType::System,
                MessageType::User,
                MessageType::Assistant,
            ],
            db_path: None,
        }
    }
}

// =============================================================================
// MEMORY STATISTICS
// =============================================================================

/// Statistics about memory usage.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total messages in storage
    pub total_messages: usize,
    /// Messages by type
    pub by_type: HashMap<MessageType, usize>,
    /// Estimated token count
    pub estimated_tokens: usize,
    /// Number of unique entries
    pub unique_entries: usize,
    /// Memory depth (nesting level)
    pub depth: usize,
}

// =============================================================================
// MEMORY MANAGER
// =============================================================================

/// Manages hierarchical message memory with scopes.
///
/// Based on Confucius' CfMemoryManager pattern, providing:
/// - Parent-child memory relationships
/// - Scope-based message filtering
/// - Thread-safe operations
/// - Optional persistence
pub struct MemoryManager {
    /// Current memory storage
    storage: Arc<RwLock<MemoryStorage>>,
    /// Parent memory manager (for scope inheritance)
    parent: Option<Arc<MemoryManager>>,
    /// Current entry name
    entry_name: Option<String>,
    /// Current runnable name
    runnable_name: Option<String>,
    /// Hierarchical path
    path: Vec<String>,
    /// Configuration
    config: MemoryConfig,
}

impl MemoryManager {
    /// Create a new root memory manager.
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            storage: Arc::new(RwLock::new(MemoryStorage::new())),
            parent: None,
            entry_name: None,
            runnable_name: None,
            path: Vec::new(),
            config,
        }
    }

    /// Create a new memory manager with default config.
    pub fn with_defaults() -> Self {
        Self::new(MemoryConfig::default())
    }

    /// Create a child memory manager for a nested scope.
    pub fn child(&self, runnable_name: Option<String>) -> Self {
        let mut child_path = self.path.clone();
        if let Some(ref name) = runnable_name {
            child_path.push(name.clone());
        }

        Self {
            storage: Arc::new(RwLock::new(MemoryStorage::new())),
            parent: Some(Arc::new(Self {
                storage: Arc::clone(&self.storage),
                parent: self.parent.clone(),
                entry_name: self.entry_name.clone(),
                runnable_name: self.runnable_name.clone(),
                path: self.path.clone(),
                config: self.config.clone(),
            })),
            entry_name: self.entry_name.clone(),
            runnable_name,
            path: child_path,
            config: self.config.clone(),
        }
    }

    /// Set the entry name for this scope.
    pub fn with_entry(mut self, entry: impl Into<String>) -> Self {
        self.entry_name = Some(entry.into());
        self
    }

    /// Add a message to memory.
    pub fn add_message(&self, mut message: MemoryMessage) {
        // Enrich message with context
        if message.entry_name.is_none() {
            message.entry_name = self.entry_name.clone();
        }
        if message.runnable_name.is_none() {
            message.runnable_name = self.runnable_name.clone();
        }
        if message.path.is_empty() {
            message.path = self.path.clone();
        }

        if let Ok(mut storage) = self.storage.write() {
            storage.add(message);
        }
    }

    /// Add multiple messages to memory.
    pub fn add_messages(&self, messages: Vec<MemoryMessage>) {
        let enriched: Vec<_> = messages
            .into_iter()
            .map(|mut m| {
                if m.entry_name.is_none() {
                    m.entry_name = self.entry_name.clone();
                }
                if m.runnable_name.is_none() {
                    m.runnable_name = self.runnable_name.clone();
                }
                if m.path.is_empty() {
                    m.path = self.path.clone();
                }
                m
            })
            .collect();

        if let Ok(mut storage) = self.storage.write() {
            storage.add_all(enriched);
        }
    }

    /// Get messages from the current session (including parent memories).
    pub fn get_session_memory(&self) -> Vec<MemoryMessage> {
        self.collect_messages_recursive()
    }

    /// Get messages from the current entry only.
    pub fn get_entry_memory(&self) -> Vec<MemoryMessage> {
        let entry = match &self.entry_name {
            Some(e) => e.clone(),
            None => return Vec::new(),
        };

        self.collect_entry_messages(&entry)
    }

    /// Get messages from the current analect scope only.
    pub fn get_analect_memory(&self) -> Vec<MemoryMessage> {
        if let Ok(storage) = self.storage.read() {
            storage.messages.clone()
        } else {
            Vec::new()
        }
    }

    /// Get messages by visibility scope.
    pub fn get_memory(&self, visibility: HistoryVisibility) -> Vec<MemoryMessage> {
        match visibility {
            HistoryVisibility::Session => self.get_session_memory(),
            HistoryVisibility::Entry => self.get_entry_memory(),
            HistoryVisibility::Analect => self.get_analect_memory(),
            HistoryVisibility::Runnable => {
                if let Some(ref name) = self.runnable_name {
                    self.collect_runnable_messages(name)
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Get messages as provider Messages.
    pub fn to_provider_messages(
        &self,
        visibility: HistoryVisibility,
    ) -> Vec<crate::providers::types::Message> {
        self.get_memory(visibility)
            .into_iter()
            .filter(|m| {
                // Filter out internal thoughts by default
                m.msg_type != MessageType::Thought
            })
            .map(|m| m.to_provider_message())
            .collect()
    }

    /// Clear all messages in current scope.
    pub fn clear(&self) {
        if let Ok(mut storage) = self.storage.write() {
            storage.clear();
        }
    }

    /// Get memory statistics.
    pub fn stats(&self) -> MemoryStats {
        let messages = self.get_session_memory();

        let mut by_type = HashMap::new();
        let mut entries = std::collections::HashSet::new();
        let mut estimated_tokens = 0;

        for msg in &messages {
            *by_type.entry(msg.msg_type).or_insert(0) += 1;
            if let Some(ref entry) = msg.entry_name {
                entries.insert(entry.clone());
            }
            // Rough token estimate: 4 chars per token
            estimated_tokens += msg.content.len() / 4;
        }

        MemoryStats {
            total_messages: messages.len(),
            by_type,
            estimated_tokens,
            unique_entries: entries.len(),
            depth: self.path.len(),
        }
    }

    /// Check if memory is empty.
    pub fn is_empty(&self) -> bool {
        if let Ok(storage) = self.storage.read() {
            storage.is_empty()
        } else {
            true
        }
    }

    /// Consolidate child memory into this memory.
    ///
    /// This merges messages from a child memory manager, filtering by
    /// the configured included message types.
    pub fn consolidate(&self, child: &MemoryManager) {
        let included_types = &self.config.child_included_types;

        if let Ok(child_storage) = child.storage.read() {
            let filtered: Vec<_> = child_storage
                .filter_by_type(included_types)
                .into_iter()
                .collect();

            if let Ok(mut storage) = self.storage.write() {
                storage.add_all(filtered);
            }
        }
    }

    // Helper: Recursively collect all messages from this and parent memories
    fn collect_messages_recursive(&self) -> Vec<MemoryMessage> {
        let mut messages = if let Some(ref parent) = self.parent {
            parent.collect_messages_recursive()
        } else {
            Vec::new()
        };

        if let Ok(storage) = self.storage.read() {
            messages.extend(storage.messages.iter().cloned());
        }

        messages.sort();
        messages
    }

    // Helper: Collect messages for a specific entry
    fn collect_entry_messages(&self, entry_name: &str) -> Vec<MemoryMessage> {
        let mut messages = if let Some(ref parent) = self.parent {
            parent.collect_entry_messages(entry_name)
        } else {
            Vec::new()
        };

        if let Ok(storage) = self.storage.read() {
            messages.extend(storage.filter_by_entry(entry_name));
        }

        messages.sort();
        messages
    }

    // Helper: Collect messages for a specific runnable
    fn collect_runnable_messages(&self, runnable_name: &str) -> Vec<MemoryMessage> {
        let mut messages = if let Some(ref parent) = self.parent {
            parent.collect_runnable_messages(runnable_name)
        } else {
            Vec::new()
        };

        if let Ok(storage) = self.storage.read() {
            messages.extend(storage.filter_by_runnable(runnable_name));
        }

        messages.sort();
        messages
    }
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("entry_name", &self.entry_name)
            .field("runnable_name", &self.runnable_name)
            .field("path", &self.path)
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

impl Clone for MemoryManager {
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            parent: self.parent.clone(),
            entry_name: self.entry_name.clone(),
            runnable_name: self.runnable_name.clone(),
            path: self.path.clone(),
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_message_creation() {
        let msg = MemoryMessage::user("Hello");
        assert_eq!(msg.msg_type, MessageType::User);
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_memory_message_ordering() {
        let msg1 = MemoryMessage::user("First");
        let msg2 = MemoryMessage::user("Second");
        assert!(msg1 < msg2);
    }

    #[test]
    fn test_memory_manager_add_retrieve() {
        let manager = MemoryManager::with_defaults();

        manager.add_message(MemoryMessage::system("You are helpful"));
        manager.add_message(MemoryMessage::user("Hi"));
        manager.add_message(MemoryMessage::assistant("Hello!"));

        let messages = manager.get_session_memory();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].msg_type, MessageType::System);
        assert_eq!(messages[2].msg_type, MessageType::Assistant);
    }

    #[test]
    fn test_memory_manager_child() {
        let parent = MemoryManager::with_defaults().with_entry("main_task");

        parent.add_message(MemoryMessage::system("Parent context"));

        let child = parent.child(Some("subtask".to_string()));
        child.add_message(MemoryMessage::user("Child message"));

        // Child should see both parent and own messages
        let child_messages = child.get_session_memory();
        assert_eq!(child_messages.len(), 2);

        // Parent should only see its own messages
        let parent_messages = parent.get_session_memory();
        assert_eq!(parent_messages.len(), 1);
    }

    #[test]
    fn test_memory_manager_consolidation() {
        let parent = MemoryManager::with_defaults();
        let child = parent.child(Some("child".to_string()));

        child.add_message(MemoryMessage::user("User in child"));
        child.add_message(MemoryMessage::thought("Internal thought")); // Should be filtered
        child.add_message(MemoryMessage::assistant("Response"));

        parent.consolidate(&child);

        let parent_messages = parent.get_session_memory();
        // Thought messages are not in default included types
        assert_eq!(parent_messages.len(), 2);
    }

    #[test]
    fn test_memory_stats() {
        let manager = MemoryManager::with_defaults();

        manager.add_message(MemoryMessage::user("One"));
        manager.add_message(MemoryMessage::user("Two"));
        manager.add_message(MemoryMessage::assistant("Response"));

        let stats = manager.stats();
        assert_eq!(stats.total_messages, 3);
        assert_eq!(*stats.by_type.get(&MessageType::User).unwrap(), 2);
        assert_eq!(*stats.by_type.get(&MessageType::Assistant).unwrap(), 1);
    }

    #[test]
    fn test_to_provider_messages() {
        let manager = MemoryManager::with_defaults();

        manager.add_message(MemoryMessage::system("Be helpful"));
        manager.add_message(MemoryMessage::user("Hello"));
        manager.add_message(MemoryMessage::thought("Thinking...")); // Filtered out
        manager.add_message(MemoryMessage::assistant("Hi there!"));

        let provider_messages = manager.to_provider_messages(HistoryVisibility::Session);

        // Thoughts are filtered out
        assert_eq!(provider_messages.len(), 3);
        assert_eq!(
            provider_messages[0].role,
            crate::providers::types::Role::System
        );
        assert_eq!(
            provider_messages[1].role,
            crate::providers::types::Role::User
        );
        assert_eq!(
            provider_messages[2].role,
            crate::providers::types::Role::Assistant
        );
    }

    #[test]
    fn test_entry_filtering() {
        let manager = MemoryManager::with_defaults();

        manager.add_message(MemoryMessage::user("Global message"));
        manager.add_message(MemoryMessage::user("Task A").with_entry("task_a"));
        manager.add_message(MemoryMessage::user("Task B").with_entry("task_b"));

        let manager_a = manager.clone().with_entry("task_a");
        let entry_messages = manager_a.get_entry_memory();

        assert_eq!(entry_messages.len(), 1);
        assert_eq!(entry_messages[0].content, "Task A");
    }
}
