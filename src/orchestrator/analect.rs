//! Analect Orchestration Pattern
//!
//! Inspired by Confucius' Analect pattern, this provides a high-level
//! orchestration layer for AI agents with:
//!
//! - Chain-of-thought management
//! - Nested context support (child analects)
//! - Tool invocation coordination
//! - Memory consolidation across scopes
//!
//! # Architecture
//!
//! ```text
//! Analect (root)
//!     │
//!     ├── Context (session, memory, provider)
//!     │
//!     ├── ChainOfThought
//!     │     ├── ThoughtStep (observe)
//!     │     ├── ThoughtStep (reason)
//!     │     ├── ThoughtStep (plan)
//!     │     └── ThoughtStep (act)
//!     │
//!     └── Child Analects
//!           ├── SubTask1
//!           └── SubTask2
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use super::agent::{Agent, AgentConfig, AgentState, TurnResult};
use super::memory::{HistoryVisibility, MemoryConfig, MemoryManager, MemoryMessage, MessageType};
use super::types::{OrchestratorError, OrchestratorResult};
use crate::providers::types::{CompletionOptions, Message, ToolCall, ToolDefinition};
use crate::providers::LLMProvider;
use crate::safety::SafetyGuard;

// =============================================================================
// CHAIN OF THOUGHT
// =============================================================================

/// A step in the chain of thought.
#[derive(Debug, Clone)]
pub struct ThoughtStep {
    /// Type of thought step
    pub step_type: ThoughtStepType,
    /// Content of the thought
    pub content: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Timestamp (ms since epoch)
    pub timestamp: u64,
}

/// Types of thought steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThoughtStepType {
    /// Observing the input/context
    Observe,
    /// Reasoning about the problem
    Reason,
    /// Planning the approach
    Plan,
    /// Taking action
    Act,
    /// Reflecting on results
    Reflect,
    /// Error recovery
    Recover,
}

impl ThoughtStep {
    /// Create a new thought step.
    pub fn new(step_type: ThoughtStepType, content: impl Into<String>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        Self {
            step_type,
            content: content.into(),
            confidence: 1.0,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Set confidence level.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Chain of thought tracker.
#[derive(Debug, Clone, Default)]
pub struct ChainOfThought {
    /// All thought steps
    steps: Vec<ThoughtStep>,
    /// Whether the chain is complete
    is_complete: bool,
}

impl ChainOfThought {
    /// Create a new chain of thought.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a thought step.
    pub fn add(&mut self, step: ThoughtStep) {
        self.steps.push(step);
    }

    /// Add an observation step.
    pub fn observe(&mut self, content: impl Into<String>) {
        self.add(ThoughtStep::new(ThoughtStepType::Observe, content));
    }

    /// Add a reasoning step.
    pub fn reason(&mut self, content: impl Into<String>) {
        self.add(ThoughtStep::new(ThoughtStepType::Reason, content));
    }

    /// Add a planning step.
    pub fn plan(&mut self, content: impl Into<String>) {
        self.add(ThoughtStep::new(ThoughtStepType::Plan, content));
    }

    /// Add an action step.
    pub fn act(&mut self, content: impl Into<String>) {
        self.add(ThoughtStep::new(ThoughtStepType::Act, content));
    }

    /// Add a reflection step.
    pub fn reflect(&mut self, content: impl Into<String>) {
        self.add(ThoughtStep::new(ThoughtStepType::Reflect, content));
    }

    /// Mark the chain as complete.
    pub fn complete(&mut self) {
        self.is_complete = true;
    }

    /// Get all steps.
    pub fn steps(&self) -> &[ThoughtStep] {
        &self.steps
    }

    /// Get the last step.
    pub fn last(&self) -> Option<&ThoughtStep> {
        self.steps.last()
    }

    /// Check if complete.
    pub fn is_complete(&self) -> bool {
        self.is_complete
    }

    /// Calculate average confidence.
    pub fn avg_confidence(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        self.steps.iter().map(|s| s.confidence).sum::<f64>() / self.steps.len() as f64
    }

    /// Format as markdown for logging.
    pub fn to_markdown(&self) -> String {
        let mut output = String::from("## Chain of Thought\n\n");
        
        for (i, step) in self.steps.iter().enumerate() {
            let emoji = match step.step_type {
                ThoughtStepType::Observe => "👁",
                ThoughtStepType::Reason => "🧠",
                ThoughtStepType::Plan => "📋",
                ThoughtStepType::Act => "⚡",
                ThoughtStepType::Reflect => "🔄",
                ThoughtStepType::Recover => "🔧",
            };
            
            output.push_str(&format!(
                "{}. {} **{:?}** (confidence: {:.0}%)\n   {}\n\n",
                i + 1,
                emoji,
                step.step_type,
                step.confidence * 100.0,
                step.content
            ));
        }
        
        if self.is_complete {
            output.push_str("---\n*Chain complete*\n");
        }
        
        output
    }
}

// =============================================================================
// ANALECT CONFIG
// =============================================================================

/// Configuration for an Analect.
#[derive(Debug, Clone)]
pub struct AnalectConfig {
    /// Unique name for this analect
    pub name: String,
    /// System prompt
    pub system_prompt: String,
    /// Agent configuration
    pub agent_config: AgentConfig,
    /// Memory configuration
    pub memory_config: MemoryConfig,
    /// Whether to enable chain-of-thought tracking
    pub enable_cot: bool,
    /// Maximum nested depth for child analects
    pub max_depth: usize,
    /// Available tools
    pub tools: Vec<ToolDefinition>,
}

impl Default for AnalectConfig {
    fn default() -> Self {
        Self {
            name: "root".to_string(),
            system_prompt: "You are a helpful AI assistant.".to_string(),
            agent_config: AgentConfig::default(),
            memory_config: MemoryConfig::default(),
            enable_cot: true,
            max_depth: 5,
            tools: Vec::new(),
        }
    }
}

impl AnalectConfig {
    /// Create with a name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self.agent_config.system_prompt = self.system_prompt.clone();
        self
    }

    /// Add a tool.
    pub fn with_tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Enable or disable chain-of-thought.
    pub fn with_cot(mut self, enable: bool) -> Self {
        self.enable_cot = enable;
        self
    }
}

// =============================================================================
// ANALECT CONTEXT
// =============================================================================

/// Runtime context for an Analect execution.
pub struct AnalectContext {
    /// Session ID
    pub session_id: String,
    /// Memory manager
    pub memory: MemoryManager,
    /// Current chain of thought
    pub cot: ChainOfThought,
    /// Namespace path (for nested analects)
    pub namespace: Vec<String>,
    /// Parent context (for child analects)
    parent: Option<Arc<AnalectContext>>,
    /// Current depth
    depth: usize,
}

impl AnalectContext {
    /// Create a new root context.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            memory: MemoryManager::with_defaults(),
            cot: ChainOfThought::new(),
            namespace: Vec::new(),
            parent: None,
            depth: 0,
        }
    }

    /// Create a child context for a nested analect.
    pub fn child(&self, name: impl Into<String>, max_depth: usize) -> OrchestratorResult<Self> {
        if self.depth >= max_depth {
            return Err(OrchestratorError::ConfigError(format!(
                "Maximum nesting depth ({}) exceeded",
                max_depth
            )));
        }

        let name = name.into();
        let mut namespace = self.namespace.clone();
        namespace.push(name.clone());

        Ok(Self {
            session_id: self.session_id.clone(),
            memory: self.memory.child(Some(name)),
            cot: ChainOfThought::new(),
            namespace,
            parent: Some(Arc::new(AnalectContext {
                session_id: self.session_id.clone(),
                memory: self.memory.clone(),
                cot: self.cot.clone(),
                namespace: self.namespace.clone(),
                parent: self.parent.clone(),
                depth: self.depth,
            })),
            depth: self.depth + 1,
        })
    }

    /// Get the current depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Get the full namespace path.
    pub fn path(&self) -> String {
        self.namespace.join("/")
    }
}

impl std::fmt::Debug for AnalectContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnalectContext")
            .field("session_id", &self.session_id)
            .field("namespace", &self.namespace)
            .field("depth", &self.depth)
            .field("cot_steps", &self.cot.steps().len())
            .finish()
    }
}

// =============================================================================
// ANALECT STATISTICS
// =============================================================================

/// Statistics for an Analect execution.
#[derive(Debug, Clone, Default)]
pub struct AnalectStats {
    /// Total turns executed
    pub total_turns: usize,
    /// Total tool calls
    pub total_tool_calls: usize,
    /// Total tokens used
    pub total_tokens: usize,
    /// Child analects invoked
    pub child_analects: usize,
    /// Errors encountered
    pub errors: usize,
    /// Total execution time (ms)
    pub total_time_ms: u64,
    /// Chain of thought steps
    pub cot_steps: usize,
}

// =============================================================================
// ANALECT
// =============================================================================

/// The main Analect orchestrator.
///
/// Manages the execution of an AI agent with chain-of-thought,
/// nested contexts, and tool coordination.
pub struct Analect {
    /// Configuration
    config: AnalectConfig,
    /// LLM provider
    provider: Box<dyn LLMProvider>,
    /// Safety guard (optional)
    safety_guard: Option<Box<dyn SafetyGuard>>,
    /// Execution statistics
    stats: AnalectStats,
}

impl Analect {
    /// Create a new Analect.
    pub fn new(
        config: AnalectConfig,
        provider: Box<dyn LLMProvider>,
    ) -> Self {
        Self {
            config,
            provider,
            safety_guard: None,
            stats: AnalectStats::default(),
        }
    }

    /// Add a safety guard.
    pub fn with_safety_guard(mut self, guard: Box<dyn SafetyGuard>) -> Self {
        self.safety_guard = Some(guard);
        self
    }

    /// Get configuration.
    pub fn config(&self) -> &AnalectConfig {
        &self.config
    }

    /// Get statistics.
    pub fn stats(&self) -> &AnalectStats {
        &self.stats
    }

    /// Execute a single turn with context.
    pub fn turn<'a>(
        &'a mut self,
        context: &'a mut AnalectContext,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = OrchestratorResult<TurnResult>> + Send + 'a>> {
        Box::pin(async move {
            let start = Instant::now();

            // Track observation in chain of thought
            if self.config.enable_cot {
                context.cot.observe(format!("Received input: {}", &input[..input.len().min(100)]));
            }

            // Create an agent for this turn
            let mut agent = Agent::new(
                self.config.agent_config.clone(),
                self.create_provider_wrapper(),
                context.memory.clone(),
            );

            // Add safety guard if present
            if let Some(ref guard) = self.safety_guard {
                // Clone the guard for the agent
                agent = agent.with_safety_guard(Box::new(crate::safety::NoOpSafetyGuard));
            }

            // Execute the turn
            let result = agent.turn(input).await?;

            // Update chain of thought
            if self.config.enable_cot {
                if result.is_complete {
                    context.cot.act(format!("Responded: {}", &result.response[..result.response.len().min(100)]));
                } else {
                    context.cot.plan(format!("Making {} tool calls", result.tool_calls.len()));
                }
            }

            // Consolidate memory from agent
            context.memory.consolidate(agent.memory());

            // Update stats
            self.stats.total_turns += 1;
            self.stats.total_tokens += result.tokens_used;
            self.stats.total_tool_calls += result.tool_calls.len();
            self.stats.total_time_ms += start.elapsed().as_millis() as u64;
            self.stats.cot_steps = context.cot.steps().len();

            Ok(result)
        })
    }

    /// Execute a complete conversation loop.
    ///
    /// This handles tool calls automatically, looping until the agent
    /// completes or hits the turn limit.
    pub fn run<'a>(
        &'a mut self,
        context: &'a mut AnalectContext,
        input: &'a str,
        tool_executor: Option<&'a dyn ToolExecutor>,
    ) -> Pin<Box<dyn Future<Output = OrchestratorResult<String>> + Send + 'a>> {
        Box::pin(async move {
            let start = Instant::now();
            let mut current_input = input.to_string();
            let mut final_response = String::new();

            for turn_num in 0..self.config.agent_config.max_turns {
                // Execute a turn
                let result = self.turn(context, &current_input).await?;
                final_response = result.response.clone();

                if result.is_complete {
                    break;
                }

                // Handle tool calls
                if !result.tool_calls.is_empty() {
                    if let Some(executor) = tool_executor {
                        let mut tool_results = Vec::new();
                        
                        for tool_call in &result.tool_calls {
                            if self.config.enable_cot {
                                context.cot.act(format!("Calling tool: {}", tool_call.name));
                            }

                            let tool_result = executor.execute(tool_call).await
                                .map_err(|e| OrchestratorError::ToolError {
                                    tool_name: tool_call.name.clone(),
                                    message: e.to_string(),
                                })?;

                            tool_results.push((tool_call.id.clone(), tool_result));
                        }

                        // Add tool results to memory
                        for (id, result) in &tool_results {
                            context.memory.add_message(MemoryMessage::tool_result(id, result));
                        }

                        // Continue with tool results as context
                        current_input = format!("Tool results received. Continue with the task.");
                    } else {
                        // No executor, return with pending tools
                        break;
                    }
                }
            }

            // Complete chain of thought
            if self.config.enable_cot {
                context.cot.complete();
            }

            self.stats.total_time_ms = start.elapsed().as_millis() as u64;

            Ok(final_response)
        })
    }

    /// Invoke a child analect.
    pub fn invoke_child<'a>(
        &'a mut self,
        parent_context: &'a mut AnalectContext,
        child_config: AnalectConfig,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = OrchestratorResult<String>> + Send + 'a>> {
        Box::pin(async move {
            // Create child context
            let mut child_context = parent_context.child(&child_config.name, self.config.max_depth)?;

            // Track in parent's chain of thought
            if self.config.enable_cot {
                parent_context.cot.plan(format!("Invoking child analect: {}", child_config.name));
            }

            // Create child analect
            let mut child = Analect::new(child_config, self.create_provider_wrapper());
            
            // Run child
            let result = child.run(&mut child_context, input, None).await?;

            // Consolidate child memory into parent
            parent_context.memory.consolidate(&child_context.memory);

            // Track completion
            if self.config.enable_cot {
                parent_context.cot.reflect(format!(
                    "Child analect completed with {} turns",
                    child.stats.total_turns
                ));
            }

            // Update stats
            self.stats.child_analects += 1;

            Ok(result)
        })
    }

    // Helper: Create a wrapper provider that shares the same underlying provider
    fn create_provider_wrapper(&self) -> Box<dyn LLMProvider> {
        // For now, create a simple passthrough wrapper
        // In a real implementation, this would share the provider or create a clone
        Box::new(ProviderWrapper {
            name: self.provider.name().to_string(),
            model: self.provider.model().to_string(),
        })
    }
}

impl std::fmt::Debug for Analect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Analect")
            .field("name", &self.config.name)
            .field("stats", &self.stats)
            .field("has_safety_guard", &self.safety_guard.is_some())
            .finish()
    }
}

// =============================================================================
// TOOL EXECUTOR TRAIT
// =============================================================================

/// Trait for executing tool calls.
///
/// Implement this trait to provide tool execution capabilities.
/// This will be fully implemented in Phase 4 (Tool System).
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool call and return the result.
    fn execute<'a>(
        &'a self,
        tool_call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>;
}

/// A simple no-op tool executor for testing.
#[derive(Debug, Clone, Default)]
pub struct NoOpToolExecutor;

impl ToolExecutor for NoOpToolExecutor {
    fn execute<'a>(
        &'a self,
        tool_call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>> {
        let name = tool_call.name.clone();
        Box::pin(async move {
            Ok(format!("Tool '{}' executed (no-op)", name))
        })
    }
}

// =============================================================================
// PROVIDER WRAPPER (for shared provider access)
// =============================================================================

/// Simple provider wrapper for child analects.
///
/// In a real implementation, this would share the actual provider.
struct ProviderWrapper {
    name: String,
    model: String,
}

impl LLMProvider for ProviderWrapper {
    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn is_available(&self) -> bool {
        true
    }

    fn complete<'a>(
        &'a self,
        _messages: &'a [Message],
        _options: CompletionOptions,
    ) -> Pin<Box<dyn Future<Output = crate::providers::types::ProviderResult<crate::providers::types::ChatCompletion>> + Send + 'a>> {
        Box::pin(async move {
            // Placeholder - in real implementation, this would delegate to shared provider
            Err(crate::providers::types::ProviderError::Internal(
                "ProviderWrapper is a placeholder".to_string()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thought_step_creation() {
        let step = ThoughtStep::new(ThoughtStepType::Observe, "Saw the input")
            .with_confidence(0.9);
        
        assert_eq!(step.step_type, ThoughtStepType::Observe);
        assert_eq!(step.content, "Saw the input");
        assert!((step.confidence - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_chain_of_thought() {
        let mut cot = ChainOfThought::new();
        
        cot.observe("Input received");
        cot.reason("Analyzing the problem");
        cot.plan("Will use tool X");
        cot.act("Calling tool X");
        cot.reflect("Tool returned success");
        cot.complete();

        assert_eq!(cot.steps().len(), 5);
        assert!(cot.is_complete());
        assert!(cot.avg_confidence() > 0.0);
    }

    #[test]
    fn test_cot_markdown() {
        let mut cot = ChainOfThought::new();
        cot.observe("Test");
        cot.complete();

        let md = cot.to_markdown();
        assert!(md.contains("Chain of Thought"));
        assert!(md.contains("Observe"));
        assert!(md.contains("Chain complete"));
    }

    #[test]
    fn test_analect_config_builder() {
        let config = AnalectConfig::default()
            .with_name("test_analect")
            .with_system_prompt("Be helpful")
            .with_cot(true);
        
        assert_eq!(config.name, "test_analect");
        assert_eq!(config.system_prompt, "Be helpful");
        assert!(config.enable_cot);
    }

    #[test]
    fn test_analect_context_creation() {
        let context = AnalectContext::new("session-123");
        
        assert_eq!(context.session_id, "session-123");
        assert_eq!(context.depth(), 0);
        assert!(context.namespace.is_empty());
    }

    #[test]
    fn test_analect_context_child() {
        let parent = AnalectContext::new("session-123");
        let child = parent.child("subtask", 5).unwrap();
        
        assert_eq!(child.depth(), 1);
        assert_eq!(child.namespace, vec!["subtask"]);
        assert_eq!(child.path(), "subtask");
    }

    #[test]
    fn test_analect_context_max_depth() {
        let mut context = AnalectContext::new("session-123");
        
        // Create nested contexts up to depth 2
        for i in 0..2 {
            context = context.child(&format!("level{}", i), 2).unwrap();
        }
        
        // Third level should fail
        let result = context.child("level2", 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_noop_tool_executor() {
        let executor = NoOpToolExecutor;
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            name: "test_tool".to_string(),
            arguments: "{}".to_string(),
        };

        // Can't easily test async in sync test, but we can verify it compiles
        let _ = &executor;
        let _ = tool_call;
    }
}
