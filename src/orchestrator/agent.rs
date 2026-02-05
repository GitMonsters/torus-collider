//! Agent State Machine for turn-based execution.
//!
//! Manages the agent's state transitions during a conversation turn:
//! - Idle: Waiting for input
//! - Thinking: Processing input, generating thoughts
//! - Acting: Executing actions (tool calls, responses)
//! - WaitingForTool: Waiting for tool execution results
//! - Error: Recovery from errors
//!
//! Integrates with SafetyGuard for Prime Directive enforcement.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::memory::{HistoryVisibility, MemoryManager, MemoryMessage, MessageType};
use super::types::{OrchestratorError, OrchestratorResult};
use crate::providers::types::{ChatCompletion, CompletionOptions, Message, ToolCall};
use crate::providers::LLMProvider;
use crate::safety::{ProposedAction, SafetyActionResult, SafetyGuard};

// =============================================================================
// AGENT STATE
// =============================================================================

/// State of the agent in its execution cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentState {
    /// Agent is idle, waiting for input
    Idle,
    /// Agent is processing/thinking
    Thinking,
    /// Agent is executing an action
    Acting,
    /// Agent is waiting for tool results
    WaitingForTool,
    /// Agent encountered an error
    Error,
    /// Agent has completed its turn
    Complete,
}

impl AgentState {
    /// Check if transition to a new state is valid.
    pub fn can_transition_to(&self, target: AgentState) -> bool {
        use AgentState::*;
        
        match (self, target) {
            // From Idle
            (Idle, Thinking) => true,
            (Idle, Error) => true,
            
            // From Thinking
            (Thinking, Acting) => true,
            (Thinking, Complete) => true, // No action needed
            (Thinking, Error) => true,
            
            // From Acting
            (Acting, WaitingForTool) => true,
            (Acting, Complete) => true,
            (Acting, Thinking) => true, // Multi-step reasoning
            (Acting, Error) => true,
            
            // From WaitingForTool
            (WaitingForTool, Thinking) => true, // Process tool results
            (WaitingForTool, Acting) => true,   // More actions after tool
            (WaitingForTool, Complete) => true,
            (WaitingForTool, Error) => true,
            
            // From Error
            (Error, Idle) => true, // Reset
            (Error, Complete) => true,
            
            // From Complete
            (Complete, Idle) => true, // New turn
            
            // All other transitions are invalid
            _ => false,
        }
    }
}

impl std::fmt::Display for AgentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Thinking => write!(f, "Thinking"),
            Self::Acting => write!(f, "Acting"),
            Self::WaitingForTool => write!(f, "WaitingForTool"),
            Self::Error => write!(f, "Error"),
            Self::Complete => write!(f, "Complete"),
        }
    }
}

// =============================================================================
// AGENT CONFIG
// =============================================================================

/// Configuration for the agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// System prompt
    pub system_prompt: String,
    /// Maximum turns before forcing completion
    pub max_turns: usize,
    /// Maximum tool calls per turn
    pub max_tool_calls: usize,
    /// Timeout per LLM call (milliseconds)
    pub llm_timeout_ms: u64,
    /// Whether to enable chain-of-thought
    pub enable_cot: bool,
    /// Completion options
    pub completion_options: CompletionOptions,
    /// Whether to validate actions with safety guard
    pub enforce_safety: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful AI assistant.".to_string(),
            max_turns: 10,
            max_tool_calls: 5,
            llm_timeout_ms: 60_000,
            enable_cot: true,
            completion_options: CompletionOptions::default(),
            enforce_safety: true,
        }
    }
}

impl AgentConfig {
    /// Create a config with a custom system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Enable or disable chain-of-thought.
    pub fn with_cot(mut self, enable: bool) -> Self {
        self.enable_cot = enable;
        self
    }

    /// Set maximum turns.
    pub fn with_max_turns(mut self, max: usize) -> Self {
        self.max_turns = max;
        self
    }

    /// Enable or disable safety enforcement.
    pub fn with_safety(mut self, enforce: bool) -> Self {
        self.enforce_safety = enforce;
        self
    }
}

// =============================================================================
// AGENT STATISTICS
// =============================================================================

/// Statistics for the current agent session.
#[derive(Debug, Clone, Default)]
pub struct AgentStats {
    /// Number of turns completed
    pub turns_completed: usize,
    /// Total tool calls made
    pub tool_calls_made: usize,
    /// Total tokens used (prompt + completion)
    pub total_tokens: usize,
    /// Actions blocked by safety
    pub actions_blocked: usize,
    /// Errors encountered
    pub errors: usize,
    /// Average response time (ms)
    pub avg_response_time_ms: u64,
    /// Total time in thinking state (ms)
    pub thinking_time_ms: u64,
}

// =============================================================================
// TURN RESULT
// =============================================================================

/// Result of a single agent turn.
#[derive(Debug, Clone)]
pub struct TurnResult {
    /// The agent's response
    pub response: String,
    /// Tool calls made (if any)
    pub tool_calls: Vec<ToolCall>,
    /// Whether the turn is complete
    pub is_complete: bool,
    /// Finish reason
    pub finish_reason: TurnFinishReason,
    /// Token usage
    pub tokens_used: usize,
    /// Time taken (ms)
    pub duration_ms: u64,
}

/// Reason why the turn finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnFinishReason {
    /// Natural completion
    Complete,
    /// Tool calls need execution
    ToolCallsPending,
    /// Max turns reached
    MaxTurnsReached,
    /// Blocked by safety
    SafetyBlocked,
    /// Error occurred
    Error,
}

// =============================================================================
// AGENT
// =============================================================================

/// The main agent implementation with state machine.
pub struct Agent {
    /// Current state
    state: AgentState,
    /// Configuration
    config: AgentConfig,
    /// LLM provider
    provider: Box<dyn LLMProvider>,
    /// Memory manager
    memory: MemoryManager,
    /// Safety guard (optional)
    safety_guard: Option<Box<dyn SafetyGuard>>,
    /// Statistics
    stats: AgentStats,
    /// Pending tool calls
    pending_tool_calls: Vec<ToolCall>,
    /// Current turn count
    turn_count: usize,
}

impl Agent {
    /// Create a new agent.
    pub fn new(
        config: AgentConfig,
        provider: Box<dyn LLMProvider>,
        memory: MemoryManager,
    ) -> Self {
        Self {
            state: AgentState::Idle,
            config,
            provider,
            memory,
            safety_guard: None,
            stats: AgentStats::default(),
            pending_tool_calls: Vec::new(),
            turn_count: 0,
        }
    }

    /// Add a safety guard.
    pub fn with_safety_guard(mut self, guard: Box<dyn SafetyGuard>) -> Self {
        self.safety_guard = Some(guard);
        self
    }

    /// Get current state.
    pub fn state(&self) -> AgentState {
        self.state
    }

    /// Get statistics.
    pub fn stats(&self) -> &AgentStats {
        &self.stats
    }

    /// Get pending tool calls.
    pub fn pending_tool_calls(&self) -> &[ToolCall] {
        &self.pending_tool_calls
    }

    /// Transition to a new state.
    fn transition(&mut self, target: AgentState) -> OrchestratorResult<()> {
        if self.state.can_transition_to(target) {
            self.state = target;
            Ok(())
        } else {
            Err(OrchestratorError::StateTransitionError {
                from: self.state.to_string(),
                to: target.to_string(),
                reason: "Invalid state transition".to_string(),
            })
        }
    }

    /// Execute a single turn.
    pub fn turn<'a>(
        &'a mut self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = OrchestratorResult<TurnResult>> + Send + 'a>> {
        Box::pin(async move {
            let start = Instant::now();

            // Check turn limit
            if self.turn_count >= self.config.max_turns {
                return Ok(TurnResult {
                    response: "Maximum turns reached.".to_string(),
                    tool_calls: Vec::new(),
                    is_complete: true,
                    finish_reason: TurnFinishReason::MaxTurnsReached,
                    tokens_used: 0,
                    duration_ms: 0,
                });
            }

            // Transition to thinking
            self.transition(AgentState::Thinking)?;
            let thinking_start = Instant::now();

            // Add user message to memory
            self.memory.add_message(MemoryMessage::user(input));

            // Build messages for the LLM
            let mut messages = vec![Message::system(&self.config.system_prompt)];
            messages.extend(self.memory.to_provider_messages(HistoryVisibility::Session));

            // Validate the request if safety is enabled
            if self.config.enforce_safety {
                if let Some(ref guard) = self.safety_guard {
                    let action = ProposedAction::new(input)
                        .with_benefit_to_self(0.3)
                        .with_benefit_to_other(0.7);
                    
                    let result = guard.validate_action(&action);
                    if !result.allowed {
                        self.stats.actions_blocked += 1;
                        self.transition(AgentState::Complete)?;
                        
                        return Ok(TurnResult {
                            response: format!("Action blocked: {}", result.reason),
                            tool_calls: Vec::new(),
                            is_complete: true,
                            finish_reason: TurnFinishReason::SafetyBlocked,
                            tokens_used: 0,
                            duration_ms: start.elapsed().as_millis() as u64,
                        });
                    }
                }
            }

            // Call the LLM
            let completion = self.provider.complete(&messages, self.config.completion_options.clone()).await
                .map_err(|e| OrchestratorError::ProviderError(e.to_string()))?;

            self.stats.thinking_time_ms += thinking_start.elapsed().as_millis() as u64;

            // Transition to acting
            self.transition(AgentState::Acting)?;

            // Process the response
            let response_text = completion.message.content.clone();
            let tool_calls = completion.message.tool_calls.clone().unwrap_or_default();

            // Add assistant message to memory
            self.memory.add_message(MemoryMessage::assistant(&response_text));

            // Handle tool calls
            let (is_complete, finish_reason) = if !tool_calls.is_empty() {
                self.pending_tool_calls = tool_calls.clone();
                self.stats.tool_calls_made += tool_calls.len();
                self.transition(AgentState::WaitingForTool)?;
                (false, TurnFinishReason::ToolCallsPending)
            } else {
                self.transition(AgentState::Complete)?;
                (true, TurnFinishReason::Complete)
            };

            // Update stats
            self.turn_count += 1;
            self.stats.turns_completed += 1;
            self.stats.total_tokens += completion.usage.total_tokens;

            let duration = start.elapsed().as_millis() as u64;
            self.stats.avg_response_time_ms = 
                (self.stats.avg_response_time_ms * (self.stats.turns_completed - 1) as u64 + duration) 
                / self.stats.turns_completed as u64;

            Ok(TurnResult {
                response: response_text,
                tool_calls,
                is_complete,
                finish_reason,
                tokens_used: completion.usage.total_tokens,
                duration_ms: duration,
            })
        })
    }

    /// Provide tool results and continue the turn.
    pub fn provide_tool_results<'a>(
        &'a mut self,
        results: Vec<(String, String)>, // (tool_call_id, result)
    ) -> Pin<Box<dyn Future<Output = OrchestratorResult<TurnResult>> + Send + 'a>> {
        Box::pin(async move {
            if self.state != AgentState::WaitingForTool {
                return Err(OrchestratorError::StateTransitionError {
                    from: self.state.to_string(),
                    to: "WaitingForTool".to_string(),
                    reason: "Not waiting for tool results".to_string(),
                });
            }

            let start = Instant::now();

            // Add tool results to memory
            for (tool_call_id, result) in &results {
                self.memory.add_message(MemoryMessage::tool_result(tool_call_id, result));
            }

            // Clear pending tool calls
            self.pending_tool_calls.clear();

            // Transition to thinking for next step
            self.transition(AgentState::Thinking)?;
            let thinking_start = Instant::now();

            // Build messages including tool results
            let mut messages = vec![Message::system(&self.config.system_prompt)];
            messages.extend(self.memory.to_provider_messages(HistoryVisibility::Session));

            // Call LLM again
            let completion = self.provider.complete(&messages, self.config.completion_options.clone()).await
                .map_err(|e| OrchestratorError::ProviderError(e.to_string()))?;

            self.stats.thinking_time_ms += thinking_start.elapsed().as_millis() as u64;

            // Transition to acting
            self.transition(AgentState::Acting)?;

            let response_text = completion.message.content.clone();
            let tool_calls = completion.message.tool_calls.clone().unwrap_or_default();

            // Add assistant response to memory
            self.memory.add_message(MemoryMessage::assistant(&response_text));

            // Handle any new tool calls
            let (is_complete, finish_reason) = if !tool_calls.is_empty() {
                self.pending_tool_calls = tool_calls.clone();
                self.stats.tool_calls_made += tool_calls.len();
                self.transition(AgentState::WaitingForTool)?;
                (false, TurnFinishReason::ToolCallsPending)
            } else {
                self.transition(AgentState::Complete)?;
                (true, TurnFinishReason::Complete)
            };

            // Update stats
            self.stats.turns_completed += 1;
            self.stats.total_tokens += completion.usage.total_tokens;

            let duration = start.elapsed().as_millis() as u64;

            Ok(TurnResult {
                response: response_text,
                tool_calls,
                is_complete,
                finish_reason,
                tokens_used: completion.usage.total_tokens,
                duration_ms: duration,
            })
        })
    }

    /// Reset the agent to idle state for a new conversation.
    pub fn reset(&mut self) {
        self.state = AgentState::Idle;
        self.turn_count = 0;
        self.pending_tool_calls.clear();
        self.memory.clear();
    }

    /// Get memory reference.
    pub fn memory(&self) -> &MemoryManager {
        &self.memory
    }

    /// Get mutable memory reference.
    pub fn memory_mut(&mut self) -> &mut MemoryManager {
        &mut self.memory
    }
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("state", &self.state)
            .field("turn_count", &self.turn_count)
            .field("pending_tool_calls", &self.pending_tool_calls.len())
            .field("stats", &self.stats)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::types::{FinishReason, Usage};

    // Mock provider for testing
    struct MockProvider {
        responses: Vec<String>,
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockProvider {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses,
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl LLMProvider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }

        fn model(&self) -> &str {
            "mock-model"
        }

        fn is_available(&self) -> bool {
            true
        }

        fn complete<'a>(
            &'a self,
            _messages: &'a [Message],
            _options: CompletionOptions,
        ) -> Pin<Box<dyn Future<Output = crate::providers::types::ProviderResult<ChatCompletion>> + Send + 'a>> {
            let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let response = self.responses.get(count).cloned().unwrap_or_default();
            
            Box::pin(async move {
                Ok(ChatCompletion {
                    id: format!("mock-{}", count),
                    model: "mock-model".to_string(),
                    message: Message::assistant(response),
                    finish_reason: FinishReason::Stop,
                    usage: Usage::new(10, 20),
                    metadata: std::collections::HashMap::new(),
                })
            })
        }
    }

    #[test]
    fn test_state_transitions() {
        use AgentState::*;
        
        assert!(Idle.can_transition_to(Thinking));
        assert!(Thinking.can_transition_to(Acting));
        assert!(Acting.can_transition_to(Complete));
        assert!(Acting.can_transition_to(WaitingForTool));
        assert!(WaitingForTool.can_transition_to(Thinking));
        
        assert!(!Idle.can_transition_to(Complete));
        assert!(!Complete.can_transition_to(Acting));
    }

    #[test]
    fn test_agent_config_builder() {
        let config = AgentConfig::default()
            .with_system_prompt("Be helpful")
            .with_max_turns(5)
            .with_cot(false);
        
        assert_eq!(config.system_prompt, "Be helpful");
        assert_eq!(config.max_turns, 5);
        assert!(!config.enable_cot);
    }

    #[tokio::test]
    async fn test_agent_basic_turn() {
        let provider = MockProvider::new(vec!["Hello! How can I help?".to_string()]);
        let memory = MemoryManager::with_defaults();
        let config = AgentConfig::default();
        
        let mut agent = Agent::new(config, Box::new(provider), memory);
        
        assert_eq!(agent.state(), AgentState::Idle);
        
        let result = agent.turn("Hi there").await.unwrap();
        
        assert_eq!(result.response, "Hello! How can I help?");
        assert!(result.is_complete);
        assert_eq!(result.finish_reason, TurnFinishReason::Complete);
        assert_eq!(agent.state(), AgentState::Complete);
        assert_eq!(agent.stats().turns_completed, 1);
    }

    #[tokio::test]
    async fn test_agent_max_turns() {
        let provider = MockProvider::new(vec!["Response".to_string(); 5]);
        let memory = MemoryManager::with_defaults();
        let config = AgentConfig::default().with_max_turns(2);
        
        let mut agent = Agent::new(config, Box::new(provider), memory);
        
        // First turn
        agent.turn("Turn 1").await.unwrap();
        agent.state = AgentState::Idle; // Reset for next turn
        
        // Second turn
        agent.turn("Turn 2").await.unwrap();
        agent.state = AgentState::Idle;
        
        // Third turn should be blocked
        let result = agent.turn("Turn 3").await.unwrap();
        assert_eq!(result.finish_reason, TurnFinishReason::MaxTurnsReached);
    }

    #[test]
    fn test_agent_reset() {
        let provider = MockProvider::new(vec![]);
        let memory = MemoryManager::with_defaults();
        let config = AgentConfig::default();
        
        let mut agent = Agent::new(config, Box::new(provider), memory);
        agent.state = AgentState::Complete;
        agent.turn_count = 5;
        
        agent.reset();
        
        assert_eq!(agent.state(), AgentState::Idle);
        assert_eq!(agent.turn_count, 0);
    }
}
