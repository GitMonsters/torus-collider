//! # Agent Runner
//!
//! The main agentic loop that integrates LLM providers, tools, and safety.
//! This implements the complete flow: input → LLM → tools → LLM → ... → output.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::context::{AgentContext, ContextConfig};
use super::types::{
    AgentError, AgentEvent, AgentResponse, AgentResult, ToolCallInfo, ToolCallRecord,
};
use crate::providers::traits::LLMProvider;
use crate::providers::types::{CompletionOptions, FinishReason};
use crate::safety::ethics::EthicsEnforcer;
use crate::safety::proposed_action::ProposedAction;
use crate::tools::traits::ToolExecutor;

// =============================================================================
// RUNNER CONFIGURATION
// =============================================================================

/// Configuration for the agent runner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunnerConfig {
    /// Maximum iterations before stopping
    pub max_iterations: usize,
    /// Maximum total tokens before stopping
    pub max_total_tokens: usize,
    /// Whether to enable safety checks
    pub enable_safety: bool,
    /// System prompt template
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Completion options for LLM
    pub completion_options: CompletionOptions,
    /// Context configuration
    pub context_config: ContextConfig,
    /// Whether to emit events
    pub emit_events: bool,
    /// Stop on first tool error
    pub stop_on_tool_error: bool,
    /// Maximum tool calls per iteration
    pub max_tool_calls_per_iteration: usize,
}

impl Default for AgentRunnerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_total_tokens: 100_000,
            enable_safety: true,
            system_prompt: None,
            completion_options: CompletionOptions::default(),
            context_config: ContextConfig::default(),
            emit_events: false,
            stop_on_tool_error: false,
            max_tool_calls_per_iteration: 10,
        }
    }
}

impl AgentRunnerConfig {
    /// Create a minimal config for testing
    pub fn minimal() -> Self {
        Self {
            max_iterations: 3,
            max_total_tokens: 10_000,
            enable_safety: false,
            system_prompt: None,
            completion_options: CompletionOptions::default(),
            context_config: ContextConfig::minimal(),
            emit_events: false,
            stop_on_tool_error: true,
            max_tool_calls_per_iteration: 5,
        }
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set max iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Enable or disable safety
    pub fn with_safety(mut self, enabled: bool) -> Self {
        self.enable_safety = enabled;
        self
    }
}

// =============================================================================
// EVENT HANDLER TRAIT
// =============================================================================

/// Handler for agent events
pub trait EventHandler: Send + Sync {
    /// Handle an event
    fn handle<'a>(
        &'a self,
        event: &'a AgentEvent,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

/// No-op event handler
pub struct NoOpEventHandler;

impl EventHandler for NoOpEventHandler {
    fn handle<'a>(
        &'a self,
        _event: &'a AgentEvent,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {})
    }
}

/// Collecting event handler (stores events)
pub struct CollectingEventHandler {
    events: std::sync::Mutex<Vec<AgentEvent>>,
}

impl CollectingEventHandler {
    pub fn new() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn events(&self) -> Vec<AgentEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl Default for CollectingEventHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl EventHandler for CollectingEventHandler {
    fn handle<'a>(
        &'a self,
        event: &'a AgentEvent,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        let event_clone = event.clone();
        Box::pin(async move {
            if let Ok(mut events) = self.events.lock() {
                events.push(event_clone);
            }
        })
    }
}

// =============================================================================
// AGENT RUNNER
// =============================================================================

/// The main agent runner that orchestrates the agentic loop
pub struct AgentRunner {
    /// LLM provider
    provider: Box<dyn LLMProvider>,
    /// Tool executor
    tools: Arc<dyn ToolExecutor>,
    /// Optional safety enforcer
    safety: Option<EthicsEnforcer>,
    /// Configuration
    pub config: AgentRunnerConfig,
    /// Event handler
    event_handler: Arc<dyn EventHandler>,
}

impl AgentRunner {
    /// Create a new agent runner
    pub fn new(
        provider: Box<dyn LLMProvider>,
        tools: Arc<dyn ToolExecutor>,
    ) -> Self {
        Self {
            provider,
            tools,
            safety: None,
            config: AgentRunnerConfig::default(),
            event_handler: Arc::new(NoOpEventHandler),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: AgentRunnerConfig) -> Self {
        self.config = config;
        self
    }

    /// Set safety enforcer
    pub fn with_safety(mut self, enforcer: EthicsEnforcer) -> Self {
        self.safety = Some(enforcer);
        self
    }

    /// Set event handler
    pub fn with_event_handler(mut self, handler: Arc<dyn EventHandler>) -> Self {
        self.event_handler = handler;
        self
    }

    /// Run the agent with the given input
    pub fn run<'a>(
        &'a mut self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = AgentResult<AgentResponse>> + Send + 'a>> {
        Box::pin(async move {
            let run_id = AgentContext::generate_run_id();
            let mut context = AgentContext::new(&run_id)
                .with_config(self.config.context_config.clone());

            // Set system prompt if configured
            if let Some(ref system) = self.config.system_prompt {
                context = context.with_system_prompt(system);
            }

            self.run_with_context(input, &mut context).await
        })
    }

    /// Run with an existing context
    pub fn run_with_context<'a>(
        &'a mut self,
        input: &'a str,
        context: &'a mut AgentContext,
    ) -> Pin<Box<dyn Future<Output = AgentResult<AgentResponse>> + Send + 'a>> {
        Box::pin(async move {
            let start = Instant::now();

            // Emit started event
            self.emit_event(&AgentEvent::Started {
                run_id: context.run_id.clone(),
                input: input.to_string(),
                timestamp_ms: now_ms(),
            })
            .await;

            // Add user input
            context.add_user_message(input);

            // Main agentic loop
            let mut final_content = String::new();

            loop {
                // Check iteration limit
                if context.iteration >= self.config.max_iterations {
                    return Err(AgentError::MaxIterationsExceeded {
                        iterations: context.iteration,
                        limit: self.config.max_iterations,
                    });
                }

                // Check token limit
                if context.total_usage.total_tokens >= self.config.max_total_tokens {
                    return Err(AgentError::MaxTokensExceeded {
                        tokens: context.total_usage.total_tokens,
                        limit: self.config.max_total_tokens,
                    });
                }

                // Prepare completion options with tools
                let mut options = self.config.completion_options.clone();
                options.tools = self.tools.tool_definitions();

                // Get LLM completion
                let messages = context.get_messages_for_llm();
                let completion = self.provider.complete(&messages, options).await?;

                // Track usage
                context.add_usage(completion.usage);

                // Emit LLM response event
                let has_tool_calls = completion.has_tool_calls();
                self.emit_event(&AgentEvent::LLMResponse {
                    run_id: context.run_id.clone(),
                    content: completion.content().to_string(),
                    has_tool_calls,
                    usage: completion.usage,
                    iteration: context.iteration,
                })
                .await;

                // Handle response based on finish reason
                match completion.finish_reason {
                    FinishReason::Stop | FinishReason::StopSequence => {
                        // Natural completion - we're done
                        final_content = completion.content().to_string();
                        context.add_assistant_message(&final_content);
                        break;
                    }
                    FinishReason::ToolCalls => {
                        // Tool calls - need to execute and continue
                        let tool_calls = completion.tool_calls().cloned().unwrap_or_default();

                        if tool_calls.is_empty() {
                            // Weird state - finish reason says tool calls but none present
                            final_content = completion.content().to_string();
                            context.add_assistant_message(&final_content);
                            break;
                        }

                        // Add assistant message with tool calls
                        context.add_assistant_with_tools(
                            completion.content(),
                            tool_calls.clone(),
                        );

                        // Execute tool calls (up to max per iteration)
                        let calls_to_execute =
                            tool_calls.into_iter().take(self.config.max_tool_calls_per_iteration);

                        for tool_call in calls_to_execute {
                            // Safety check if enabled
                            if self.config.enable_safety {
                                if let Some(ref mut safety) = self.safety {
                                    let action = ProposedAction::new(format!(
                                        "Execute tool '{}' with args: {}",
                                        tool_call.name, tool_call.arguments
                                    ))
                                    .with_source("agent_runner")
                                    .with_benefit_to_self(0.3)
                                    .with_benefit_to_other(0.5);

                                    let result = safety.validate_action_mut(&action);

                                    self.emit_event(&AgentEvent::SafetyCheck {
                                        run_id: context.run_id.clone(),
                                        action: format!("tool:{}", tool_call.name),
                                        allowed: result.allowed,
                                        reason: if result.allowed {
                                            None
                                        } else {
                                            Some(result.reason.clone())
                                        },
                                    })
                                    .await;

                                    if !result.allowed {
                                        // Add a tool result indicating blocked
                                        context.add_tool_result(
                                            &tool_call.id,
                                            format!(
                                                "BLOCKED by safety: {}. Suggestions: {:?}",
                                                result.reason, result.suggestions
                                            ),
                                        );
                                        continue;
                                    }
                                }
                            }

                            // Emit tool call started event
                            self.emit_event(&AgentEvent::ToolCallStarted {
                                run_id: context.run_id.clone(),
                                tool_call: ToolCallInfo::from(&tool_call),
                                iteration: context.iteration,
                            })
                            .await;

                            // Execute the tool
                            let tool_start = Instant::now();
                            let tool_result = self.tools.execute(&tool_call).await;
                            let tool_duration_ms = tool_start.elapsed().as_millis() as u64;

                            // Process result
                            let (success, output) = match &tool_result {
                                Ok(output) => (output.success, output.output.clone()),
                                Err(e) => (false, format!("Error: {}", e)),
                            };

                            // Record the tool call
                            let record = ToolCallRecord::new(
                                &tool_call,
                                success,
                                output.clone(),
                                tool_duration_ms,
                                context.iteration,
                            );
                            context.add_tool_call_record(record);

                            // Emit tool call completed event
                            self.emit_event(&AgentEvent::ToolCallCompleted {
                                run_id: context.run_id.clone(),
                                tool_call_id: tool_call.id.clone(),
                                success,
                                output: truncate_for_event(&output, 500),
                                duration_ms: tool_duration_ms,
                            })
                            .await;

                            // Add tool result to context
                            context.add_tool_result(&tool_call.id, &output);

                            // Handle error if configured to stop
                            if !success && self.config.stop_on_tool_error {
                                if let Err(e) = tool_result {
                                    return Err(AgentError::Tool(e));
                                }
                            }
                        }
                    }
                    FinishReason::Length => {
                        // Hit token limit - add partial response and continue
                        context.add_assistant_message(completion.content());
                        final_content = completion.content().to_string();
                        // Don't break - let the loop continue to check limits
                    }
                    FinishReason::ContentFilter => {
                        // Content was filtered - report error
                        return Err(AgentError::SafetyViolation {
                            violation_type: crate::safety::EthicsViolationType::HarmToOther,
                            message: "Content was blocked by provider safety filter".to_string(),
                            suggestions: vec!["Rephrase the request".to_string()],
                        });
                    }
                    FinishReason::Other => {
                        // Unknown finish reason - treat as completion
                        final_content = completion.content().to_string();
                        context.add_assistant_message(&final_content);
                        break;
                    }
                }

                // Move to next iteration
                context.next_iteration();
            }

            // Calculate final metrics
            let duration_ms = start.elapsed().as_millis() as u64;

            // Build response
            let mut response = AgentResponse::new(
                context.run_id.clone(),
                final_content.clone(),
                context.messages.clone(),
                context.iteration + 1,
                context.total_usage,
                duration_ms,
            );
            response.tool_calls_made = context.tool_calls.clone();
            response.safety_applied = self.config.enable_safety && self.safety.is_some();

            // Emit completed event
            self.emit_event(&AgentEvent::Completed {
                run_id: context.run_id.clone(),
                response: truncate_for_event(&final_content, 1000),
                iterations: context.iteration + 1,
                total_usage: context.total_usage,
                duration_ms,
            })
            .await;

            Ok(response)
        })
    }

    /// Emit an event
    async fn emit_event(&self, event: &AgentEvent) {
        if self.config.emit_events {
            self.event_handler.handle(event).await;
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Get current timestamp in milliseconds
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Truncate a string for event logging
fn truncate_for_event(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...[truncated]", &s[..max_len])
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::types::{ChatCompletion, Message, ToolCall, ToolDefinition, Usage};
    use crate::tools::types::{ToolError, ToolOutput, ToolResult};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -------------------------------------------------------------------------
    // Mock Provider
    // -------------------------------------------------------------------------

    struct MockProvider {
        responses: std::sync::Mutex<Vec<ChatCompletion>>,
        call_count: AtomicUsize,
    }

    impl MockProvider {
        fn new(responses: Vec<ChatCompletion>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
                call_count: AtomicUsize::new(0),
            }
        }

        fn single_response(content: &str) -> Self {
            Self::new(vec![ChatCompletion {
                id: "comp-1".to_string(),
                model: "mock".to_string(),
                message: Message::assistant(content),
                finish_reason: FinishReason::Stop,
                usage: Usage::new(10, 20),
                metadata: Default::default(),
            }])
        }

        fn with_tool_call(tool_call: ToolCall, final_response: &str) -> Self {
            let mut msg_with_tools = Message::assistant("I'll use the tool");
            msg_with_tools.tool_calls = Some(vec![tool_call]);

            Self::new(vec![
                ChatCompletion {
                    id: "comp-1".to_string(),
                    model: "mock".to_string(),
                    message: msg_with_tools,
                    finish_reason: FinishReason::ToolCalls,
                    usage: Usage::new(10, 20),
                    metadata: Default::default(),
                },
                ChatCompletion {
                    id: "comp-2".to_string(),
                    model: "mock".to_string(),
                    message: Message::assistant(final_response),
                    finish_reason: FinishReason::Stop,
                    usage: Usage::new(10, 20),
                    metadata: Default::default(),
                },
            ])
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
        ) -> Pin<Box<dyn Future<Output = crate::providers::types::ProviderResult<ChatCompletion>> + Send + 'a>>
        {
            Box::pin(async move {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    Err(crate::providers::types::ProviderError::Internal(
                        "No more mock responses".to_string(),
                    ))
                } else {
                    Ok(responses.remove(0))
                }
            })
        }
    }

    // -------------------------------------------------------------------------
    // Mock Tool Executor
    // -------------------------------------------------------------------------

    struct MockToolExecutor {
        outputs: std::sync::Mutex<std::collections::HashMap<String, ToolOutput>>,
    }

    impl MockToolExecutor {
        fn new() -> Self {
            Self {
                outputs: std::sync::Mutex::new(std::collections::HashMap::new()),
            }
        }

        fn with_output(self, tool_name: &str, output: ToolOutput) -> Self {
            self.outputs
                .lock()
                .unwrap()
                .insert(tool_name.to_string(), output);
            self
        }
    }

    impl ToolExecutor for MockToolExecutor {
        fn execute<'a>(
            &'a self,
            call: &'a ToolCall,
        ) -> Pin<Box<dyn Future<Output = ToolResult<ToolOutput>> + Send + 'a>> {
            Box::pin(async move {
                let outputs = self.outputs.lock().unwrap();
                if let Some(output) = outputs.get(&call.name) {
                    Ok(output.clone())
                } else {
                    Err(ToolError::not_found(&call.name))
                }
            })
        }

        fn has_tool(&self, name: &str) -> bool {
            self.outputs.lock().unwrap().contains_key(name)
        }

        fn tool_definitions(&self) -> Vec<ToolDefinition> {
            self.outputs
                .lock()
                .unwrap()
                .keys()
                .map(|name| ToolDefinition::new(name, "Mock tool", serde_json::json!({})))
                .collect()
        }

        fn tool_specs(&self) -> Vec<crate::tools::types::ToolSpec> {
            vec![]
        }
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = AgentRunnerConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert!(config.enable_safety);
    }

    #[test]
    fn test_config_minimal() {
        let config = AgentRunnerConfig::minimal();
        assert_eq!(config.max_iterations, 3);
        assert!(!config.enable_safety);
    }

    #[test]
    fn test_config_with_system_prompt() {
        let config = AgentRunnerConfig::default().with_system_prompt("Be helpful");
        assert_eq!(config.system_prompt, Some("Be helpful".to_string()));
    }

    #[tokio::test]
    async fn test_simple_completion() {
        let provider = MockProvider::single_response("Hello! How can I help?");
        let tools = Arc::new(MockToolExecutor::new());

        let mut runner = AgentRunner::new(Box::new(provider), tools)
            .with_config(AgentRunnerConfig::minimal());

        let response = runner.run("Hi there").await.unwrap();

        assert_eq!(response.content, "Hello! How can I help?");
        assert_eq!(response.iterations, 1);
        assert!(response.tool_calls_made.is_empty());
    }

    #[tokio::test]
    async fn test_with_tool_call() {
        let tool_call = ToolCall {
            id: "call-1".to_string(),
            name: "read".to_string(),
            arguments: r#"{"path": "/tmp/test"}"#.to_string(),
        };
        let provider = MockProvider::with_tool_call(tool_call, "The file contains test data.");
        let tools = Arc::new(
            MockToolExecutor::new().with_output("read", ToolOutput::success("test content")),
        );

        let mut runner = AgentRunner::new(Box::new(provider), tools)
            .with_config(AgentRunnerConfig::minimal());

        let response = runner.run("Read the file").await.unwrap();

        assert_eq!(response.content, "The file contains test data.");
        assert_eq!(response.tool_calls_made.len(), 1);
        assert_eq!(response.tool_calls_made[0].name, "read");
        assert!(response.tool_calls_made[0].success);
    }

    #[tokio::test]
    async fn test_max_iterations_exceeded() {
        // Provider that always returns tool calls
        let mut responses = Vec::new();
        for i in 0..5 {
            let mut msg = Message::assistant("Using tool");
            msg.tool_calls = Some(vec![ToolCall {
                id: format!("call-{}", i),
                name: "read".to_string(),
                arguments: "{}".to_string(),
            }]);
            responses.push(ChatCompletion {
                id: format!("comp-{}", i),
                model: "mock".to_string(),
                message: msg,
                finish_reason: FinishReason::ToolCalls,
                usage: Usage::new(10, 20),
                metadata: Default::default(),
            });
        }
        let provider = MockProvider::new(responses);
        let tools = Arc::new(
            MockToolExecutor::new().with_output("read", ToolOutput::success("data")),
        );

        let mut runner = AgentRunner::new(Box::new(provider), tools)
            .with_config(AgentRunnerConfig::minimal().with_max_iterations(2));

        let result = runner.run("Keep reading").await;

        assert!(matches!(
            result,
            Err(AgentError::MaxIterationsExceeded { iterations: 2, limit: 2 })
        ));
    }

    #[tokio::test]
    async fn test_event_collection() {
        let provider = MockProvider::single_response("Done!");
        let tools = Arc::new(MockToolExecutor::new());
        let handler = Arc::new(CollectingEventHandler::new());

        let mut runner = AgentRunner::new(Box::new(provider), tools)
            .with_config(AgentRunnerConfig {
                emit_events: true,
                ..AgentRunnerConfig::minimal()
            })
            .with_event_handler(handler.clone());

        let _ = runner.run("Test").await.unwrap();

        let events = handler.events();
        assert!(events.len() >= 2); // At least Started and Completed
        assert!(matches!(events[0], AgentEvent::Started { .. }));
        assert!(matches!(events.last(), Some(AgentEvent::Completed { .. })));
    }

    #[tokio::test]
    async fn test_tool_error_handling() {
        let tool_call = ToolCall {
            id: "call-1".to_string(),
            name: "nonexistent".to_string(),
            arguments: "{}".to_string(),
        };
        let provider = MockProvider::with_tool_call(tool_call, "Done anyway");
        let tools = Arc::new(MockToolExecutor::new()); // No tools registered

        let mut runner = AgentRunner::new(Box::new(provider), tools)
            .with_config(AgentRunnerConfig {
                stop_on_tool_error: false, // Allow completion even with tool errors
                ..AgentRunnerConfig::minimal()
            });

        let response = runner.run("Use tool").await.unwrap();

        // Should complete but tool call should be marked as failed
        assert_eq!(response.tool_calls_made.len(), 1);
        assert!(!response.tool_calls_made[0].success);
    }

    #[tokio::test]
    async fn test_stop_on_tool_error() {
        let tool_call = ToolCall {
            id: "call-1".to_string(),
            name: "nonexistent".to_string(),
            arguments: "{}".to_string(),
        };
        let provider = MockProvider::with_tool_call(tool_call, "Done anyway");
        let tools = Arc::new(MockToolExecutor::new());

        let mut runner = AgentRunner::new(Box::new(provider), tools).with_config(
            AgentRunnerConfig {
                stop_on_tool_error: true,
                ..AgentRunnerConfig::minimal()
            },
        );

        let result = runner.run("Use tool").await;

        assert!(matches!(result, Err(AgentError::Tool(_))));
    }

    #[test]
    fn test_truncate_for_event() {
        assert_eq!(truncate_for_event("short", 10), "short");
        assert_eq!(
            truncate_for_event("this is a long string", 10),
            "this is a ...[truncated]"
        );
    }

    #[test]
    fn test_now_ms() {
        let ts = now_ms();
        assert!(ts > 0);
    }
}
