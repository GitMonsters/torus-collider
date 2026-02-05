//! # Agent Module
//!
//! Unified agent system that integrates LLM providers, tools, and safety.
//!
//! This module provides the [`AgentRunner`] which implements the complete agentic loop:
//! user input → LLM completion → tool execution → LLM completion → ... → final response.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        AgentRunner                                  │
//! │                                                                     │
//! │  ┌──────────┐    ┌───────────────┐    ┌──────────────┐             │
//! │  │  LLM     │◄──►│  AgentContext │◄──►│  ToolExecutor│             │
//! │  │ Provider │    │  (history,    │    │  (Registry)  │             │
//! │  └──────────┘    │   state)      │    └──────────────┘             │
//! │       ▲          └───────────────┘            ▲                    │
//! │       │                  ▲                    │                    │
//! │       │                  │                    │                    │
//! │       │          ┌───────────────┐            │                    │
//! │       └──────────│ EthicsEnforcer│────────────┘                    │
//! │                  │   (optional)  │                                 │
//! │                  └───────────────┘                                 │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use torus_attention::agent::{AgentRunner, AgentRunnerConfig};
//! use torus_attention::providers::local::LocalProvider;
//! use torus_attention::tools::ToolRegistry;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create LLM provider
//!     let provider = Box::new(LocalProvider::new("path/to/model"));
//!     
//!     // Create tool registry
//!     let tools = Arc::new(ToolRegistry::new());
//!     
//!     // Create and run agent
//!     let mut agent = AgentRunner::new(provider, tools)
//!         .with_config(AgentRunnerConfig::default()
//!             .with_system_prompt("You are a helpful assistant.")
//!             .with_max_iterations(10));
//!     
//!     let response = agent.run("Hello, what can you do?").await.unwrap();
//!     println!("{}", response.content);
//! }
//! ```
//!
//! ## Safety Integration
//!
//! The agent can optionally integrate with the [`EthicsEnforcer`] to validate
//! tool calls against the Prime Directive before execution:
//!
//! ```rust,ignore
//! use torus_attention::safety::ethics::EthicsEnforcer;
//!
//! let agent = AgentRunner::new(provider, tools)
//!     .with_safety(EthicsEnforcer::default());
//! ```
//!
//! ## Event Handling
//!
//! The agent emits events during execution for observability:
//!
//! ```rust,ignore
//! use torus_attention::agent::{AgentRunner, CollectingEventHandler};
//! use std::sync::Arc;
//!
//! let handler = Arc::new(CollectingEventHandler::new());
//! let mut agent = AgentRunner::new(provider, tools)
//!     .with_event_handler(handler.clone());
//!
//! let _ = agent.run("Do something").await;
//! for event in handler.events() {
//!     println!("{:?}", event);
//! }
//! ```

pub mod context;
pub mod runner;
pub mod types;

// Re-exports for convenience
pub use context::{AgentContext, ContextConfig, ContextSummary, ThinkingEntry};
pub use runner::{
    AgentRunner, AgentRunnerConfig, CollectingEventHandler, EventHandler, NoOpEventHandler,
};
pub use types::{
    AgentError, AgentEvent, AgentMessage, AgentResponse, AgentResult, ToolCallInfo, ToolCallRecord,
};

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reexports_available() {
        // Verify all re-exports are accessible
        let _ = AgentRunnerConfig::default();
        let _ = ContextConfig::default();
        let _: AgentResult<()> = Ok(());
    }
}
