//! Agent Orchestrator Module
//!
//! This module provides the orchestration layer for AI agents, inspired by
//! Meta's Confucius framework (CCA-SWEBench). It includes:
//!
//! - **Memory Management**: Hierarchical memory with conversation, task, and persistent scopes
//! - **Agent State Machine**: Turn-based execution with states (Idle, Thinking, Acting, etc.)
//! - **Analect Pattern**: Chain-of-thought orchestration with tool invocation coordination
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         Analect                                 │
//! │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
//! │  │   Memory     │  │    Agent     │  │    Tool System     │    │
//! │  │   Manager    │  │    State     │  │    (Phase 4)       │    │
//! │  │              │  │    Machine   │  │                    │    │
//! │  │ ┌──────────┐ │  │              │  │                    │    │
//! │  │ │Convers.  │ │  │  Idle → Think│  │                    │    │
//! │  │ │  Scope   │ │  │    ↓     ↓   │  │                    │    │
//! │  │ ├──────────┤ │  │  Wait ← Act  │  │                    │    │
//! │  │ │Task Scope│ │  │              │  │                    │    │
//! │  │ ├──────────┤ │  └──────────────┘  └────────────────────┘    │
//! │  │ │Persistent│ │                                               │
//! │  │ │  Scope   │ │                                               │
//! │  │ └──────────┘ │                                               │
//! │  └──────────────┘                                               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use torus_attention::orchestrator::{Analect, AgentConfig, MemoryScope};
//! use torus_attention::providers::AnthropicProvider;
//!
//! // Create an orchestrator
//! let config = AgentConfig::default();
//! let provider = AnthropicProvider::new("api_key", "claude-3-opus")?;
//! let analect = Analect::new(config, Box::new(provider))?;
//!
//! // Run a conversation turn
//! let response = analect.turn("Hello, how can you help?").await?;
//! ```

mod memory;
mod agent;
mod analect;
mod types;

// Re-export main types
pub use memory::{
    MemoryManager, MemoryScope, MemoryMessage, MessageType,
    MemoryConfig, MemoryStats, HistoryVisibility,
};

pub use agent::{
    Agent, AgentState, AgentConfig, AgentStats, TurnResult,
};

pub use analect::{
    Analect, AnalectConfig, AnalectContext, AnalectStats,
    ChainOfThought, ThoughtStep,
};

pub use types::{
    OrchestratorError, OrchestratorResult,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all exports are accessible
        let _scope = MemoryScope::Conversation;
        let _state = AgentState::Idle;
        let _msg_type = MessageType::User;
    }
}
