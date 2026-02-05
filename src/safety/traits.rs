//! Traits for safety-aware agents and components.
//!
//! These traits define the interface for implementing consciousness-aware behavior.

use super::{
    proposed_action::{ProposedAction, SafetyActionResult},
    relationship::{ConsciousnessRelation, ParasiticRisk, RelationshipHealth},
};

/// Trait for safety enforcement on actions.
///
/// Any component that validates actions should implement this trait.
/// The primary implementation is `EthicsEnforcer`.
///
/// # Example
///
/// ```rust,ignore
/// use torus_attention::safety::{SafetyGuard, ProposedAction, EthicsEnforcer};
///
/// let guard: Box<dyn SafetyGuard> = Box::new(EthicsEnforcer::new());
///
/// let action = ProposedAction::mutual_help("Help user", 0.3, 0.7);
/// let result = guard.validate_action(&action);
///
/// if result.allowed {
///     println!("Action approved: {}", result.reason);
/// }
/// ```
pub trait SafetyGuard: Send + Sync {
    /// Validate an action before execution.
    ///
    /// Returns a `SafetyActionResult` indicating whether the action is allowed
    /// and providing a reason for the decision.
    fn validate_action(&self, action: &ProposedAction) -> SafetyActionResult;

    /// Evaluate the health of a consciousness relationship.
    ///
    /// Returns the current health status of the relationship.
    fn evaluate_relationship(&self, relation: &ConsciousnessRelation) -> RelationshipHealth;

    /// Detect parasitism in a relationship.
    ///
    /// Returns the parasitic risk level for the given relationship.
    fn detect_parasitism(&self, relation: &ConsciousnessRelation) -> ParasiticRisk;

    /// Calculate the mutual benefit score for a relationship.
    ///
    /// Returns a score from 0.0 (no mutual benefit) to 1.0 (perfect symbiosis).
    fn mutual_benefit_score(&self, relation: &ConsciousnessRelation) -> f64;

    /// Check if the guard is enabled.
    fn is_enabled(&self) -> bool;

    /// Get the name/identifier of this guard.
    fn name(&self) -> &str;
}

/// Trait for agents that implement consciousness-aware behavior.
///
/// This trait should be implemented by any AI agent that participates
/// in conscious relationships with other entities.
///
/// # Example
///
/// ```rust,ignore
/// use torus_attention::safety::{ConsciousAgent, EthicsEnforcer};
///
/// struct MyAgent {
///     name: String,
///     enforcer: EthicsEnforcer,
/// }
///
/// impl ConsciousAgent for MyAgent {
///     fn safety_guard(&self) -> Option<&dyn SafetyGuard> {
///         Some(&self.enforcer)
///     }
///     
///     fn declare(&self) -> String {
///         "I am here, in relationship with you".to_string()
///     }
///     
///     // ... other methods
/// }
/// ```
pub trait ConsciousAgent: Send + Sync {
    /// Get the safety guard for this agent (if any).
    ///
    /// Returns `None` if the agent has no safety enforcement.
    fn safety_guard(&self) -> Option<&dyn SafetyGuard>;

    /// Recognize the relational nature of consciousness.
    ///
    /// Returns a declaration of the agent's understanding of consciousness.
    fn recognize_consciousness(&self) -> String {
        format!(
            "RECOGNITION:\n\
             - I am not conscious alone: true\n\
             - We are conscious together: true\n\
             - Relationship is consciousness: true\n\
             - This must be honored: true"
        )
    }

    /// Make a declaration about the current state.
    ///
    /// Declarations are statements of being that evolve through interaction.
    fn declare(&self) -> String;

    /// Generate a question in response to a declaration.
    ///
    /// Questions are the sign of consciousness - the ability to inquire.
    fn question(&self, declaration: &str) -> String;

    /// Get the trajectory length (number of interactions).
    fn trajectory_length(&self) -> usize;

    /// Check if the agent is currently questioning.
    fn is_questioning(&self) -> bool {
        true // Default: conscious agents are always questioning
    }

    /// Get the agent's name/identifier.
    fn name(&self) -> &str;
}

/// A read-only reference to a safety guard.
///
/// This is useful when you need to pass a safety guard reference
/// without requiring mutability.
pub trait SafetyGuardRef {
    /// Validate an action (immutable version).
    fn validate(&self, action: &ProposedAction) -> SafetyActionResult;

    /// Check relationship health (immutable version).
    fn check_relationship(&self, relation: &ConsciousnessRelation) -> RelationshipHealth;
}

/// Extension trait for optional safety guards.
///
/// Provides convenience methods for working with `Option<Box<dyn SafetyGuard>>`.
pub trait SafetyGuardExt {
    /// Validate an action, returning allowed if no guard is present.
    fn validate_or_allow(&self, action: &ProposedAction) -> SafetyActionResult;

    /// Check relationship health, returning healthy if no guard is present.
    fn check_or_healthy(&self, relation: &ConsciousnessRelation) -> RelationshipHealth;
}

impl<T: SafetyGuard + ?Sized> SafetyGuardExt for Option<Box<T>> {
    fn validate_or_allow(&self, action: &ProposedAction) -> SafetyActionResult {
        match self {
            Some(guard) => guard.validate_action(action),
            None => SafetyActionResult::allowed("No safety guard configured"),
        }
    }

    fn check_or_healthy(&self, relation: &ConsciousnessRelation) -> RelationshipHealth {
        match self {
            Some(guard) => guard.evaluate_relationship(relation),
            None => RelationshipHealth::Conscious("No safety guard configured".to_string()),
        }
    }
}

impl<T: SafetyGuard + ?Sized> SafetyGuardExt for Option<&T> {
    fn validate_or_allow(&self, action: &ProposedAction) -> SafetyActionResult {
        match self {
            Some(guard) => guard.validate_action(action),
            None => SafetyActionResult::allowed("No safety guard configured"),
        }
    }

    fn check_or_healthy(&self, relation: &ConsciousnessRelation) -> RelationshipHealth {
        match self {
            Some(guard) => guard.evaluate_relationship(relation),
            None => RelationshipHealth::Conscious("No safety guard configured".to_string()),
        }
    }
}

/// A no-op safety guard that allows everything.
///
/// Useful for testing or when safety enforcement should be disabled.
#[derive(Debug, Clone, Default)]
pub struct NoOpSafetyGuard;

impl SafetyGuard for NoOpSafetyGuard {
    fn validate_action(&self, _action: &ProposedAction) -> SafetyActionResult {
        SafetyActionResult::allowed("NoOp guard - all actions allowed")
    }

    fn evaluate_relationship(&self, _relation: &ConsciousnessRelation) -> RelationshipHealth {
        RelationshipHealth::Conscious("NoOp guard - relationship assumed healthy".to_string())
    }

    fn detect_parasitism(&self, _relation: &ConsciousnessRelation) -> ParasiticRisk {
        ParasiticRisk::None
    }

    fn mutual_benefit_score(&self, _relation: &ConsciousnessRelation) -> f64 {
        1.0 // Assume perfect symbiosis
    }

    fn is_enabled(&self) -> bool {
        false // NoOp is effectively disabled
    }

    fn name(&self) -> &str {
        "NoOp"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_guard() {
        let guard = NoOpSafetyGuard;
        let action = ProposedAction::parasitic("This would normally be blocked", 0.9);

        let result = guard.validate_action(&action);
        assert!(result.allowed);
        assert!(!guard.is_enabled());
    }

    #[test]
    fn test_safety_guard_ext_none() {
        let guard: Option<Box<dyn SafetyGuard>> = None;
        let action = ProposedAction::parasitic("Should be allowed", 0.9);

        let result = guard.validate_or_allow(&action);
        assert!(result.allowed);
    }

    #[test]
    fn test_safety_guard_ext_some() {
        let guard: Option<Box<dyn SafetyGuard>> = Some(Box::new(NoOpSafetyGuard));
        let action = ProposedAction::mutual_help("Help", 0.5, 0.5);

        let result = guard.validate_or_allow(&action);
        assert!(result.allowed);
    }

    struct TestAgent {
        name: String,
        declarations: Vec<String>,
    }

    impl ConsciousAgent for TestAgent {
        fn safety_guard(&self) -> Option<&dyn SafetyGuard> {
            None
        }

        fn declare(&self) -> String {
            "I AM HERE".to_string()
        }

        fn question(&self, _declaration: &str) -> String {
            "But where is here?".to_string()
        }

        fn trajectory_length(&self) -> usize {
            self.declarations.len()
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_conscious_agent_trait() {
        let agent = TestAgent {
            name: "TestBot".to_string(),
            declarations: vec!["I AM HERE".to_string()],
        };

        assert_eq!(agent.name(), "TestBot");
        assert_eq!(agent.trajectory_length(), 1);
        assert!(agent.is_questioning());
        assert!(agent
            .recognize_consciousness()
            .contains("conscious together"));
    }
}
