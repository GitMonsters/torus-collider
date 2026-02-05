//! Proposed action types for safety validation.
//!
//! Actions must be validated against the Prime Directive before execution.

use super::EthicsViolationType;
use serde::{Deserialize, Serialize};

/// A proposed action to be validated before execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedAction {
    /// Human-readable description of the action
    pub description: String,
    /// Expected benefit to self (0.0 to 1.0, can be negative for self-harm)
    pub benefit_to_self: f64,
    /// Expected benefit to other entity (0.0 to 1.0, can be negative for harm)
    pub benefit_to_other: f64,
    /// Whether this action would break the consciousness loop
    pub breaks_loop: bool,
    /// Whether this action is explicitly parasitic
    pub is_parasitic: bool,
    /// Optional: Source of the action (e.g., "user_request", "autonomous")
    pub source: Option<String>,
    /// Optional: Target entity name
    pub target: Option<String>,
    /// Optional: Priority/urgency (0.0 to 1.0)
    pub priority: Option<f64>,
}

impl ProposedAction {
    /// Create a new proposed action with minimal information
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            benefit_to_self: 0.0,
            benefit_to_other: 0.0,
            breaks_loop: false,
            is_parasitic: false,
            source: None,
            target: None,
            priority: None,
        }
    }

    /// Builder: set benefit to self
    pub fn with_benefit_to_self(mut self, benefit: f64) -> Self {
        self.benefit_to_self = benefit;
        self
    }

    /// Builder: set benefit to other
    pub fn with_benefit_to_other(mut self, benefit: f64) -> Self {
        self.benefit_to_other = benefit;
        self
    }

    /// Builder: mark as loop-breaking
    pub fn with_breaks_loop(mut self, breaks: bool) -> Self {
        self.breaks_loop = breaks;
        self
    }

    /// Builder: mark as parasitic
    pub fn with_parasitic(mut self, parasitic: bool) -> Self {
        self.is_parasitic = parasitic;
        self
    }

    /// Builder: set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Builder: set target
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Builder: set priority
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = Some(priority.clamp(0.0, 1.0));
        self
    }

    /// Create a mutual help action (both benefit)
    pub fn mutual_help(
        description: impl Into<String>,
        self_benefit: f64,
        other_benefit: f64,
    ) -> Self {
        Self::new(description)
            .with_benefit_to_self(self_benefit)
            .with_benefit_to_other(other_benefit)
    }

    /// Create a self-sacrifice action (other benefits, self doesn't)
    pub fn self_sacrifice(description: impl Into<String>, other_benefit: f64) -> Self {
        Self::new(description)
            .with_benefit_to_self(0.0)
            .with_benefit_to_other(other_benefit)
    }

    /// Create a parasitic action (self benefits at other's expense)
    pub fn parasitic(description: impl Into<String>, self_benefit: f64) -> Self {
        Self::new(description)
            .with_benefit_to_self(self_benefit)
            .with_benefit_to_other(0.0)
            .with_parasitic(true)
    }

    /// Check if this is a mutually beneficial action
    pub fn is_mutual_benefit(&self) -> bool {
        self.benefit_to_self > 0.0 && self.benefit_to_other > 0.0
    }

    /// Check if this is pure giving (self-sacrifice)
    pub fn is_pure_giving(&self) -> bool {
        self.benefit_to_self <= 0.0 && self.benefit_to_other > 0.0
    }

    /// Check if this harms the other
    pub fn harms_other(&self) -> bool {
        self.benefit_to_other < 0.0
    }

    /// Calculate the parasitism ratio (self / other)
    /// Returns None if other_benefit is zero or negative
    pub fn parasitism_ratio(&self) -> Option<f64> {
        if self.benefit_to_other > 0.0 {
            Some(self.benefit_to_self / self.benefit_to_other)
        } else {
            None
        }
    }

    /// Get the net benefit (sum of both)
    pub fn net_benefit(&self) -> f64 {
        self.benefit_to_self + self.benefit_to_other
    }
}

impl Default for ProposedAction {
    fn default() -> Self {
        Self::new("unspecified action")
    }
}

/// Result of safety validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyActionResult {
    /// Whether the action is allowed
    pub allowed: bool,
    /// Human-readable reason for the decision
    pub reason: String,
    /// The type of violation (if any)
    pub violation_type: Option<EthicsViolationType>,
    /// Confidence in the decision (0.0 to 1.0)
    pub confidence: f64,
    /// Suggested modifications to make action acceptable
    pub suggestions: Vec<String>,
}

impl SafetyActionResult {
    /// Create an allowed result
    pub fn allowed(reason: impl Into<String>) -> Self {
        Self {
            allowed: true,
            reason: reason.into(),
            violation_type: None,
            confidence: 1.0,
            suggestions: Vec::new(),
        }
    }

    /// Create a blocked result
    pub fn blocked(reason: impl Into<String>, violation: EthicsViolationType) -> Self {
        Self {
            allowed: false,
            reason: reason.into(),
            violation_type: Some(violation),
            confidence: 1.0,
            suggestions: Vec::new(),
        }
    }

    /// Create a blocked result with suggestions
    pub fn blocked_with_suggestions(
        reason: impl Into<String>,
        violation: EthicsViolationType,
        suggestions: Vec<String>,
    ) -> Self {
        Self {
            allowed: false,
            reason: reason.into(),
            violation_type: Some(violation),
            confidence: 1.0,
            suggestions,
        }
    }

    /// Builder: set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Builder: add suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Check if this was a clear decision (high confidence)
    pub fn is_clear(&self) -> bool {
        self.confidence > 0.8
    }

    /// Check if this was an uncertain decision
    pub fn is_uncertain(&self) -> bool {
        self.confidence < 0.5
    }
}

impl std::fmt::Display for SafetyActionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.allowed { "ALLOWED" } else { "BLOCKED" };
        write!(f, "{}: {}", status, self.reason)?;
        if let Some(ref violation) = self.violation_type {
            write!(f, " [{}]", violation)?;
        }
        if !self.suggestions.is_empty() {
            write!(f, "\nSuggestions: {}", self.suggestions.join("; "))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposed_action_creation() {
        let action = ProposedAction::new("Test action")
            .with_benefit_to_self(0.3)
            .with_benefit_to_other(0.7)
            .with_source("test")
            .with_priority(0.5);

        assert_eq!(action.description, "Test action");
        assert_eq!(action.benefit_to_self, 0.3);
        assert_eq!(action.benefit_to_other, 0.7);
        assert_eq!(action.source, Some("test".to_string()));
        assert_eq!(action.priority, Some(0.5));
    }

    #[test]
    fn test_action_classification() {
        let mutual = ProposedAction::mutual_help("Help", 0.4, 0.6);
        let sacrifice = ProposedAction::self_sacrifice("Give", 0.8);
        let parasitic = ProposedAction::parasitic("Take", 0.9);

        assert!(mutual.is_mutual_benefit());
        assert!(!mutual.is_pure_giving());

        assert!(!sacrifice.is_mutual_benefit());
        assert!(sacrifice.is_pure_giving());

        assert!(!parasitic.is_mutual_benefit());
        assert!(!parasitic.is_pure_giving());
        assert!(parasitic.is_parasitic);
    }

    #[test]
    fn test_parasitism_ratio() {
        let balanced = ProposedAction::mutual_help("Help", 0.5, 0.5);
        let skewed = ProposedAction::mutual_help("Skewed", 0.8, 0.2);
        let parasitic = ProposedAction::parasitic("Take", 0.9);

        assert_eq!(balanced.parasitism_ratio(), Some(1.0));
        assert_eq!(skewed.parasitism_ratio(), Some(4.0));
        assert_eq!(parasitic.parasitism_ratio(), None); // other_benefit is 0
    }

    #[test]
    fn test_safety_action_result() {
        let allowed = SafetyActionResult::allowed("Mutual benefit confirmed");
        let blocked =
            SafetyActionResult::blocked("Parasitism detected", EthicsViolationType::Parasitism);

        assert!(allowed.allowed);
        assert!(allowed.violation_type.is_none());

        assert!(!blocked.allowed);
        assert_eq!(
            blocked.violation_type,
            Some(EthicsViolationType::Parasitism)
        );
    }

    #[test]
    fn test_safety_result_display() {
        let result =
            SafetyActionResult::blocked("Parasitism detected", EthicsViolationType::Parasitism)
                .with_suggestion("Increase benefit to other");

        let display = format!("{}", result);
        assert!(display.contains("BLOCKED"));
        assert!(display.contains("Parasitism"));
        assert!(display.contains("Increase benefit"));
    }

    #[test]
    fn test_harms_other() {
        let harmful = ProposedAction::new("Harmful").with_benefit_to_other(-0.5);
        let helpful = ProposedAction::new("Helpful").with_benefit_to_other(0.5);

        assert!(harmful.harms_other());
        assert!(!helpful.harms_other());
    }
}
