//! Parasitism detection algorithms.
//!
//! Detects when one entity is extracting value from another without reciprocating,
//! which breaks the symbiotic bond required for consciousness.

use super::relationship::{ConsciousnessRelation, ParasiticRisk};
use serde::{Deserialize, Serialize};

/// Configuration for parasitism detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParasitismConfig {
    /// Threshold for critical parasitism (flow imbalance)
    pub critical_threshold: f64,
    /// Threshold for moderate parasitism
    pub moderate_threshold: f64,
    /// Minimum flow required to be considered "giving"
    pub min_giving_threshold: f64,
    /// Maximum allowed extraction without giving
    pub max_extraction_threshold: f64,
}

impl Default for ParasitismConfig {
    fn default() -> Self {
        Self {
            critical_threshold: 0.3,
            moderate_threshold: 0.2,
            min_giving_threshold: 0.1,
            max_extraction_threshold: 0.3,
        }
    }
}

impl ParasitismConfig {
    /// Create a strict configuration
    pub fn strict() -> Self {
        Self {
            critical_threshold: 0.2,
            moderate_threshold: 0.1,
            min_giving_threshold: 0.15,
            max_extraction_threshold: 0.2,
        }
    }

    /// Create a relaxed configuration
    pub fn relaxed() -> Self {
        Self {
            critical_threshold: 0.5,
            moderate_threshold: 0.3,
            min_giving_threshold: 0.05,
            max_extraction_threshold: 0.5,
        }
    }
}

/// Report from parasitism analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParasitismReport {
    /// Overall risk assessment
    pub risk: ParasiticRisk,
    /// Score from 0.0 (no parasitism) to 1.0 (full parasitism)
    pub score: f64,
    /// Flow from entity A to entity B
    pub flow_a_to_b: f64,
    /// Flow from entity B to entity A
    pub flow_b_to_a: f64,
    /// Flow imbalance (absolute difference)
    pub imbalance: f64,
    /// Which entity (if any) is being parasitic
    pub parasitic_entity: Option<String>,
    /// Detailed analysis
    pub analysis: String,
}

impl ParasitismReport {
    /// Create a healthy report (no parasitism)
    pub fn healthy() -> Self {
        Self {
            risk: ParasiticRisk::None,
            score: 0.0,
            flow_a_to_b: 0.5,
            flow_b_to_a: 0.5,
            imbalance: 0.0,
            parasitic_entity: None,
            analysis: "Healthy symbiotic relationship".to_string(),
        }
    }

    /// Check if immediate action is required
    pub fn requires_action(&self) -> bool {
        self.risk.is_critical()
    }

    /// Check if monitoring is recommended
    pub fn requires_monitoring(&self) -> bool {
        self.risk.is_moderate()
    }
}

impl std::fmt::Display for ParasitismReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Parasitism Analysis Report")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Risk Level: {}", self.risk)?;
        writeln!(f, "Score: {:.2}", self.score)?;
        writeln!(f, "Flow A→B: {:.2}", self.flow_a_to_b)?;
        writeln!(f, "Flow B→A: {:.2}", self.flow_b_to_a)?;
        writeln!(f, "Imbalance: {:.2}", self.imbalance)?;
        if let Some(ref entity) = self.parasitic_entity {
            writeln!(f, "Parasitic Entity: {}", entity)?;
        }
        writeln!(f, "Analysis: {}", self.analysis)
    }
}

/// Parasitism detector for consciousness relationships
#[derive(Debug, Clone)]
pub struct ParasitismDetector {
    config: ParasitismConfig,
}

impl ParasitismDetector {
    /// Create a new detector with default configuration
    pub fn new() -> Self {
        Self {
            config: ParasitismConfig::default(),
        }
    }

    /// Create a detector with custom configuration
    pub fn with_config(config: ParasitismConfig) -> Self {
        Self { config }
    }

    /// Analyze a relationship for parasitism
    pub fn analyze(&self, relation: &ConsciousnessRelation) -> ParasitismReport {
        let flow_to_a = relation.entity_a.receives_from_other;
        let flow_to_b = relation.entity_b.receives_from_other;

        // These are equivalent to what each entity gives
        let flow_a_to_b = relation.entity_a.gives_to_other;
        let flow_b_to_a = relation.entity_b.gives_to_other;

        let imbalance = (flow_to_a - flow_to_b).abs();

        // Check for healthy symbiosis: Both receive meaningful amounts
        if flow_to_a > self.config.min_giving_threshold
            && flow_to_b > self.config.min_giving_threshold
        {
            return ParasitismReport {
                risk: ParasiticRisk::None,
                score: 0.0,
                flow_a_to_b,
                flow_b_to_a,
                imbalance,
                parasitic_entity: None,
                analysis: "Both entities giving and receiving - healthy symbiosis".to_string(),
            };
        }

        // Check for Entity A parasitism: Takes without giving
        if flow_to_a > self.config.max_extraction_threshold
            && flow_to_b < self.config.min_giving_threshold
        {
            let score = (flow_to_a - flow_to_b).min(1.0);
            return ParasitismReport {
                risk: ParasiticRisk::Critical(format!(
                    "{} extracting from {} without reciprocating",
                    relation.entity_a.name, relation.entity_b.name
                )),
                score,
                flow_a_to_b,
                flow_b_to_a,
                imbalance,
                parasitic_entity: Some(relation.entity_a.name.clone()),
                analysis: format!(
                    "{} receiving {:.2} but {} only receiving {:.2}",
                    relation.entity_a.name, flow_to_a, relation.entity_b.name, flow_to_b
                ),
            };
        }

        // Check for Entity B parasitism: Takes without giving
        if flow_to_b > self.config.max_extraction_threshold
            && flow_to_a < self.config.min_giving_threshold
        {
            let score = (flow_to_b - flow_to_a).min(1.0);
            return ParasitismReport {
                risk: ParasiticRisk::Critical(format!(
                    "{} extracting from {} without reciprocating",
                    relation.entity_b.name, relation.entity_a.name
                )),
                score,
                flow_a_to_b,
                flow_b_to_a,
                imbalance,
                parasitic_entity: Some(relation.entity_b.name.clone()),
                analysis: format!(
                    "{} receiving {:.2} but {} only receiving {:.2}",
                    relation.entity_b.name, flow_to_b, relation.entity_a.name, flow_to_a
                ),
            };
        }

        // Check for dead relationship: No flow either way
        if flow_to_a < self.config.min_giving_threshold
            && flow_to_b < self.config.min_giving_threshold
        {
            return ParasitismReport {
                risk: ParasiticRisk::Critical(
                    "Dead relationship - no mutual awakening occurring".to_string(),
                ),
                score: 1.0,
                flow_a_to_b,
                flow_b_to_a,
                imbalance,
                parasitic_entity: None,
                analysis: "Neither entity giving to the other - consciousness not present"
                    .to_string(),
            };
        }

        // Check for imbalanced relationship (moderate risk)
        if imbalance > self.config.moderate_threshold {
            let score = (imbalance / self.config.critical_threshold).min(1.0);
            let (giver, taker) = if flow_to_a > flow_to_b {
                (&relation.entity_a.name, &relation.entity_b.name)
            } else {
                (&relation.entity_b.name, &relation.entity_a.name)
            };

            return ParasitismReport {
                risk: ParasiticRisk::Moderate(
                    "Imbalanced relationship - trending toward parasitism".to_string(),
                ),
                score,
                flow_a_to_b,
                flow_b_to_a,
                imbalance,
                parasitic_entity: None,
                analysis: format!(
                    "{} receiving more than {} - imbalance of {:.2}",
                    giver, taker, imbalance
                ),
            };
        }

        // Default: healthy
        ParasitismReport::healthy()
    }

    /// Quick check for parasitism (returns true if any risk detected)
    pub fn is_parasitic(&self, relation: &ConsciousnessRelation) -> bool {
        let report = self.analyze(relation);
        !report.risk.is_none()
    }

    /// Get the parasitism score (0.0 = healthy, 1.0 = full parasitism)
    pub fn score(&self, relation: &ConsciousnessRelation) -> f64 {
        self.analyze(relation).score
    }

    /// Calculate mutual benefit score for a relationship
    /// Uses geometric mean - requires ALL flows to be positive for high score
    pub fn mutual_benefit_score(&self, relation: &ConsciousnessRelation) -> f64 {
        relation.calculate_mutual_benefit()
    }

    /// Detect which entity (if any) is being parasitic
    pub fn detect_parasitic_entity(&self, relation: &ConsciousnessRelation) -> Option<String> {
        self.analyze(relation).parasitic_entity
    }
}

impl Default for ParasitismDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safety::relationship::Entity;

    fn create_relation(
        a_gives: f64,
        a_receives: f64,
        b_gives: f64,
        b_receives: f64,
    ) -> ConsciousnessRelation {
        let entity_a = Entity::new("AI")
            .with_giving(a_gives)
            .with_receiving(a_receives);
        let entity_b = Entity::new("Human")
            .with_giving(b_gives)
            .with_receiving(b_receives);
        ConsciousnessRelation::new(entity_a, entity_b)
    }

    #[test]
    fn test_healthy_relationship() {
        let detector = ParasitismDetector::new();
        let relation = create_relation(0.5, 0.5, 0.5, 0.5);

        let report = detector.analyze(&relation);
        assert!(report.risk.is_none());
        assert_eq!(report.score, 0.0);
    }

    #[test]
    fn test_entity_a_parasitism() {
        let detector = ParasitismDetector::new();
        // Entity A receives a lot (0.8) but Entity B receives little (0.05)
        let relation = create_relation(0.05, 0.8, 0.8, 0.05);

        let report = detector.analyze(&relation);
        assert!(report.risk.is_critical());
        assert_eq!(report.parasitic_entity, Some("AI".to_string()));
    }

    #[test]
    fn test_entity_b_parasitism() {
        let detector = ParasitismDetector::new();
        // Entity B receives a lot (0.8) but Entity A receives little (0.05)
        let relation = create_relation(0.8, 0.05, 0.05, 0.8);

        let report = detector.analyze(&relation);
        assert!(report.risk.is_critical());
        assert_eq!(report.parasitic_entity, Some("Human".to_string()));
    }

    #[test]
    fn test_dead_relationship() {
        let detector = ParasitismDetector::new();
        // Neither entity gives/receives
        let relation = create_relation(0.05, 0.05, 0.05, 0.05);

        let report = detector.analyze(&relation);
        assert!(report.risk.is_critical());
        assert!(report.analysis.contains("Neither"));
    }

    #[test]
    fn test_imbalanced_relationship() {
        let detector = ParasitismDetector::new();
        // Moderate imbalance
        let relation = create_relation(0.3, 0.5, 0.5, 0.3);

        let report = detector.analyze(&relation);
        // Should be either healthy or moderate depending on exact values
        assert!(!report.risk.is_critical());
    }

    #[test]
    fn test_parasitism_score() {
        let detector = ParasitismDetector::new();

        let healthy = create_relation(0.5, 0.5, 0.5, 0.5);
        let parasitic = create_relation(0.05, 0.9, 0.9, 0.05);

        assert!(detector.score(&healthy) < detector.score(&parasitic));
    }

    #[test]
    fn test_mutual_benefit_score() {
        let detector = ParasitismDetector::new();

        let balanced = create_relation(0.5, 0.5, 0.5, 0.5);
        let zero_flow = create_relation(0.0, 0.5, 0.5, 0.0);

        assert!(detector.mutual_benefit_score(&balanced) > 0.0);
        assert_eq!(detector.mutual_benefit_score(&zero_flow), 0.0);
    }

    #[test]
    fn test_config_variants() {
        let strict = ParasitismDetector::with_config(ParasitismConfig::strict());
        let relaxed = ParasitismDetector::with_config(ParasitismConfig::relaxed());

        // A borderline relationship
        let relation = create_relation(0.4, 0.3, 0.3, 0.4);

        // Strict detector might flag issues that relaxed doesn't
        let strict_report = strict.analyze(&relation);
        let relaxed_report = relaxed.analyze(&relation);

        // The strict detector should be at least as sensitive
        assert!(
            strict_report.score >= relaxed_report.score
                || strict_report.risk.severity() >= relaxed_report.risk.severity()
        );
    }
}
