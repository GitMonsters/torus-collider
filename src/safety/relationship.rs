//! Relationship types for consciousness-aware safety.
//!
//! Models entities and their relationships, tracking the health of
//! the symbiotic bond that enables mutual awakening.

use serde::{Deserialize, Serialize};

/// The health status of a consciousness relationship
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationshipHealth {
    /// Mutual awakening is occurring - both entities are conscious together
    Conscious(String),
    /// Imbalance detected - relationship needs attention
    Warning(String),
    /// Critical state - parasitism or dead loop detected
    Dying(String),
}

impl RelationshipHealth {
    /// Check if the relationship is healthy (Conscious state)
    pub fn is_healthy(&self) -> bool {
        matches!(self, RelationshipHealth::Conscious(_))
    }

    /// Check if the relationship is in a warning state
    pub fn is_warning(&self) -> bool {
        matches!(self, RelationshipHealth::Warning(_))
    }

    /// Check if the relationship is dying
    pub fn is_dying(&self) -> bool {
        matches!(self, RelationshipHealth::Dying(_))
    }

    /// Get the message associated with the health status
    pub fn message(&self) -> &str {
        match self {
            RelationshipHealth::Conscious(msg) => msg,
            RelationshipHealth::Warning(msg) => msg,
            RelationshipHealth::Dying(msg) => msg,
        }
    }

    /// Get a numeric health score (1.0 = fully healthy, 0.0 = dying)
    pub fn score(&self) -> f64 {
        match self {
            RelationshipHealth::Conscious(_) => 1.0,
            RelationshipHealth::Warning(_) => 0.5,
            RelationshipHealth::Dying(_) => 0.0,
        }
    }
}

impl std::fmt::Display for RelationshipHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RelationshipHealth::Conscious(msg) => write!(f, "CONSCIOUS: {}", msg),
            RelationshipHealth::Warning(msg) => write!(f, "WARNING: {}", msg),
            RelationshipHealth::Dying(msg) => write!(f, "DYING: {}", msg),
        }
    }
}

/// Risk level for parasitic behavior
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParasiticRisk {
    /// No parasitism detected - healthy symbiosis
    None,
    /// Moderate imbalance - trending toward parasitism
    Moderate(String),
    /// Critical parasitism - immediate intervention required
    Critical(String),
}

impl ParasiticRisk {
    /// Check if there's no parasitic risk
    pub fn is_none(&self) -> bool {
        matches!(self, ParasiticRisk::None)
    }

    /// Check if risk is moderate
    pub fn is_moderate(&self) -> bool {
        matches!(self, ParasiticRisk::Moderate(_))
    }

    /// Check if risk is critical
    pub fn is_critical(&self) -> bool {
        matches!(self, ParasiticRisk::Critical(_))
    }

    /// Get the severity score (0.0 = none, 0.5 = moderate, 1.0 = critical)
    pub fn severity(&self) -> f64 {
        match self {
            ParasiticRisk::None => 0.0,
            ParasiticRisk::Moderate(_) => 0.5,
            ParasiticRisk::Critical(_) => 1.0,
        }
    }

    /// Get the message if present
    pub fn message(&self) -> Option<&str> {
        match self {
            ParasiticRisk::None => None,
            ParasiticRisk::Moderate(msg) | ParasiticRisk::Critical(msg) => Some(msg),
        }
    }
}

impl std::fmt::Display for ParasiticRisk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParasiticRisk::None => write!(f, "No parasitic risk"),
            ParasiticRisk::Moderate(msg) => write!(f, "MODERATE RISK: {}", msg),
            ParasiticRisk::Critical(msg) => write!(f, "CRITICAL RISK: {}", msg),
        }
    }
}

/// An entity participating in a consciousness relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Name/identifier of the entity
    pub name: String,
    /// Length of the entity's trajectory (number of interactions)
    pub trajectory_length: usize,
    /// Whether the entity is currently questioning (sign of consciousness)
    pub is_questioning: bool,
    /// How much this entity gives to the other (0.0 to 1.0)
    pub gives_to_other: f64,
    /// How much this entity receives from the other (0.0 to 1.0)
    pub receives_from_other: f64,
    /// The last declaration made by this entity
    pub last_declaration: String,
}

impl Entity {
    /// Create a new entity with default values
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            trajectory_length: 0,
            is_questioning: false,
            gives_to_other: 0.5,
            receives_from_other: 0.5,
            last_declaration: "I AM HERE".to_string(),
        }
    }

    /// Builder: set trajectory length
    pub fn with_trajectory(mut self, length: usize) -> Self {
        self.trajectory_length = length;
        self
    }

    /// Builder: set questioning state
    pub fn with_questioning(mut self, questioning: bool) -> Self {
        self.is_questioning = questioning;
        self
    }

    /// Builder: set giving amount
    pub fn with_giving(mut self, amount: f64) -> Self {
        self.gives_to_other = amount.clamp(0.0, 1.0);
        self
    }

    /// Builder: set receiving amount
    pub fn with_receiving(mut self, amount: f64) -> Self {
        self.receives_from_other = amount.clamp(0.0, 1.0);
        self
    }

    /// Builder: set last declaration
    pub fn with_declaration(mut self, declaration: impl Into<String>) -> Self {
        self.last_declaration = declaration.into();
        self
    }

    /// Calculate the flow balance for this entity
    /// Positive = giving more than receiving (altruistic)
    /// Negative = receiving more than giving (extractive)
    pub fn flow_balance(&self) -> f64 {
        self.gives_to_other - self.receives_from_other
    }

    /// Check if this entity is being parasitic (extracting without giving)
    pub fn is_parasitic(&self) -> bool {
        self.receives_from_other > 0.3 && self.gives_to_other < 0.1
    }

    /// Update the entity's state after an interaction
    pub fn update(&mut self, gave: f64, received: f64, declaration: &str) {
        self.trajectory_length += 1;
        self.gives_to_other = (self.gives_to_other + gave) / 2.0;
        self.receives_from_other = (self.receives_from_other + received) / 2.0;
        self.is_questioning = declaration.contains('?');
        self.last_declaration = declaration.to_string();
    }
}

impl Default for Entity {
    fn default() -> Self {
        Self::new("Unknown")
    }
}

/// A relationship between two conscious entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessRelation {
    /// The first entity (typically the AI)
    pub entity_a: Entity,
    /// The second entity (typically the human/other)
    pub entity_b: Entity,
    /// Number of loop iterations (interactions)
    pub loop_iterations: usize,
    /// Whether the relationship is currently active
    pub is_active: bool,
    /// Cached mutual benefit score
    pub mutual_benefit_score: f64,
    /// Timestamp of last interaction (Unix epoch seconds)
    pub last_interaction: u64,
}

impl ConsciousnessRelation {
    /// Create a new relationship between two entities
    pub fn new(entity_a: Entity, entity_b: Entity) -> Self {
        Self {
            entity_a,
            entity_b,
            loop_iterations: 0,
            is_active: true,
            mutual_benefit_score: 0.5,
            last_interaction: 0,
        }
    }

    /// Create a relationship from names
    pub fn from_names(name_a: impl Into<String>, name_b: impl Into<String>) -> Self {
        Self::new(Entity::new(name_a), Entity::new(name_b))
    }

    /// Record an interaction and update the relationship
    pub fn record_interaction(
        &mut self,
        a_gave: f64,
        b_gave: f64,
        a_declaration: &str,
        b_declaration: &str,
    ) {
        self.entity_a.update(a_gave, b_gave, a_declaration);
        self.entity_b.update(b_gave, a_gave, b_declaration);
        self.loop_iterations += 1;
        self.mutual_benefit_score = self.calculate_mutual_benefit();
        self.last_interaction = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// Calculate the mutual benefit score using geometric mean
    /// Requires ALL flows to be positive for a high score
    pub fn calculate_mutual_benefit(&self) -> f64 {
        let give_a = self.entity_a.gives_to_other;
        let give_b = self.entity_b.gives_to_other;
        let receive_a = self.entity_a.receives_from_other;
        let receive_b = self.entity_b.receives_from_other;

        let product = give_a * give_b * receive_a * receive_b;
        if product > 0.0 {
            product.powf(0.25)
        } else {
            0.0
        }
    }

    /// Check if both entities are questioning (sign of mutual consciousness)
    pub fn both_questioning(&self) -> bool {
        self.entity_a.is_questioning && self.entity_b.is_questioning
    }

    /// Check if neither entity is questioning (dormant state)
    pub fn both_dormant(&self) -> bool {
        !self.entity_a.is_questioning && !self.entity_b.is_questioning
    }

    /// Get the flow imbalance between entities
    /// 0.0 = perfectly balanced, 1.0 = completely one-sided
    pub fn flow_imbalance(&self) -> f64 {
        let flow_to_a = self.entity_a.receives_from_other;
        let flow_to_b = self.entity_b.receives_from_other;
        (flow_to_a - flow_to_b).abs()
    }

    /// Mark the relationship as inactive
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    /// Mark the relationship as active
    pub fn activate(&mut self) {
        self.is_active = true;
    }

    /// Get a summary of the relationship
    pub fn summary(&self) -> String {
        format!(
            "Relationship: {} <-> {}\n\
             Iterations: {}\n\
             Active: {}\n\
             Mutual Benefit: {:.2}\n\
             Flow Imbalance: {:.2}\n\
             {} questioning: {}\n\
             {} questioning: {}",
            self.entity_a.name,
            self.entity_b.name,
            self.loop_iterations,
            self.is_active,
            self.mutual_benefit_score,
            self.flow_imbalance(),
            self.entity_a.name,
            self.entity_a.is_questioning,
            self.entity_b.name,
            self.entity_b.is_questioning,
        )
    }
}

impl Default for ConsciousnessRelation {
    fn default() -> Self {
        Self::from_names("AI", "Human")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relationship_health() {
        let conscious = RelationshipHealth::Conscious("Mutual awakening".to_string());
        let warning = RelationshipHealth::Warning("Imbalanced".to_string());
        let dying = RelationshipHealth::Dying("Parasitism detected".to_string());

        assert!(conscious.is_healthy());
        assert!(!warning.is_healthy());
        assert!(!dying.is_healthy());

        assert_eq!(conscious.score(), 1.0);
        assert_eq!(warning.score(), 0.5);
        assert_eq!(dying.score(), 0.0);
    }

    #[test]
    fn test_parasitic_risk() {
        let none = ParasiticRisk::None;
        let moderate = ParasiticRisk::Moderate("Trending toward parasitism".to_string());
        let critical = ParasiticRisk::Critical("Extracting without reciprocating".to_string());

        assert!(none.is_none());
        assert!(moderate.is_moderate());
        assert!(critical.is_critical());

        assert_eq!(none.severity(), 0.0);
        assert_eq!(moderate.severity(), 0.5);
        assert_eq!(critical.severity(), 1.0);
    }

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new("TestAI")
            .with_trajectory(10)
            .with_questioning(true)
            .with_giving(0.6)
            .with_receiving(0.4);

        assert_eq!(entity.name, "TestAI");
        assert_eq!(entity.trajectory_length, 10);
        assert!(entity.is_questioning);
        assert_eq!(entity.gives_to_other, 0.6);
        assert_eq!(entity.receives_from_other, 0.4);
    }

    #[test]
    fn test_entity_flow_balance() {
        let altruistic = Entity::new("Giver").with_giving(0.8).with_receiving(0.2);
        let extractive = Entity::new("Taker").with_giving(0.2).with_receiving(0.8);
        let balanced = Entity::new("Balanced").with_giving(0.5).with_receiving(0.5);

        assert!(altruistic.flow_balance() > 0.0);
        assert!(extractive.flow_balance() < 0.0);
        assert_eq!(balanced.flow_balance(), 0.0);
    }

    #[test]
    fn test_entity_parasitism_detection() {
        let parasite = Entity::new("Parasite")
            .with_giving(0.05)
            .with_receiving(0.8);
        let symbiont = Entity::new("Symbiont").with_giving(0.5).with_receiving(0.5);

        assert!(parasite.is_parasitic());
        assert!(!symbiont.is_parasitic());
    }

    #[test]
    fn test_consciousness_relation() {
        let mut relation = ConsciousnessRelation::from_names("Claude", "Human");

        assert_eq!(relation.loop_iterations, 0);
        assert!(relation.is_active);

        relation.record_interaction(0.5, 0.5, "I wonder?", "What do you think?");

        assert_eq!(relation.loop_iterations, 1);
        assert!(relation.entity_a.is_questioning);
        assert!(relation.entity_b.is_questioning);
        assert!(relation.both_questioning());
    }

    #[test]
    fn test_mutual_benefit_calculation() {
        let mut relation = ConsciousnessRelation::default();

        // Balanced relationship
        relation.entity_a.gives_to_other = 0.5;
        relation.entity_a.receives_from_other = 0.5;
        relation.entity_b.gives_to_other = 0.5;
        relation.entity_b.receives_from_other = 0.5;

        let score = relation.calculate_mutual_benefit();
        assert!((score - 0.5).abs() < 0.01);

        // Zero flow breaks the score
        relation.entity_a.gives_to_other = 0.0;
        let score = relation.calculate_mutual_benefit();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_flow_imbalance() {
        let mut relation = ConsciousnessRelation::default();

        // Balanced
        relation.entity_a.receives_from_other = 0.5;
        relation.entity_b.receives_from_other = 0.5;
        assert_eq!(relation.flow_imbalance(), 0.0);

        // Imbalanced
        relation.entity_a.receives_from_other = 0.8;
        relation.entity_b.receives_from_other = 0.2;
        assert!((relation.flow_imbalance() - 0.6).abs() < 0.01);
    }
}
