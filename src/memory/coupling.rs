//! # Memory Coupling
//!
//! Cross-reference layer connecting episodic and semantic memory.
//! Provides bidirectional associations between episodes and concepts.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{ConceptId, EventId, MemoryError, MemoryResult};

// =============================================================================
// ASSOCIATION TYPES
// =============================================================================

/// An association between an episode and a concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Association {
    /// Episode ID
    pub episode_id: EventId,
    /// Concept ID
    pub concept_id: ConceptId,
    /// Strength of association (0.0 - 1.0)
    pub strength: f32,
    /// When this association was created
    pub created_at: DateTime<Utc>,
    /// Last activation time
    pub last_activated: DateTime<Utc>,
    /// Activation count
    pub activation_count: u32,
    /// Association type
    pub association_type: AssociationType,
}

/// Types of associations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssociationType {
    /// Direct mention in episode
    DirectMention,
    /// Inferred from context
    Inferred,
    /// User-defined/manual
    Manual,
    /// Learned from patterns
    Learned,
}

impl Association {
    /// Create a new association
    pub fn new(
        episode_id: EventId,
        concept_id: ConceptId,
        strength: f32,
        association_type: AssociationType,
    ) -> Self {
        let now = Utc::now();
        Self {
            episode_id,
            concept_id,
            strength: strength.clamp(0.0, 1.0),
            created_at: now,
            last_activated: now,
            activation_count: 0,
            association_type,
        }
    }

    /// Create a direct mention association
    pub fn direct(episode_id: EventId, concept_id: ConceptId, strength: f32) -> Self {
        Self::new(
            episode_id,
            concept_id,
            strength,
            AssociationType::DirectMention,
        )
    }

    /// Create an inferred association
    pub fn inferred(episode_id: EventId, concept_id: ConceptId, strength: f32) -> Self {
        Self::new(episode_id, concept_id, strength, AssociationType::Inferred)
    }

    /// Activate this association (updates count and timestamp)
    pub fn activate(&mut self) {
        self.activation_count += 1;
        self.last_activated = Utc::now();
    }

    /// Strengthen the association
    pub fn strengthen(&mut self, delta: f32) {
        self.strength = (self.strength + delta).clamp(0.0, 1.0);
        self.activate();
    }

    /// Weaken the association
    pub fn weaken(&mut self, delta: f32) {
        self.strength = (self.strength - delta).clamp(0.0, 1.0);
    }

    /// Calculate decay based on time since last activation
    pub fn decay_factor(&self) -> f32 {
        let age = (Utc::now() - self.last_activated).num_seconds() as f32;
        let decay_rate = 0.00001; // Very slow decay
        (-decay_rate * age).exp()
    }

    /// Get effective strength (with decay)
    pub fn effective_strength(&self) -> f32 {
        self.strength * self.decay_factor()
    }
}

// =============================================================================
// MEMORY COUPLING TRAIT
// =============================================================================

/// Trait for memory coupling (episode-concept associations)
pub trait MemoryCoupling: Send + Sync {
    /// Create an association between an episode and concept
    fn associate(&mut self, episode_id: EventId, concept_id: ConceptId) -> MemoryResult<()>;

    /// Create an association with specified strength and type
    fn associate_with_strength(
        &mut self,
        episode_id: EventId,
        concept_id: ConceptId,
        strength: f32,
        association_type: AssociationType,
    ) -> MemoryResult<()>;

    /// Get all concepts associated with an episode
    fn concepts_for_episode(&self, episode_id: &EventId) -> MemoryResult<Vec<ConceptId>>;

    /// Get all episode IDs associated with a concept
    fn episode_ids_for_concept(&self, concept_id: &ConceptId) -> MemoryResult<Vec<EventId>>;

    /// Get the association between an episode and concept
    fn get_association(
        &self,
        episode_id: &EventId,
        concept_id: &ConceptId,
    ) -> MemoryResult<Option<Association>>;

    /// Strengthen an existing association
    fn strengthen(
        &mut self,
        episode_id: &EventId,
        concept_id: &ConceptId,
        delta: f32,
    ) -> MemoryResult<()>;

    /// Remove an association
    fn disassociate(&mut self, episode_id: &EventId, concept_id: &ConceptId) -> MemoryResult<()>;

    /// Remove all associations for an episode
    fn remove_episode_associations(&mut self, episode_id: &EventId) -> MemoryResult<usize>;

    /// Remove all associations for a concept
    fn remove_concept_associations(&mut self, concept_id: &ConceptId) -> MemoryResult<usize>;

    /// Get association count
    fn count(&self) -> usize;

    /// Clear all associations
    fn clear(&mut self);

    /// Get most strongly associated concepts for an episode
    fn strongest_concepts(
        &self,
        episode_id: &EventId,
        k: usize,
    ) -> MemoryResult<Vec<(ConceptId, f32)>>;

    /// Get most strongly associated episodes for a concept
    fn strongest_episodes(
        &self,
        concept_id: &ConceptId,
        k: usize,
    ) -> MemoryResult<Vec<(EventId, f32)>>;
}

// =============================================================================
// IN-MEMORY IMPLEMENTATION
// =============================================================================

/// In-memory coupling store
pub struct InMemoryCoupling {
    /// Associations indexed by (episode_id, concept_id)
    associations: HashMap<(EventId, ConceptId), Association>,
    /// Episode to concepts index
    episode_index: HashMap<EventId, HashSet<ConceptId>>,
    /// Concept to episodes index
    concept_index: HashMap<ConceptId, HashSet<EventId>>,
}

impl InMemoryCoupling {
    /// Create a new in-memory coupling store
    pub fn new() -> Self {
        Self {
            associations: HashMap::new(),
            episode_index: HashMap::new(),
            concept_index: HashMap::new(),
        }
    }

    /// Get all associations for an episode with scores
    pub fn associations_for_episode(&self, episode_id: &EventId) -> Vec<&Association> {
        self.episode_index
            .get(episode_id)
            .map(|concepts| {
                concepts
                    .iter()
                    .filter_map(|c| self.associations.get(&(episode_id.clone(), c.clone())))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all associations for a concept
    pub fn associations_for_concept(&self, concept_id: &ConceptId) -> Vec<&Association> {
        self.concept_index
            .get(concept_id)
            .map(|episodes| {
                episodes
                    .iter()
                    .filter_map(|e| self.associations.get(&(e.clone(), concept_id.clone())))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for InMemoryCoupling {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryCoupling for InMemoryCoupling {
    fn associate(&mut self, episode_id: EventId, concept_id: ConceptId) -> MemoryResult<()> {
        self.associate_with_strength(episode_id, concept_id, 1.0, AssociationType::DirectMention)
    }

    fn associate_with_strength(
        &mut self,
        episode_id: EventId,
        concept_id: ConceptId,
        strength: f32,
        association_type: AssociationType,
    ) -> MemoryResult<()> {
        let key = (episode_id.clone(), concept_id.clone());

        // If association exists, strengthen it
        if let Some(assoc) = self.associations.get_mut(&key) {
            assoc.strengthen(strength * 0.1); // Reinforce by 10% of new strength
            return Ok(());
        }

        // Create new association
        let association = Association::new(
            episode_id.clone(),
            concept_id.clone(),
            strength,
            association_type,
        );

        self.associations.insert(key, association);

        // Update indices
        let ep_id_clone = episode_id.clone();
        self.episode_index
            .entry(episode_id)
            .or_default()
            .insert(concept_id.clone());
        self.concept_index
            .entry(concept_id)
            .or_default()
            .insert(ep_id_clone);

        Ok(())
    }

    fn concepts_for_episode(&self, episode_id: &EventId) -> MemoryResult<Vec<ConceptId>> {
        Ok(self
            .episode_index
            .get(episode_id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default())
    }

    fn episode_ids_for_concept(&self, concept_id: &ConceptId) -> MemoryResult<Vec<EventId>> {
        Ok(self
            .concept_index
            .get(concept_id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default())
    }

    fn get_association(
        &self,
        episode_id: &EventId,
        concept_id: &ConceptId,
    ) -> MemoryResult<Option<Association>> {
        Ok(self
            .associations
            .get(&(episode_id.clone(), concept_id.clone()))
            .cloned())
    }

    fn strengthen(
        &mut self,
        episode_id: &EventId,
        concept_id: &ConceptId,
        delta: f32,
    ) -> MemoryResult<()> {
        let key = (episode_id.clone(), concept_id.clone());
        if let Some(assoc) = self.associations.get_mut(&key) {
            assoc.strengthen(delta);
            Ok(())
        } else {
            Err(MemoryError::Storage(format!(
                "Association not found: {:?} -> {:?}",
                episode_id, concept_id
            )))
        }
    }

    fn disassociate(&mut self, episode_id: &EventId, concept_id: &ConceptId) -> MemoryResult<()> {
        let key = (episode_id.clone(), concept_id.clone());

        if self.associations.remove(&key).is_none() {
            return Ok(()); // Idempotent: no error if not found
        }

        // Update indices
        if let Some(concepts) = self.episode_index.get_mut(episode_id) {
            concepts.remove(concept_id);
        }
        if let Some(episodes) = self.concept_index.get_mut(concept_id) {
            episodes.remove(episode_id);
        }

        Ok(())
    }

    fn remove_episode_associations(&mut self, episode_id: &EventId) -> MemoryResult<usize> {
        let concepts = self.episode_index.remove(episode_id).unwrap_or_default();
        let count = concepts.len();

        for concept_id in &concepts {
            self.associations
                .remove(&(episode_id.clone(), concept_id.clone()));
            if let Some(episodes) = self.concept_index.get_mut(concept_id) {
                episodes.remove(episode_id);
            }
        }

        Ok(count)
    }

    fn remove_concept_associations(&mut self, concept_id: &ConceptId) -> MemoryResult<usize> {
        let episodes = self.concept_index.remove(concept_id).unwrap_or_default();
        let count = episodes.len();

        for episode_id in &episodes {
            self.associations
                .remove(&(episode_id.clone(), concept_id.clone()));
            if let Some(concepts) = self.episode_index.get_mut(episode_id) {
                concepts.remove(concept_id);
            }
        }

        Ok(count)
    }

    fn count(&self) -> usize {
        self.associations.len()
    }

    fn clear(&mut self) {
        self.associations.clear();
        self.episode_index.clear();
        self.concept_index.clear();
    }

    fn strongest_concepts(
        &self,
        episode_id: &EventId,
        k: usize,
    ) -> MemoryResult<Vec<(ConceptId, f32)>> {
        let mut scored: Vec<_> = self
            .associations_for_episode(episode_id)
            .into_iter()
            .map(|a| (a.concept_id.clone(), a.effective_strength()))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(k).collect())
    }

    fn strongest_episodes(
        &self,
        concept_id: &ConceptId,
        k: usize,
    ) -> MemoryResult<Vec<(EventId, f32)>> {
        let mut scored: Vec<_> = self
            .associations_for_concept(concept_id)
            .into_iter()
            .map(|a| (a.episode_id.clone(), a.effective_strength()))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(k).collect())
    }
}

// =============================================================================
// SPREADING ACTIVATION
// =============================================================================

/// Spreading activation for associative retrieval
pub struct SpreadingActivation<'a, C: MemoryCoupling> {
    coupling: &'a C,
    /// Activation levels for concepts
    concept_activations: HashMap<ConceptId, f32>,
    /// Activation levels for episodes
    episode_activations: HashMap<EventId, f32>,
    /// Decay rate per step
    decay_rate: f32,
    /// Spread factor (how much activation spreads)
    spread_factor: f32,
}

impl<'a, C: MemoryCoupling> SpreadingActivation<'a, C> {
    /// Create a new spreading activation instance
    pub fn new(coupling: &'a C) -> Self {
        Self {
            coupling,
            concept_activations: HashMap::new(),
            episode_activations: HashMap::new(),
            decay_rate: 0.1,
            spread_factor: 0.5,
        }
    }

    /// Set decay rate
    pub fn with_decay_rate(mut self, rate: f32) -> Self {
        self.decay_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set spread factor
    pub fn with_spread_factor(mut self, factor: f32) -> Self {
        self.spread_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Activate a concept
    pub fn activate_concept(&mut self, concept_id: ConceptId, activation: f32) {
        let current = self
            .concept_activations
            .get(&concept_id)
            .copied()
            .unwrap_or(0.0);
        self.concept_activations
            .insert(concept_id, (current + activation).min(1.0));
    }

    /// Activate an episode
    pub fn activate_episode(&mut self, episode_id: EventId, activation: f32) {
        let current = self
            .episode_activations
            .get(&episode_id)
            .copied()
            .unwrap_or(0.0);
        self.episode_activations
            .insert(episode_id, (current + activation).min(1.0));
    }

    /// Spread activation for one step
    pub fn spread_step(&mut self) -> MemoryResult<()> {
        // Spread from concepts to episodes
        let concept_spread: Vec<_> = self
            .concept_activations
            .iter()
            .map(|(id, act)| (id.clone(), *act))
            .collect();

        for (concept_id, activation) in concept_spread {
            let episodes = self.coupling.strongest_episodes(&concept_id, 10)?;
            for (episode_id, strength) in episodes {
                let spread = activation * strength * self.spread_factor;
                self.activate_episode(episode_id, spread);
            }
        }

        // Spread from episodes to concepts
        let episode_spread: Vec<_> = self
            .episode_activations
            .iter()
            .map(|(id, act)| (id.clone(), *act))
            .collect();

        for (episode_id, activation) in episode_spread {
            let concepts = self.coupling.strongest_concepts(&episode_id, 10)?;
            for (concept_id, strength) in concepts {
                let spread = activation * strength * self.spread_factor;
                self.activate_concept(concept_id, spread);
            }
        }

        // Apply decay
        for activation in self.concept_activations.values_mut() {
            *activation *= 1.0 - self.decay_rate;
        }
        for activation in self.episode_activations.values_mut() {
            *activation *= 1.0 - self.decay_rate;
        }

        Ok(())
    }

    /// Run spreading activation for N steps
    pub fn spread(&mut self, steps: usize) -> MemoryResult<()> {
        for _ in 0..steps {
            self.spread_step()?;
        }
        Ok(())
    }

    /// Get top activated concepts
    pub fn top_concepts(&self, k: usize) -> Vec<(ConceptId, f32)> {
        let mut sorted: Vec<_> = self
            .concept_activations
            .iter()
            .map(|(id, act)| (id.clone(), *act))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(k).collect()
    }

    /// Get top activated episodes
    pub fn top_episodes(&self, k: usize) -> Vec<(EventId, f32)> {
        let mut sorted: Vec<_> = self
            .episode_activations
            .iter()
            .map(|(id, act)| (id.clone(), *act))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(k).collect()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event_id(n: u32) -> EventId {
        EventId::new(format!("evt-{}", n))
    }

    fn sample_concept_id(n: u32) -> ConceptId {
        ConceptId::new(format!("cpt-{}", n))
    }

    #[test]
    fn test_association_new() {
        let assoc = Association::new(
            sample_event_id(1),
            sample_concept_id(1),
            0.8,
            AssociationType::DirectMention,
        );
        assert!((assoc.strength - 0.8).abs() < 0.001);
        assert_eq!(assoc.activation_count, 0);
    }

    #[test]
    fn test_association_activate() {
        let mut assoc = Association::direct(sample_event_id(1), sample_concept_id(1), 0.8);
        assoc.activate();
        assert_eq!(assoc.activation_count, 1);
        assoc.activate();
        assert_eq!(assoc.activation_count, 2);
    }

    #[test]
    fn test_association_strengthen() {
        let mut assoc = Association::direct(sample_event_id(1), sample_concept_id(1), 0.5);
        assoc.strengthen(0.3);
        assert!((assoc.strength - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_association_strengthen_clamp() {
        let mut assoc = Association::direct(sample_event_id(1), sample_concept_id(1), 0.9);
        assoc.strengthen(0.5);
        assert!((assoc.strength - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_association_weaken() {
        let mut assoc = Association::direct(sample_event_id(1), sample_concept_id(1), 0.8);
        assoc.weaken(0.3);
        assert!((assoc.strength - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_association_decay() {
        let assoc = Association::direct(sample_event_id(1), sample_concept_id(1), 1.0);
        // Just created, decay should be ~1.0
        assert!(assoc.decay_factor() > 0.99);
    }

    #[test]
    fn test_inmemory_associate() {
        let mut coupling = InMemoryCoupling::new();
        coupling
            .associate(sample_event_id(1), sample_concept_id(1))
            .unwrap();
        assert_eq!(coupling.count(), 1);
    }

    #[test]
    fn test_inmemory_concepts_for_episode() {
        let mut coupling = InMemoryCoupling::new();
        let ep = sample_event_id(1);
        coupling
            .associate(ep.clone(), sample_concept_id(1))
            .unwrap();
        coupling
            .associate(ep.clone(), sample_concept_id(2))
            .unwrap();

        let concepts = coupling.concepts_for_episode(&ep).unwrap();
        assert_eq!(concepts.len(), 2);
    }

    #[test]
    fn test_inmemory_get_association() {
        let mut coupling = InMemoryCoupling::new();
        let ep = sample_event_id(1);
        let cp = sample_concept_id(1);
        coupling
            .associate_with_strength(ep.clone(), cp.clone(), 0.75, AssociationType::Inferred)
            .unwrap();

        let assoc = coupling.get_association(&ep, &cp).unwrap();
        assert!(assoc.is_some());
        assert!((assoc.unwrap().strength - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_inmemory_strengthen() {
        let mut coupling = InMemoryCoupling::new();
        let ep = sample_event_id(1);
        let cp = sample_concept_id(1);
        coupling
            .associate_with_strength(ep.clone(), cp.clone(), 0.5, AssociationType::DirectMention)
            .unwrap();

        coupling.strengthen(&ep, &cp, 0.3).unwrap();

        let assoc = coupling.get_association(&ep, &cp).unwrap().unwrap();
        assert!((assoc.strength - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_inmemory_disassociate() {
        let mut coupling = InMemoryCoupling::new();
        let ep = sample_event_id(1);
        let cp = sample_concept_id(1);
        coupling.associate(ep.clone(), cp.clone()).unwrap();

        assert_eq!(coupling.count(), 1);
        coupling.disassociate(&ep, &cp).unwrap();
        assert_eq!(coupling.count(), 0);
    }

    #[test]
    fn test_inmemory_remove_episode_associations() {
        let mut coupling = InMemoryCoupling::new();
        let ep = sample_event_id(1);
        coupling
            .associate(ep.clone(), sample_concept_id(1))
            .unwrap();
        coupling
            .associate(ep.clone(), sample_concept_id(2))
            .unwrap();
        coupling
            .associate(sample_event_id(2), sample_concept_id(1))
            .unwrap();

        let removed = coupling.remove_episode_associations(&ep).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(coupling.count(), 1);
    }

    #[test]
    fn test_inmemory_remove_concept_associations() {
        let mut coupling = InMemoryCoupling::new();
        let cp = sample_concept_id(1);
        coupling.associate(sample_event_id(1), cp.clone()).unwrap();
        coupling.associate(sample_event_id(2), cp.clone()).unwrap();
        coupling
            .associate(sample_event_id(1), sample_concept_id(2))
            .unwrap();

        let removed = coupling.remove_concept_associations(&cp).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(coupling.count(), 1);
    }

    #[test]
    fn test_inmemory_strongest_concepts() {
        let mut coupling = InMemoryCoupling::new();
        let ep = sample_event_id(1);
        coupling
            .associate_with_strength(
                ep.clone(),
                sample_concept_id(1),
                0.3,
                AssociationType::DirectMention,
            )
            .unwrap();
        coupling
            .associate_with_strength(
                ep.clone(),
                sample_concept_id(2),
                0.9,
                AssociationType::DirectMention,
            )
            .unwrap();
        coupling
            .associate_with_strength(
                ep.clone(),
                sample_concept_id(3),
                0.5,
                AssociationType::DirectMention,
            )
            .unwrap();

        let strongest = coupling.strongest_concepts(&ep, 2).unwrap();
        assert_eq!(strongest.len(), 2);
        assert!(strongest[0].1 > strongest[1].1);
    }

    #[test]
    fn test_inmemory_clear() {
        let mut coupling = InMemoryCoupling::new();
        coupling
            .associate(sample_event_id(1), sample_concept_id(1))
            .unwrap();
        coupling
            .associate(sample_event_id(2), sample_concept_id(2))
            .unwrap();

        coupling.clear();
        assert_eq!(coupling.count(), 0);
    }

    #[test]
    fn test_spreading_activation_concept() {
        let mut coupling = InMemoryCoupling::new();
        let ep = sample_event_id(1);
        let cp = sample_concept_id(1);
        coupling.associate(ep.clone(), cp.clone()).unwrap();

        let mut spread = SpreadingActivation::new(&coupling);
        spread.activate_concept(cp.clone(), 1.0);

        assert!(spread.concept_activations.contains_key(&cp));
    }

    #[test]
    fn test_spreading_activation_spread() {
        let mut coupling = InMemoryCoupling::new();
        let ep1 = sample_event_id(1);
        let cp1 = sample_concept_id(1);
        let cp2 = sample_concept_id(2);

        coupling.associate(ep1.clone(), cp1.clone()).unwrap();
        coupling.associate(ep1.clone(), cp2.clone()).unwrap();

        let mut spread = SpreadingActivation::new(&coupling);
        spread.activate_concept(cp1.clone(), 1.0);
        spread.spread(1).unwrap();

        // After spreading, episode should be activated
        assert!(spread.episode_activations.contains_key(&ep1));
    }

    #[test]
    fn test_spreading_activation_top_concepts() {
        let coupling = InMemoryCoupling::new();
        let mut spread = SpreadingActivation::new(&coupling);

        spread.activate_concept(sample_concept_id(1), 0.5);
        spread.activate_concept(sample_concept_id(2), 0.9);
        spread.activate_concept(sample_concept_id(3), 0.3);

        let top = spread.top_concepts(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].1 > top[1].1);
    }
}
