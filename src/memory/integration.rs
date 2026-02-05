//! # Memory-Compounding Integration
//!
//! Bridges the persistent memory system with the Compounding Cognitive Cohesion
//! architecture. This enables:
//!
//! - Coherence-modulated memory storage and retrieval
//! - Memory participation in compound_interactions
//! - Coordinated consolidation with StreamGraphMemory
//! - SOC-based importance scoring
//!
//! ## Integration Points
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    CompoundingCohesionSystem                             │
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐  │
//! │  │ StreamGraphMem  │◄──►│ MemoryBridge    │◄──►│ MemorySystem        │  │
//! │  │ (neural traces) │    │ (this module)   │    │ (episodes+concepts) │  │
//! │  └─────────────────┘    └────────┬────────┘    └─────────────────────┘  │
//! │                                  │                                       │
//! │  ┌─────────────────┐    ┌────────▼────────┐                             │
//! │  │ HierarchicalSOC │───►│ CoherenceScorer │ (modulates importance)      │
//! │  └─────────────────┘    └─────────────────┘                             │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use super::consolidation::{ConsolidationConfig, MemoryConsolidator};
use super::coupling::{AssociationType, InMemoryCoupling, MemoryCoupling};
use super::episodic::{Episode, EpisodeType, EpisodicStore, InMemoryEpisodicStore};
use super::semantic::{Concept, InMemorySemanticStore, SemanticStore};
use super::{
    ConceptId, EventId, MemoryConfig, MemoryError, MemoryResult, MemorySystem, RelevanceScore,
};

// =============================================================================
// COMPOUNDING-AWARE TRAIT
// =============================================================================

/// Trait for memory systems that can integrate with compounding cohesion
pub trait CompoundingAware {
    /// Record an experience with coherence-based importance
    fn record_with_coherence(
        &mut self,
        content: &str,
        episode_type: EpisodeType,
        coherence_score: f64,
        prediction_error: f64,
    ) -> MemoryResult<EventId>;

    /// Retrieve memories modulated by current coherence state
    fn retrieve_coherent(
        &mut self,
        query: &str,
        k: usize,
        coherence_score: f64,
        temperature: f64,
    ) -> MemoryResult<Vec<(Episode, RelevanceScore)>>;

    /// Consolidate memories in coordination with episode boundaries
    fn consolidate_episode(
        &mut self,
        episode_num: usize,
    ) -> MemoryResult<IntegrationConsolidationResult>;

    /// Receive credit assignment from successful goal completion
    fn receive_credit(&mut self, goal_id: usize, credit_strength: f64) -> MemoryResult<usize>;

    /// Get memories relevant to imagination/planning
    fn get_planning_context(&self, state_features: &[f64], k: usize) -> MemoryResult<Vec<Episode>>;
}

/// Result of integrated consolidation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntegrationConsolidationResult {
    /// Episodes processed
    pub episodes_processed: usize,
    /// Episodes merged
    pub episodes_merged: usize,
    /// Concepts extracted
    pub concepts_extracted: usize,
    /// Associations strengthened
    pub associations_strengthened: usize,
    /// Episode number
    pub episode: usize,
}

// =============================================================================
// COHERENCE SCORER
// =============================================================================

/// Scores memory importance based on coherence state
#[derive(Debug, Clone)]
pub struct CoherenceScorer {
    /// Base importance for new memories
    pub base_importance: f64,
    /// Weight for coherence contribution
    pub coherence_weight: f64,
    /// Weight for surprise/prediction error
    pub surprise_weight: f64,
    /// Minimum importance threshold
    pub min_importance: f64,
    /// Maximum importance
    pub max_importance: f64,
}

impl Default for CoherenceScorer {
    fn default() -> Self {
        Self {
            base_importance: 0.5,
            coherence_weight: 0.3,
            surprise_weight: 0.4,
            min_importance: 0.1,
            max_importance: 1.0,
        }
    }
}

impl CoherenceScorer {
    /// Create a new coherence scorer
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate importance score based on coherence and surprise
    ///
    /// - High coherence + low surprise = routine, lower importance
    /// - High coherence + high surprise = unexpected success, high importance  
    /// - Low coherence + high surprise = confusion, moderate importance
    /// - Low coherence + low surprise = expected difficulty, lower importance
    pub fn score_importance(&self, coherence: f64, prediction_error: f64) -> f64 {
        // Normalize inputs
        let coherence = coherence.clamp(0.0, 1.0);
        let surprise = prediction_error.clamp(0.0, 1.0);

        // Importance = base + coherence_contribution + surprise_contribution
        // High surprise is always interesting
        // High coherence + high surprise is especially important (unexpected in stable context)
        let coherence_boost = if coherence > 0.7 && surprise > 0.5 {
            0.2 // Bonus for surprising events in coherent state
        } else {
            0.0
        };

        let importance = self.base_importance
            + self.coherence_weight * coherence
            + self.surprise_weight * surprise
            + coherence_boost;

        importance.clamp(self.min_importance, self.max_importance)
    }

    /// Adjust retrieval relevance based on current coherence
    ///
    /// - High coherence: prefer recent, focused memories
    /// - Low coherence: broader search, older memories may help
    pub fn modulate_relevance(
        &self,
        base_score: f64,
        memory_recency: f64,
        current_coherence: f64,
    ) -> f64 {
        let coherence = current_coherence.clamp(0.0, 1.0);

        // High coherence = weight recency more (stay focused)
        // Low coherence = weight recency less (explore older memories)
        let recency_weight = 0.2 + 0.3 * coherence;
        let base_weight = 1.0 - recency_weight;

        (base_weight * base_score + recency_weight * memory_recency).clamp(0.0, 1.0)
    }
}

// =============================================================================
// MEMORY BRIDGE
// =============================================================================

/// Bridge between MemorySystem and CompoundingCohesionSystem
pub struct MemoryBridge<E, S, C>
where
    E: EpisodicStore,
    S: SemanticStore,
    C: MemoryCoupling,
{
    /// The underlying memory system
    pub memory: MemorySystem<E, S, C>,
    /// Coherence scorer
    pub scorer: CoherenceScorer,
    /// Memory consolidator
    pub consolidator: MemoryConsolidator,
    /// Current episode number
    pub current_episode: usize,
    /// Recent event IDs for credit assignment
    recent_events: Vec<EventId>,
    /// Maximum recent events to track
    max_recent_events: usize,
    /// Goal-to-events mapping for credit assignment
    goal_events: HashMap<usize, Vec<EventId>>,
    /// Integration statistics
    pub stats: IntegrationStats,
}

/// Statistics for memory-compounding integration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntegrationStats {
    /// Total coherence-modulated records
    pub coherence_records: usize,
    /// Total coherence-modulated retrievals
    pub coherence_retrievals: usize,
    /// Total credit assignments received
    pub credits_received: usize,
    /// Total planning context requests
    pub planning_requests: usize,
    /// Total episodes consolidated
    pub episodes_consolidated: usize,
}

impl<E, S, C> MemoryBridge<E, S, C>
where
    E: EpisodicStore,
    S: SemanticStore,
    C: MemoryCoupling,
{
    /// Create a new memory bridge
    pub fn new(memory: MemorySystem<E, S, C>) -> Self {
        Self {
            memory,
            scorer: CoherenceScorer::new(),
            consolidator: MemoryConsolidator::new(),
            current_episode: 0,
            recent_events: Vec::new(),
            max_recent_events: 100,
            goal_events: HashMap::new(),
            stats: IntegrationStats::default(),
        }
    }

    /// Create with custom consolidation config
    pub fn with_consolidation(mut self, config: ConsolidationConfig) -> Self {
        self.consolidator = MemoryConsolidator::with_config(config);
        self
    }

    /// Associate current memories with a goal for credit assignment
    pub fn associate_with_goal(&mut self, goal_id: usize) {
        let events = self.recent_events.clone();
        self.goal_events.insert(goal_id, events);
    }

    /// Get a summary of integration stats
    pub fn summary(&self) -> String {
        format!(
            "MemoryBridge: {} episodes, {} coherence_records, {} retrievals, {} credits",
            self.current_episode,
            self.stats.coherence_records,
            self.stats.coherence_retrievals,
            self.stats.credits_received,
        )
    }

    /// Track a recent event for credit assignment
    fn track_event(&mut self, event_id: EventId) {
        self.recent_events.push(event_id);
        if self.recent_events.len() > self.max_recent_events {
            self.recent_events.remove(0);
        }
    }
}

impl<E, S, C> CompoundingAware for MemoryBridge<E, S, C>
where
    E: EpisodicStore,
    S: SemanticStore,
    C: MemoryCoupling,
{
    fn record_with_coherence(
        &mut self,
        content: &str,
        episode_type: EpisodeType,
        coherence_score: f64,
        prediction_error: f64,
    ) -> MemoryResult<EventId> {
        // Calculate importance based on coherence state
        let importance = self
            .scorer
            .score_importance(coherence_score, prediction_error);

        // Create episode with coherence-based importance
        let mut episode = Episode::new(content, episode_type, "compounding_system");
        episode.importance = importance as f32;
        episode
            .metadata
            .insert("coherence".to_string(), format!("{:.4}", coherence_score));
        episode.metadata.insert(
            "prediction_error".to_string(),
            format!("{:.4}", prediction_error),
        );

        // Record the episode
        let event_id = self.memory.record(episode)?;

        // Track for credit assignment
        self.track_event(event_id.clone());
        self.stats.coherence_records += 1;

        Ok(event_id)
    }

    fn retrieve_coherent(
        &mut self,
        query: &str,
        k: usize,
        coherence_score: f64,
        temperature: f64,
    ) -> MemoryResult<Vec<(Episode, RelevanceScore)>> {
        // Get base results
        let mut results = self.memory.query(query, k * 2)?; // Get more, then filter

        // Modulate relevance scores based on coherence
        let now = Utc::now();
        for (episode, score) in &mut results {
            // Calculate recency (0-1 scale, 1 = very recent)
            let age_hours = (now - episode.timestamp).num_hours() as f64;
            let recency = (-age_hours / 24.0).exp(); // Decay over 24 hours

            // Modulate the score
            let modulated =
                self.scorer
                    .modulate_relevance(score.score as f64, recency, coherence_score);

            // Apply temperature (higher temp = more exploration of lower-ranked memories)
            let temp_adjusted = if temperature > 1.0 {
                // Higher temperature flattens the distribution
                modulated.powf(1.0 / temperature)
            } else {
                modulated
            };

            score.score = temp_adjusted as f32;
        }

        // Re-sort by adjusted scores
        results.sort_by(|a, b| {
            b.1.score
                .partial_cmp(&a.1.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top k
        results.truncate(k);

        self.stats.coherence_retrievals += 1;
        Ok(results)
    }

    fn consolidate_episode(
        &mut self,
        episode_num: usize,
    ) -> MemoryResult<IntegrationConsolidationResult> {
        self.current_episode = episode_num;

        // Run consolidation
        let result = self.consolidator.consolidate(
            &mut self.memory.episodic,
            &mut self.memory.semantic,
            &mut self.memory.coupling,
        )?;

        self.stats.episodes_consolidated += 1;

        Ok(IntegrationConsolidationResult {
            episodes_processed: result.episodes_processed,
            episodes_merged: result.episodes_merged,
            concepts_extracted: result.concepts_extracted,
            associations_strengthened: result.associations_created,
            episode: episode_num,
        })
    }

    fn receive_credit(&mut self, goal_id: usize, credit_strength: f64) -> MemoryResult<usize> {
        let mut updated_count = 0;

        // Get events associated with this goal
        let event_ids = self.goal_events.get(&goal_id).cloned().unwrap_or_default();

        // Boost importance of associated episodes
        for event_id in event_ids {
            if let Ok(mut episode) = self.memory.episodic.get(&event_id) {
                // Increase importance based on credit
                episode.importance = (episode.importance + (credit_strength * 0.2) as f32).min(1.0);
                episode.metadata.insert(
                    "goal_credit".to_string(),
                    format!("goal_{}_credit_{:.2}", goal_id, credit_strength),
                );

                if self.memory.episodic.update(episode).is_ok() {
                    updated_count += 1;
                }
            }
        }

        // Clean up goal-events mapping
        self.goal_events.remove(&goal_id);

        self.stats.credits_received += updated_count;
        Ok(updated_count)
    }

    fn get_planning_context(&self, state_features: &[f64], k: usize) -> MemoryResult<Vec<Episode>> {
        // Convert features to a query string (simple approach)
        // In production, this would use vector similarity
        let query = format!(
            "state features: {:?}",
            &state_features[..state_features.len().min(5)]
        );

        let results = self.memory.episodic.recall_similar(&query, k)?;
        Ok(results)
    }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/// Create a default in-memory bridge with compounding integration
pub fn create_compounding_memory_bridge(
    config: MemoryConfig,
) -> MemoryBridge<InMemoryEpisodicStore, InMemorySemanticStore, InMemoryCoupling> {
    let memory = super::create_in_memory_system(config);
    MemoryBridge::new(memory)
}

/// Create a bridge with custom consolidation settings
pub fn create_compounding_memory_bridge_with_consolidation(
    memory_config: MemoryConfig,
    consolidation_config: ConsolidationConfig,
) -> MemoryBridge<InMemoryEpisodicStore, InMemorySemanticStore, InMemoryCoupling> {
    let memory = super::create_in_memory_system(memory_config);
    MemoryBridge::new(memory).with_consolidation(consolidation_config)
}

// =============================================================================
// STREAM GRAPH BRIDGE (for future use with StreamGraphMemory)
// =============================================================================

/// Converts between Episode and StreamGraphMemory node features
#[derive(Debug, Clone)]
pub struct StreamGraphAdapter {
    /// Feature dimension for graph nodes
    feature_dim: usize,
}

impl StreamGraphAdapter {
    /// Create a new adapter
    pub fn new(feature_dim: usize) -> Self {
        Self { feature_dim }
    }

    /// Convert episode content to feature vector (simple hash-based approach)
    /// In production, this would use embeddings
    pub fn episode_to_features(&self, episode: &Episode) -> Vec<f32> {
        let mut features = vec![0.0f32; self.feature_dim];

        // Simple feature extraction based on content hash
        for (i, c) in episode.content.chars().enumerate() {
            let idx = (c as usize + i) % self.feature_dim;
            features[idx] += 0.1;
        }

        // Add importance
        features[0] = episode.importance;

        // Add type encoding
        let type_idx = match &episode.episode_type {
            EpisodeType::Interaction => 1,
            EpisodeType::ToolExecution => 2,
            EpisodeType::LLMResponse => 3,
            EpisodeType::Observation => 4,
            EpisodeType::Decision => 5,
            EpisodeType::Error => 6,
            EpisodeType::Learning => 7,
            EpisodeType::Custom(_) => 8,
        };
        if type_idx < self.feature_dim {
            features[type_idx] = 1.0;
        }

        // Normalize
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for f in &mut features {
                *f /= norm;
            }
        }

        features
    }

    /// Check if features match an episode (for retrieval)
    pub fn features_match(&self, features: &[f32], episode: &Episode) -> f32 {
        let episode_features = self.episode_to_features(episode);

        // Cosine similarity
        let dot: f32 = features
            .iter()
            .zip(&episode_features)
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = episode_features.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_scorer_default() {
        let scorer = CoherenceScorer::default();
        assert!((scorer.base_importance - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_coherence_scorer_high_coherence_high_surprise() {
        let scorer = CoherenceScorer::new();
        let importance = scorer.score_importance(0.9, 0.8);
        // Should be high due to bonus
        assert!(importance > 0.8);
    }

    #[test]
    fn test_coherence_scorer_low_coherence_low_surprise() {
        let scorer = CoherenceScorer::new();
        let importance = scorer.score_importance(0.2, 0.2);
        // Should be relatively low
        assert!(importance < 0.7);
    }

    #[test]
    fn test_coherence_scorer_modulate_relevance() {
        let scorer = CoherenceScorer::new();

        // High coherence should weight recency more
        let high_coh = scorer.modulate_relevance(0.5, 0.9, 0.9);
        let low_coh = scorer.modulate_relevance(0.5, 0.9, 0.2);

        // With high recency (0.9) and high coherence, score should be higher
        assert!(high_coh > low_coh);
    }

    #[test]
    fn test_memory_bridge_creation() {
        let config = MemoryConfig::default();
        let bridge = create_compounding_memory_bridge(config);
        assert_eq!(bridge.current_episode, 0);
    }

    #[test]
    fn test_memory_bridge_record_with_coherence() {
        let config = MemoryConfig::default();
        let mut bridge = create_compounding_memory_bridge(config);

        let event_id = bridge
            .record_with_coherence(
                "Test event with high coherence",
                EpisodeType::Observation,
                0.9,
                0.3,
            )
            .unwrap();

        assert!(event_id.0.starts_with("evt-"));
        assert_eq!(bridge.stats.coherence_records, 1);
    }

    #[test]
    fn test_memory_bridge_retrieve_coherent() {
        let config = MemoryConfig::default();
        let mut bridge = create_compounding_memory_bridge(config);

        // Record some events
        bridge
            .record_with_coherence("The quick brown fox", EpisodeType::Observation, 0.8, 0.2)
            .unwrap();
        bridge
            .record_with_coherence("The lazy dog sleeps", EpisodeType::Observation, 0.6, 0.4)
            .unwrap();

        let results = bridge.retrieve_coherent("brown fox", 2, 0.9, 1.0).unwrap();
        assert!(!results.is_empty());
        assert_eq!(bridge.stats.coherence_retrievals, 1);
    }

    #[test]
    fn test_memory_bridge_consolidate_episode() {
        let config = MemoryConfig::default();
        let mut bridge = create_compounding_memory_bridge(config);

        // Record some events
        for i in 0..5 {
            bridge
                .record_with_coherence(&format!("Event {}", i), EpisodeType::Observation, 0.7, 0.3)
                .unwrap();
        }

        let result = bridge.consolidate_episode(1).unwrap();
        assert_eq!(result.episode, 1);
        assert_eq!(bridge.current_episode, 1);
    }

    #[test]
    fn test_memory_bridge_credit_assignment() {
        let config = MemoryConfig::default();
        let mut bridge = create_compounding_memory_bridge(config);

        // Record events
        bridge
            .record_with_coherence("Goal-related event", EpisodeType::Decision, 0.8, 0.2)
            .unwrap();

        // Associate with goal
        bridge.associate_with_goal(42);

        // Receive credit
        let updated = bridge.receive_credit(42, 0.9).unwrap();
        assert!(updated > 0);
    }

    #[test]
    fn test_stream_graph_adapter() {
        let adapter = StreamGraphAdapter::new(32);
        let episode = Episode::new("test content", EpisodeType::Observation, "test");

        let features = adapter.episode_to_features(&episode);
        assert_eq!(features.len(), 32);

        // Features should be normalized
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stream_graph_adapter_match() {
        let adapter = StreamGraphAdapter::new(32);
        let episode = Episode::new("test content", EpisodeType::Observation, "test");

        let features = adapter.episode_to_features(&episode);
        let similarity = adapter.features_match(&features, &episode);

        // Should match itself perfectly
        assert!((similarity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_integration_stats_default() {
        let stats = IntegrationStats::default();
        assert_eq!(stats.coherence_records, 0);
        assert_eq!(stats.credits_received, 0);
    }

    #[test]
    fn test_memory_bridge_summary() {
        let config = MemoryConfig::default();
        let bridge = create_compounding_memory_bridge(config);
        let summary = bridge.summary();
        assert!(summary.contains("MemoryBridge"));
    }
}
