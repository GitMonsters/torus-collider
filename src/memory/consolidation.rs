//! # Memory Consolidation
//!
//! Background memory compression, merging, and episodic→semantic abstraction.
//! Implements sleep-like memory consolidation for efficient long-term storage.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::coupling::MemoryCoupling;
use super::episodic::{Episode, EpisodeType, EpisodicStore};
use super::semantic::{Concept, SemanticStore};
use super::{ConceptId, EventId, MemoryError, MemoryResult};

// =============================================================================
// CONSOLIDATION CONFIG
// =============================================================================

/// Configuration for memory consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// Minimum age (seconds) before an episode can be consolidated
    pub min_age_seconds: u64,
    /// Similarity threshold for merging episodes (0.0 - 1.0)
    pub merge_threshold: f32,
    /// Importance threshold below which episodes may be pruned
    pub prune_threshold: f32,
    /// Maximum number of episodes to process per consolidation run
    pub batch_size: usize,
    /// Whether to extract concepts during consolidation
    pub extract_concepts: bool,
    /// Whether to merge similar episodes
    pub merge_similar: bool,
    /// Whether to prune low-importance old episodes
    pub prune_old: bool,
    /// Age threshold (seconds) for pruning consideration
    pub prune_age_seconds: u64,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            min_age_seconds: 300, // 5 minutes
            merge_threshold: 0.85,
            prune_threshold: 0.2,
            batch_size: 100,
            extract_concepts: true,
            merge_similar: true,
            prune_old: true,
            prune_age_seconds: 86400 * 7, // 1 week
        }
    }
}

// =============================================================================
// CONSOLIDATION RESULT
// =============================================================================

/// Result of a consolidation run
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationResult {
    /// Number of episodes processed
    pub episodes_processed: usize,
    /// Number of episodes merged
    pub episodes_merged: usize,
    /// Number of episodes pruned
    pub episodes_pruned: usize,
    /// Number of new concepts extracted
    pub concepts_extracted: usize,
    /// Number of associations created
    pub associations_created: usize,
    /// Duration of consolidation (milliseconds)
    pub duration_ms: u64,
    /// Any warnings or notes
    pub notes: Vec<String>,
}

impl ConsolidationResult {
    /// Check if any consolidation occurred
    pub fn had_changes(&self) -> bool {
        self.episodes_merged > 0 || self.episodes_pruned > 0 || self.concepts_extracted > 0
    }
}

// =============================================================================
// MEMORY CONSOLIDATOR
// =============================================================================

/// Memory consolidator for compressing and abstracting memories
pub struct MemoryConsolidator {
    config: ConsolidationConfig,
}

impl MemoryConsolidator {
    /// Create a new consolidator with default config
    pub fn new() -> Self {
        Self {
            config: ConsolidationConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ConsolidationConfig) -> Self {
        Self { config }
    }

    /// Run consolidation on memory stores
    pub fn consolidate<E, S, C>(
        &self,
        episodic: &mut E,
        semantic: &mut S,
        coupling: &mut C,
    ) -> MemoryResult<ConsolidationResult>
    where
        E: EpisodicStore,
        S: SemanticStore,
        C: MemoryCoupling,
    {
        let start = std::time::Instant::now();
        let mut result = ConsolidationResult::default();

        // Get episodes to process
        let episodes = episodic.recent(self.config.batch_size)?;
        result.episodes_processed = episodes.len();

        // Phase 1: Extract concepts from episodes
        if self.config.extract_concepts {
            for episode in &episodes {
                if self.is_old_enough(episode) {
                    let extracted = self.extract_concepts_from_episode(episode);
                    for (name, confidence) in extracted {
                        // Check if concept already exists
                        if semantic.find_by_name(&name)?.is_none() {
                            let concept = Concept::new(name, vec![]).with_confidence(confidence);
                            if let Ok(concept_id) = semantic.store(concept) {
                                result.concepts_extracted += 1;

                                // Create association
                                coupling.associate(episode.id.clone(), concept_id)?;
                                result.associations_created += 1;
                            }
                        }
                    }
                }
            }
        }

        // Phase 2: Merge similar episodes
        if self.config.merge_similar {
            let merged = self.merge_similar_episodes(episodic, coupling)?;
            result.episodes_merged = merged;
        }

        // Phase 3: Prune old, low-importance episodes
        if self.config.prune_old {
            let pruned = self.prune_old_episodes(episodic, coupling)?;
            result.episodes_pruned = pruned;
        }

        result.duration_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }

    /// Check if an episode is old enough for consolidation
    fn is_old_enough(&self, episode: &Episode) -> bool {
        episode.age_seconds() as u64 >= self.config.min_age_seconds
    }

    /// Check if an episode is old enough for pruning
    fn is_old_for_pruning(&self, episode: &Episode) -> bool {
        episode.age_seconds() as u64 >= self.config.prune_age_seconds
    }

    /// Extract concepts from an episode
    fn extract_concepts_from_episode(&self, episode: &Episode) -> Vec<(String, f32)> {
        let mut concepts = Vec::new();

        // Simple keyword extraction (would use NLP in production)
        let words: Vec<_> = episode
            .content
            .split_whitespace()
            .filter(|w| w.len() > 4) // Skip short words
            .map(|w| w.to_lowercase())
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        // Count word frequencies
        let mut freq: HashMap<String, usize> = HashMap::new();
        for word in &words {
            *freq.entry(word.clone()).or_default() += 1;
        }

        // Top words become concepts
        let mut sorted: Vec<_> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        for (word, count) in sorted.into_iter().take(5) {
            let confidence = (count as f32 / words.len().max(1) as f32).min(1.0);
            if confidence > 0.1 {
                concepts.push((word, confidence));
            }
        }

        // Add metadata-based concepts
        if let Some(tool_name) = episode.metadata.get("tool_name") {
            concepts.push((format!("tool:{}", tool_name), 0.9));
        }

        concepts
    }

    /// Merge similar episodes
    fn merge_similar_episodes<E, C>(
        &self,
        episodic: &mut E,
        coupling: &mut C,
    ) -> MemoryResult<usize>
    where
        E: EpisodicStore,
        C: MemoryCoupling,
    {
        let mut merged_count = 0;
        let episodes = episodic.recent(self.config.batch_size)?;

        // Group by type
        let mut by_type: HashMap<EpisodeType, Vec<Episode>> = HashMap::new();
        for ep in episodes {
            if self.is_old_enough(&ep) {
                by_type.entry(ep.episode_type.clone()).or_default().push(ep);
            }
        }

        // Within each type, find similar episodes to merge
        for (_ep_type, type_episodes) in by_type {
            let mut i = 0;
            while i < type_episodes.len() {
                let mut j = i + 1;
                while j < type_episodes.len() {
                    let similarity =
                        self.calculate_similarity(&type_episodes[i], &type_episodes[j]);

                    if similarity >= self.config.merge_threshold {
                        // Merge j into i
                        let merged = self.merge_episodes(&type_episodes[i], &type_episodes[j]);
                        episodic.update(merged)?;
                        episodic.delete(&type_episodes[j].id)?;
                        coupling.remove_episode_associations(&type_episodes[j].id)?;
                        merged_count += 1;
                    }
                    j += 1;
                }
                i += 1;
            }
        }

        Ok(merged_count)
    }

    /// Calculate similarity between two episodes
    fn calculate_similarity(&self, a: &Episode, b: &Episode) -> f32 {
        // Simple Jaccard similarity on words
        let a_lower = a.content.to_lowercase();
        let b_lower = b.content.to_lowercase();
        let a_words: std::collections::HashSet<_> = a_lower.split_whitespace().collect();
        let b_words: std::collections::HashSet<_> = b_lower.split_whitespace().collect();

        if a_words.is_empty() || b_words.is_empty() {
            return 0.0;
        }

        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();

        intersection as f32 / union as f32
    }

    /// Merge two episodes into one
    fn merge_episodes(&self, a: &Episode, b: &Episode) -> Episode {
        // Keep the earlier episode, combine content
        let (primary, secondary) = if a.timestamp < b.timestamp {
            (a, b)
        } else {
            (b, a)
        };

        let mut merged = primary.clone();
        merged.content = format!("{}\n---\n{}", primary.content, secondary.content);
        merged.importance = (primary.importance + secondary.importance) / 2.0;
        merged.access_count = primary.access_count.max(secondary.access_count);

        // Merge metadata
        for (k, v) in &secondary.metadata {
            merged
                .metadata
                .entry(k.clone())
                .or_insert_with(|| v.clone());
        }

        merged
            .metadata
            .insert("merged_from".to_string(), secondary.id.to_string());
        merged
    }

    /// Prune old, low-importance episodes
    fn prune_old_episodes<E, C>(&self, episodic: &mut E, coupling: &mut C) -> MemoryResult<usize>
    where
        E: EpisodicStore,
        C: MemoryCoupling,
    {
        let mut pruned_count = 0;
        let episodes = episodic.recent(self.config.batch_size * 2)?;

        for ep in episodes {
            if self.is_old_for_pruning(&ep)
                && ep.importance < self.config.prune_threshold
                && ep.access_count == 0
            {
                episodic.delete(&ep.id)?;
                coupling.remove_episode_associations(&ep.id)?;
                pruned_count += 1;
            }
        }

        Ok(pruned_count)
    }
}

impl Default for MemoryConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ABSTRACTION ENGINE
// =============================================================================

/// Engine for abstracting episodic memories into semantic knowledge
pub struct AbstractionEngine {
    /// Minimum number of related episodes to form a concept
    min_episode_count: usize,
    /// Confidence threshold for abstraction
    confidence_threshold: f32,
}

impl AbstractionEngine {
    /// Create a new abstraction engine
    pub fn new() -> Self {
        Self {
            min_episode_count: 3,
            confidence_threshold: 0.6,
        }
    }

    /// Set minimum episode count
    pub fn with_min_episodes(mut self, count: usize) -> Self {
        self.min_episode_count = count;
        self
    }

    /// Abstract patterns from episodes into concepts
    pub fn abstract_patterns<E, S, C>(
        &self,
        episodic: &E,
        semantic: &mut S,
        coupling: &mut C,
    ) -> MemoryResult<Vec<ConceptId>>
    where
        E: EpisodicStore,
        S: SemanticStore,
        C: MemoryCoupling,
    {
        let mut new_concepts = Vec::new();

        // Get recent episodes grouped by type
        let episodes = episodic.recent(1000)?;

        // Count word patterns across episodes
        let mut pattern_episodes: HashMap<String, Vec<EventId>> = HashMap::new();

        for ep in &episodes {
            let words: Vec<_> = ep
                .content
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 4)
                .map(|w| w.to_string())
                .collect();

            for word in words {
                pattern_episodes
                    .entry(word)
                    .or_default()
                    .push(ep.id.clone());
            }
        }

        // Patterns appearing in enough episodes become concepts
        for (pattern, ep_ids) in pattern_episodes {
            if ep_ids.len() >= self.min_episode_count {
                // Check if concept already exists
                if semantic.find_by_name(&pattern)?.is_some() {
                    continue;
                }

                let confidence = (ep_ids.len() as f32 / episodes.len() as f32).min(1.0);

                if confidence >= self.confidence_threshold {
                    let concept = Concept::new(pattern, vec![])
                        .with_confidence(confidence)
                        .with_description(format!("Abstracted from {} episodes", ep_ids.len()));

                    if let Ok(concept_id) = semantic.store(concept) {
                        // Associate with source episodes
                        for ep_id in ep_ids {
                            coupling.associate(ep_id, concept_id.clone())?;
                        }
                        new_concepts.push(concept_id);
                    }
                }
            }
        }

        Ok(new_concepts)
    }
}

impl Default for AbstractionEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::coupling::InMemoryCoupling;
    use super::super::episodic::InMemoryEpisodicStore;
    use super::super::semantic::InMemorySemanticStore;
    use super::*;

    fn create_test_stores() -> (
        InMemoryEpisodicStore,
        InMemorySemanticStore,
        InMemoryCoupling,
    ) {
        (
            InMemoryEpisodicStore::new(1000),
            InMemorySemanticStore::new(3, 500),
            InMemoryCoupling::new(),
        )
    }

    #[test]
    fn test_consolidation_config_default() {
        let config = ConsolidationConfig::default();
        assert_eq!(config.min_age_seconds, 300);
        assert!(config.extract_concepts);
        assert!(config.merge_similar);
    }

    #[test]
    fn test_consolidation_result_had_changes() {
        let mut result = ConsolidationResult::default();
        assert!(!result.had_changes());

        result.episodes_merged = 1;
        assert!(result.had_changes());
    }

    #[test]
    fn test_consolidator_new() {
        let consolidator = MemoryConsolidator::new();
        assert_eq!(consolidator.config.batch_size, 100);
    }

    #[test]
    fn test_consolidator_with_config() {
        let config = ConsolidationConfig {
            batch_size: 50,
            ..Default::default()
        };
        let consolidator = MemoryConsolidator::with_config(config);
        assert_eq!(consolidator.config.batch_size, 50);
    }

    #[test]
    fn test_extract_concepts_from_episode() {
        let consolidator = MemoryConsolidator::new();
        let episode = Episode::new(
            "The quick brown quick quick quick fox jumps over the lazy dog",
            EpisodeType::Interaction,
            "user",
        );

        let concepts = consolidator.extract_concepts_from_episode(&episode);
        assert!(!concepts.is_empty());
        // "quick" appears most frequently
        assert!(concepts.iter().any(|(name, _)| name == "quick"));
    }

    #[test]
    fn test_calculate_similarity() {
        let consolidator = MemoryConsolidator::new();

        let a = Episode::new("hello world test", EpisodeType::Interaction, "user");
        let b = Episode::new("hello world test", EpisodeType::Interaction, "user");

        let sim = consolidator.calculate_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);

        let c = Episode::new("goodbye universe", EpisodeType::Interaction, "user");
        let sim2 = consolidator.calculate_similarity(&a, &c);
        assert!(sim2 < 0.5);
    }

    #[test]
    fn test_merge_episodes() {
        let consolidator = MemoryConsolidator::new();

        let a =
            Episode::new("first content", EpisodeType::Interaction, "user").with_importance(0.6);
        let b =
            Episode::new("second content", EpisodeType::Interaction, "user").with_importance(0.8);

        let merged = consolidator.merge_episodes(&a, &b);
        assert!(merged.content.contains("first"));
        assert!(merged.content.contains("second"));
        assert!((merged.importance - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_consolidate_empty() {
        let (mut episodic, mut semantic, mut coupling) = create_test_stores();
        let consolidator = MemoryConsolidator::new();

        let result = consolidator
            .consolidate(&mut episodic, &mut semantic, &mut coupling)
            .unwrap();
        assert_eq!(result.episodes_processed, 0);
        assert!(!result.had_changes());
    }

    #[test]
    fn test_consolidate_with_episodes() {
        let (mut episodic, mut semantic, mut coupling) = create_test_stores();

        // Add some episodes
        episodic
            .record(Episode::new(
                "test content",
                EpisodeType::Interaction,
                "user",
            ))
            .unwrap();
        episodic
            .record(Episode::new(
                "another test",
                EpisodeType::Observation,
                "agent",
            ))
            .unwrap();

        // Use config with 0 min age so episodes can be processed immediately
        let config = ConsolidationConfig {
            min_age_seconds: 0,
            ..Default::default()
        };
        let consolidator = MemoryConsolidator::with_config(config);

        let result = consolidator
            .consolidate(&mut episodic, &mut semantic, &mut coupling)
            .unwrap();
        assert_eq!(result.episodes_processed, 2);
    }

    #[test]
    fn test_abstraction_engine_new() {
        let engine = AbstractionEngine::new();
        assert_eq!(engine.min_episode_count, 3);
    }

    #[test]
    fn test_abstraction_engine_with_min_episodes() {
        let engine = AbstractionEngine::new().with_min_episodes(5);
        assert_eq!(engine.min_episode_count, 5);
    }

    #[test]
    fn test_abstract_patterns_empty() {
        let (episodic, mut semantic, mut coupling) = create_test_stores();
        let engine = AbstractionEngine::new();

        let concepts = engine
            .abstract_patterns(&episodic, &mut semantic, &mut coupling)
            .unwrap();
        assert!(concepts.is_empty());
    }

    #[test]
    fn test_abstract_patterns_with_episodes() {
        let (mut episodic, mut semantic, mut coupling) = create_test_stores();

        // Add episodes with repeated pattern
        for i in 0..5 {
            episodic
                .record(Episode::new(
                    format!("testing repeated pattern {}", i),
                    EpisodeType::Interaction,
                    "user",
                ))
                .unwrap();
        }

        let engine = AbstractionEngine::new().with_min_episodes(3);

        let concepts = engine
            .abstract_patterns(&episodic, &mut semantic, &mut coupling)
            .unwrap();
        // Should extract "testing", "repeated", "pattern" as concepts
        assert!(!concepts.is_empty());
    }

    #[test]
    fn test_is_old_enough() {
        let consolidator = MemoryConsolidator::with_config(ConsolidationConfig {
            min_age_seconds: 0,
            ..Default::default()
        });

        let episode = Episode::new("test", EpisodeType::Interaction, "user");
        assert!(consolidator.is_old_enough(&episode));
    }
}
