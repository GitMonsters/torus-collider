//! # Memory System for Agent Framework
//!
//! Phase 6: Persistent episodic + semantic memory with cross-referencing
//! and consolidation. This is the critical capability for Proto-AGI (Tier 3).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      MemorySystem                                │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
//! │  │  EpisodicStore  │←→│  MemoryCoupling │←→│  SemanticStore  │  │
//! │  │  (event traces) │  │  (associations) │  │  (concepts)     │  │
//! │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
//! │                              ↓                                   │
//! │                    ┌─────────────────┐                          │
//! │                    │  Consolidation  │                          │
//! │                    │  (compression)  │                          │
//! │                    └─────────────────┘                          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **Episodic Memory**: Event trace storage with temporal queries
//! - **Semantic Memory**: Vector embeddings for concepts and knowledge
//! - **Memory Coupling**: Bidirectional episode↔concept associations
//! - **Consolidation**: Background compression and abstraction

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub mod consolidation;
pub mod coupling;
pub mod episodic;
pub mod integration;
pub mod semantic;

// Re-exports
pub use consolidation::{ConsolidationConfig, ConsolidationResult, MemoryConsolidator};
pub use coupling::{Association, AssociationType, InMemoryCoupling, MemoryCoupling};
pub use episodic::{Episode, EpisodeType, EpisodicStore, InMemoryEpisodicStore};
pub use integration::{
    create_compounding_memory_bridge, create_compounding_memory_bridge_with_consolidation,
    CoherenceScorer, CompoundingAware, IntegrationConsolidationResult, IntegrationStats,
    MemoryBridge, StreamGraphAdapter,
};
pub use semantic::{Concept, ConceptRelation, InMemorySemanticStore, RelationType, SemanticStore};

// =============================================================================
// CORE ID TYPES
// =============================================================================

/// Unique identifier for an episode
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub String);

impl EventId {
    /// Create a new event ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a new unique event ID
    pub fn generate() -> Self {
        let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let random: u32 = rand::random();
        Self(format!("evt-{}-{:08x}", timestamp, random))
    }
}

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a concept
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConceptId(pub String);

impl ConceptId {
    /// Create a new concept ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a new unique concept ID
    pub fn generate() -> Self {
        let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let random: u32 = rand::random();
        Self(format!("cpt-{}-{:08x}", timestamp, random))
    }
}

impl fmt::Display for ConceptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// TEMPORAL TYPES
// =============================================================================

/// A time range for temporal queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start of the range (inclusive)
    pub start: DateTime<Utc>,
    /// End of the range (inclusive)
    pub end: DateTime<Utc>,
}

impl TimeRange {
    /// Create a new time range
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self { start, end }
    }

    /// Create a range for the last N duration
    pub fn last(duration: Duration) -> Self {
        let now = Utc::now();
        let start = now - chrono::Duration::from_std(duration).unwrap_or_default();
        Self { start, end: now }
    }

    /// Check if a timestamp falls within this range
    pub fn contains(&self, timestamp: &DateTime<Utc>) -> bool {
        *timestamp >= self.start && *timestamp <= self.end
    }

    /// Duration of this range
    pub fn duration(&self) -> Duration {
        (self.end - self.start).to_std().unwrap_or(Duration::ZERO)
    }
}

impl Default for TimeRange {
    fn default() -> Self {
        Self::last(Duration::from_secs(3600)) // Last hour
    }
}

// =============================================================================
// MEMORY ERROR TYPES
// =============================================================================

/// Errors that can occur in the memory system
#[derive(Debug, Clone)]
pub enum MemoryError {
    /// Episode not found
    EpisodeNotFound(EventId),
    /// Concept not found
    ConceptNotFound(ConceptId),
    /// Storage error
    Storage(String),
    /// Invalid query
    InvalidQuery(String),
    /// Embedding dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
    /// Consolidation error
    Consolidation(String),
    /// Database error (for SQLite backend)
    Database(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EpisodeNotFound(id) => write!(f, "Episode not found: {}", id),
            Self::ConceptNotFound(id) => write!(f, "Concept not found: {}", id),
            Self::Storage(msg) => write!(f, "Storage error: {}", msg),
            Self::InvalidQuery(msg) => write!(f, "Invalid query: {}", msg),
            Self::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Embedding dimension mismatch: expected {}, got {}",
                    expected, got
                )
            }
            Self::Consolidation(msg) => write!(f, "Consolidation error: {}", msg),
            Self::Database(msg) => write!(f, "Database error: {}", msg),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Result type for memory operations
pub type MemoryResult<T> = Result<T, MemoryError>;

// =============================================================================
// RELEVANCE SCORING
// =============================================================================

/// Relevance score for memory retrieval
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RelevanceScore {
    /// Overall score (0.0 - 1.0)
    pub score: f32,
    /// Semantic similarity component
    pub semantic_similarity: f32,
    /// Temporal recency component
    pub recency: f32,
    /// Access frequency component
    pub frequency: f32,
}

impl RelevanceScore {
    /// Create a new relevance score
    pub fn new(semantic_similarity: f32, recency: f32, frequency: f32) -> Self {
        // Weighted combination
        let score = 0.5 * semantic_similarity + 0.3 * recency + 0.2 * frequency;
        Self {
            score: score.clamp(0.0, 1.0),
            semantic_similarity,
            recency,
            frequency,
        }
    }

    /// Create from a single similarity score
    pub fn from_similarity(similarity: f32) -> Self {
        Self {
            score: similarity.clamp(0.0, 1.0),
            semantic_similarity: similarity,
            recency: 0.0,
            frequency: 0.0,
        }
    }
}

impl Default for RelevanceScore {
    fn default() -> Self {
        Self {
            score: 0.0,
            semantic_similarity: 0.0,
            recency: 0.0,
            frequency: 0.0,
        }
    }
}

// =============================================================================
// MEMORY SYSTEM (UNIFIED INTERFACE)
// =============================================================================

/// Configuration for the memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Embedding dimension for semantic memory
    pub embedding_dim: usize,
    /// Maximum number of episodes to store
    pub max_episodes: usize,
    /// Maximum number of concepts to store
    pub max_concepts: usize,
    /// Enable automatic consolidation
    pub auto_consolidate: bool,
    /// Consolidation threshold (number of episodes before triggering)
    pub consolidation_threshold: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384, // Common embedding dimension
            max_episodes: 10000,
            max_concepts: 5000,
            auto_consolidate: true,
            consolidation_threshold: 1000,
        }
    }
}

/// Unified memory system combining episodic, semantic, and coupling
pub struct MemorySystem<E, S, C>
where
    E: EpisodicStore,
    S: SemanticStore,
    C: MemoryCoupling,
{
    /// Episodic memory store
    pub episodic: E,
    /// Semantic memory store
    pub semantic: S,
    /// Memory coupling layer
    pub coupling: C,
    /// Configuration
    pub config: MemoryConfig,
    /// Statistics
    stats: MemoryStats,
}

/// Statistics for memory system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total episodes recorded
    pub episodes_recorded: usize,
    /// Total concepts stored
    pub concepts_stored: usize,
    /// Total associations created
    pub associations_created: usize,
    /// Total queries performed
    pub queries_performed: usize,
    /// Total consolidations run
    pub consolidations_run: usize,
}

impl<E, S, C> MemorySystem<E, S, C>
where
    E: EpisodicStore,
    S: SemanticStore,
    C: MemoryCoupling,
{
    /// Create a new memory system
    pub fn new(episodic: E, semantic: S, coupling: C, config: MemoryConfig) -> Self {
        Self {
            episodic,
            semantic,
            coupling,
            config,
            stats: MemoryStats::default(),
        }
    }

    /// Record an episode and extract concepts
    pub fn record(&mut self, episode: Episode) -> MemoryResult<EventId> {
        // Store the episode
        let event_id = self.episodic.record(episode.clone())?;
        self.stats.episodes_recorded += 1;

        // Extract concepts from episode content (simple keyword extraction)
        let keywords = Self::extract_keywords(&episode.content);

        for keyword in keywords {
            // Check if concept exists or create new one
            let concept = Concept::new(keyword.clone(), vec![]);
            if let Ok(concept_id) = self.semantic.store(concept) {
                self.stats.concepts_stored += 1;

                // Create association
                let _ = self.coupling.associate(event_id.clone(), concept_id)?;
                self.stats.associations_created += 1;
            }
        }

        Ok(event_id)
    }

    /// Query memory by semantic similarity
    pub fn query(&mut self, query: &str, k: usize) -> MemoryResult<Vec<(Episode, RelevanceScore)>> {
        self.stats.queries_performed += 1;

        // Search episodes by content similarity
        let episodes = self.episodic.recall_similar(query, k)?;

        // Calculate relevance scores
        let now = Utc::now();
        let results: Vec<_> = episodes
            .into_iter()
            .map(|ep| {
                let recency = Self::calculate_recency(&ep.timestamp, &now);
                let score = RelevanceScore::new(0.5, recency, 0.3); // Default semantic score
                (ep, score)
            })
            .collect();

        Ok(results)
    }

    /// Query memory by concept
    pub fn query_by_concept(&mut self, concept_id: &ConceptId) -> MemoryResult<Vec<Episode>> {
        self.stats.queries_performed += 1;
        let event_ids = self.coupling.episode_ids_for_concept(concept_id)?;
        let mut episodes = Vec::new();
        for event_id in event_ids {
            if let Ok(episode) = self.episodic.get(&event_id) {
                episodes.push(episode);
            }
        }
        Ok(episodes)
    }

    /// Query memory by time range
    pub fn query_temporal(&mut self, range: &TimeRange) -> MemoryResult<Vec<Episode>> {
        self.stats.queries_performed += 1;
        self.episodic.temporal_query(range)
    }

    /// Get statistics
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Simple keyword extraction (placeholder for more sophisticated NLP)
    fn extract_keywords(content: &str) -> Vec<String> {
        content
            .split_whitespace()
            .filter(|w| w.len() > 3) // Skip short words
            .map(|w| w.to_lowercase())
            .collect()
    }

    /// Calculate recency score (1.0 for now, decays over time)
    fn calculate_recency(timestamp: &DateTime<Utc>, now: &DateTime<Utc>) -> f32 {
        let age = (*now - *timestamp).num_seconds() as f32;
        let decay_rate = 0.0001; // Decay constant
        (-decay_rate * age).exp()
    }
}

/// Create a default in-memory memory system
pub fn create_in_memory_system(
    config: MemoryConfig,
) -> MemorySystem<InMemoryEpisodicStore, InMemorySemanticStore, InMemoryCoupling> {
    let episodic = InMemoryEpisodicStore::new(config.max_episodes);
    let semantic = InMemorySemanticStore::new(config.embedding_dim, config.max_concepts);
    let coupling = InMemoryCoupling::new();
    MemorySystem::new(episodic, semantic, coupling, config)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_id_generate() {
        let id1 = EventId::generate();
        let id2 = EventId::generate();
        assert!(id1.0.starts_with("evt-"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_concept_id_generate() {
        let id1 = ConceptId::generate();
        let id2 = ConceptId::generate();
        assert!(id1.0.starts_with("cpt-"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_time_range_contains() {
        let now = Utc::now();
        // Create a range that definitely contains 'now'
        let start = now - chrono::Duration::hours(1);
        let end = now + chrono::Duration::hours(1);
        let range = TimeRange::new(start, end);
        assert!(range.contains(&now));

        let old = now - chrono::Duration::hours(2);
        assert!(!range.contains(&old));

        let future = now + chrono::Duration::hours(2);
        assert!(!range.contains(&future));
    }

    #[test]
    fn test_time_range_duration() {
        let range = TimeRange::last(Duration::from_secs(3600));
        let dur = range.duration();
        // Should be approximately 1 hour (with small variance for execution time)
        assert!(dur.as_secs() >= 3599 && dur.as_secs() <= 3601);
    }

    #[test]
    fn test_relevance_score_new() {
        let score = RelevanceScore::new(0.8, 0.6, 0.4);
        // 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4 = 0.4 + 0.18 + 0.08 = 0.66
        assert!((score.score - 0.66).abs() < 0.001);
    }

    #[test]
    fn test_relevance_score_from_similarity() {
        let score = RelevanceScore::from_similarity(0.9);
        assert!((score.score - 0.9).abs() < 0.001);
        assert!((score.semantic_similarity - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_memory_error_display() {
        let err = MemoryError::EpisodeNotFound(EventId::new("test-123"));
        assert!(err.to_string().contains("test-123"));
    }

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert_eq!(config.embedding_dim, 384);
        assert_eq!(config.max_episodes, 10000);
        assert!(config.auto_consolidate);
    }

    #[test]
    fn test_extract_keywords() {
        let keywords = MemorySystem::<InMemoryEpisodicStore, InMemorySemanticStore, InMemoryCoupling>::extract_keywords(
            "The quick brown fox jumps over the lazy dog"
        );
        assert!(keywords.contains(&"quick".to_string()));
        assert!(keywords.contains(&"brown".to_string()));
        assert!(!keywords.contains(&"the".to_string())); // Too short
    }

    #[test]
    fn test_calculate_recency() {
        let now = Utc::now();
        let recent = now - chrono::Duration::seconds(10);
        let old = now - chrono::Duration::hours(24);

        let recent_score = MemorySystem::<
            InMemoryEpisodicStore,
            InMemorySemanticStore,
            InMemoryCoupling,
        >::calculate_recency(&recent, &now);
        let old_score = MemorySystem::<
            InMemoryEpisodicStore,
            InMemorySemanticStore,
            InMemoryCoupling,
        >::calculate_recency(&old, &now);

        assert!(recent_score > old_score);
        assert!(recent_score > 0.99); // Very recent
    }
}
