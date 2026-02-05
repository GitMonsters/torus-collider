//! # Episodic Memory Store
//!
//! Event trace storage with temporal queries and similarity search.
//! Supports both in-memory and SQLite-backed implementations.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{EventId, MemoryError, MemoryResult, TimeRange};

// =============================================================================
// EPISODE TYPE
// =============================================================================

/// An episode represents a discrete memory unit (event trace)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique identifier
    pub id: EventId,
    /// Main content of the episode
    pub content: String,
    /// Episode type/category
    pub episode_type: EpisodeType,
    /// When this episode occurred
    pub timestamp: DateTime<Utc>,
    /// Associated run ID (if from agent execution)
    pub run_id: Option<String>,
    /// Source of the episode
    pub source: String,
    /// Importance score (0.0 - 1.0)
    pub importance: f32,
    /// Access count for frequency-based retrieval
    pub access_count: u32,
    /// Last accessed time
    pub last_accessed: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of episodes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeType {
    /// User interaction
    Interaction,
    /// Tool execution
    ToolExecution,
    /// LLM response
    LLMResponse,
    /// Observation/perception
    Observation,
    /// Decision made
    Decision,
    /// Error/failure
    Error,
    /// Learning event
    Learning,
    /// Custom type
    Custom(String),
}

impl Episode {
    /// Create a new episode
    pub fn new(
        content: impl Into<String>,
        episode_type: EpisodeType,
        source: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: EventId::generate(),
            content: content.into(),
            episode_type,
            timestamp: now,
            run_id: None,
            source: source.into(),
            importance: 0.5,
            access_count: 0,
            last_accessed: now,
            metadata: HashMap::new(),
        }
    }

    /// Create an interaction episode
    pub fn interaction(content: impl Into<String>, source: impl Into<String>) -> Self {
        Self::new(content, EpisodeType::Interaction, source)
    }

    /// Create a tool execution episode
    pub fn tool_execution(
        tool_name: impl Into<String>,
        input: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        let tool = tool_name.into();
        let content = format!(
            "Tool: {}\nInput: {}\nOutput: {}",
            tool,
            input.into(),
            output.into()
        );
        let mut ep = Self::new(content, EpisodeType::ToolExecution, &tool);
        ep.metadata.insert("tool_name".to_string(), tool);
        ep
    }

    /// Create an observation episode
    pub fn observation(content: impl Into<String>, source: impl Into<String>) -> Self {
        Self::new(content, EpisodeType::Observation, source)
    }

    /// Create a decision episode
    pub fn decision(content: impl Into<String>, reason: impl Into<String>) -> Self {
        let mut ep = Self::new(content, EpisodeType::Decision, "agent");
        ep.metadata.insert("reason".to_string(), reason.into());
        ep
    }

    /// Create a learning episode
    pub fn learning(content: impl Into<String>, source: impl Into<String>) -> Self {
        Self::new(content, EpisodeType::Learning, source)
    }

    /// Set importance score
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Set run ID
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Mark as accessed (updates count and timestamp)
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = Utc::now();
    }

    /// Calculate age in seconds
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.timestamp).num_seconds()
    }
}

// =============================================================================
// EPISODIC STORE TRAIT
// =============================================================================

/// Trait for episodic memory storage
pub trait EpisodicStore: Send + Sync {
    /// Record a new episode
    fn record(&mut self, episode: Episode) -> MemoryResult<EventId>;

    /// Retrieve an episode by ID
    fn get(&self, id: &EventId) -> MemoryResult<Episode>;

    /// Retrieve an episode by ID and mark as accessed
    fn get_and_access(&mut self, id: &EventId) -> MemoryResult<Episode>;

    /// Find similar episodes by content
    fn recall_similar(&self, query: &str, k: usize) -> MemoryResult<Vec<Episode>>;

    /// Query episodes by time range
    fn temporal_query(&self, range: &TimeRange) -> MemoryResult<Vec<Episode>>;

    /// Query episodes by type
    fn query_by_type(&self, episode_type: &EpisodeType) -> MemoryResult<Vec<Episode>>;

    /// Query episodes by run ID
    fn query_by_run(&self, run_id: &str) -> MemoryResult<Vec<Episode>>;

    /// Get most recent episodes
    fn recent(&self, k: usize) -> MemoryResult<Vec<Episode>>;

    /// Get most important episodes
    fn most_important(&self, k: usize) -> MemoryResult<Vec<Episode>>;

    /// Get most accessed episodes
    fn most_accessed(&self, k: usize) -> MemoryResult<Vec<Episode>>;

    /// Update an episode
    fn update(&mut self, episode: Episode) -> MemoryResult<()>;

    /// Delete an episode
    fn delete(&mut self, id: &EventId) -> MemoryResult<()>;

    /// Get total number of episodes
    fn count(&self) -> usize;

    /// Clear all episodes
    fn clear(&mut self);
}

// =============================================================================
// IN-MEMORY IMPLEMENTATION
// =============================================================================

/// In-memory episodic store (for testing and lightweight use)
pub struct InMemoryEpisodicStore {
    /// Episodes indexed by ID
    episodes: HashMap<EventId, Episode>,
    /// Maximum number of episodes to store
    max_size: usize,
    /// Order of insertion for LRU eviction
    insertion_order: Vec<EventId>,
}

impl InMemoryEpisodicStore {
    /// Create a new in-memory store
    pub fn new(max_size: usize) -> Self {
        Self {
            episodes: HashMap::new(),
            max_size,
            insertion_order: Vec::new(),
        }
    }

    /// Evict oldest episodes if over capacity
    fn evict_if_needed(&mut self) {
        while self.episodes.len() >= self.max_size && !self.insertion_order.is_empty() {
            if let Some(oldest_id) = self.insertion_order.first().cloned() {
                self.episodes.remove(&oldest_id);
                self.insertion_order.remove(0);
            }
        }
    }

    /// Simple text similarity (Jaccard on words)
    fn text_similarity(a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        let a_words: std::collections::HashSet<_> = a_lower.split_whitespace().collect();
        let b_words: std::collections::HashSet<_> = b_lower.split_whitespace().collect();

        if a_words.is_empty() || b_words.is_empty() {
            return 0.0;
        }

        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();

        intersection as f32 / union as f32
    }
}

impl EpisodicStore for InMemoryEpisodicStore {
    fn record(&mut self, mut episode: Episode) -> MemoryResult<EventId> {
        self.evict_if_needed();

        let id = episode.id.clone();
        episode.timestamp = Utc::now();
        episode.last_accessed = Utc::now();

        self.insertion_order.push(id.clone());
        self.episodes.insert(id.clone(), episode);

        Ok(id)
    }

    fn get(&self, id: &EventId) -> MemoryResult<Episode> {
        self.episodes
            .get(id)
            .cloned()
            .ok_or_else(|| MemoryError::EpisodeNotFound(id.clone()))
    }

    fn get_and_access(&mut self, id: &EventId) -> MemoryResult<Episode> {
        let episode = self
            .episodes
            .get_mut(id)
            .ok_or_else(|| MemoryError::EpisodeNotFound(id.clone()))?;
        episode.mark_accessed();
        Ok(episode.clone())
    }

    fn recall_similar(&self, query: &str, k: usize) -> MemoryResult<Vec<Episode>> {
        let mut scored: Vec<_> = self
            .episodes
            .values()
            .map(|ep| {
                let score = Self::text_similarity(query, &ep.content);
                (score, ep.clone())
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored.into_iter().take(k).map(|(_, ep)| ep).collect())
    }

    fn temporal_query(&self, range: &TimeRange) -> MemoryResult<Vec<Episode>> {
        let mut episodes: Vec<_> = self
            .episodes
            .values()
            .filter(|ep| range.contains(&ep.timestamp))
            .cloned()
            .collect();

        // Sort by timestamp ascending
        episodes.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(episodes)
    }

    fn query_by_type(&self, episode_type: &EpisodeType) -> MemoryResult<Vec<Episode>> {
        let mut episodes: Vec<_> = self
            .episodes
            .values()
            .filter(|ep| &ep.episode_type == episode_type)
            .cloned()
            .collect();

        episodes.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(episodes)
    }

    fn query_by_run(&self, run_id: &str) -> MemoryResult<Vec<Episode>> {
        let mut episodes: Vec<_> = self
            .episodes
            .values()
            .filter(|ep| ep.run_id.as_deref() == Some(run_id))
            .cloned()
            .collect();

        episodes.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(episodes)
    }

    fn recent(&self, k: usize) -> MemoryResult<Vec<Episode>> {
        let mut episodes: Vec<_> = self.episodes.values().cloned().collect();
        episodes.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(episodes.into_iter().take(k).collect())
    }

    fn most_important(&self, k: usize) -> MemoryResult<Vec<Episode>> {
        let mut episodes: Vec<_> = self.episodes.values().cloned().collect();
        episodes.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(episodes.into_iter().take(k).collect())
    }

    fn most_accessed(&self, k: usize) -> MemoryResult<Vec<Episode>> {
        let mut episodes: Vec<_> = self.episodes.values().cloned().collect();
        episodes.sort_by(|a, b| b.access_count.cmp(&a.access_count));
        Ok(episodes.into_iter().take(k).collect())
    }

    fn update(&mut self, episode: Episode) -> MemoryResult<()> {
        if !self.episodes.contains_key(&episode.id) {
            return Err(MemoryError::EpisodeNotFound(episode.id.clone()));
        }
        self.episodes.insert(episode.id.clone(), episode);
        Ok(())
    }

    fn delete(&mut self, id: &EventId) -> MemoryResult<()> {
        if self.episodes.remove(id).is_none() {
            return Err(MemoryError::EpisodeNotFound(id.clone()));
        }
        self.insertion_order.retain(|i| i != id);
        Ok(())
    }

    fn count(&self) -> usize {
        self.episodes.len()
    }

    fn clear(&mut self) {
        self.episodes.clear();
        self.insertion_order.clear();
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_episode_new() {
        let ep = Episode::new("test content", EpisodeType::Interaction, "user");
        assert_eq!(ep.content, "test content");
        assert_eq!(ep.episode_type, EpisodeType::Interaction);
        assert_eq!(ep.source, "user");
    }

    #[test]
    fn test_episode_interaction() {
        let ep = Episode::interaction("hello", "user");
        assert_eq!(ep.episode_type, EpisodeType::Interaction);
    }

    #[test]
    fn test_episode_tool_execution() {
        let ep = Episode::tool_execution("bash", "ls -la", "file1\nfile2");
        assert_eq!(ep.episode_type, EpisodeType::ToolExecution);
        assert!(ep.content.contains("bash"));
        assert!(ep.content.contains("ls -la"));
        assert_eq!(ep.metadata.get("tool_name").unwrap(), "bash");
    }

    #[test]
    fn test_episode_with_importance() {
        let ep = Episode::new("test", EpisodeType::Decision, "agent").with_importance(0.9);
        assert!((ep.importance - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_episode_with_importance_clamped() {
        let ep = Episode::new("test", EpisodeType::Decision, "agent").with_importance(1.5);
        assert!((ep.importance - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_episode_with_run_id() {
        let ep = Episode::new("test", EpisodeType::Interaction, "user").with_run_id("run-123");
        assert_eq!(ep.run_id, Some("run-123".to_string()));
    }

    #[test]
    fn test_episode_mark_accessed() {
        let mut ep = Episode::new("test", EpisodeType::Interaction, "user");
        assert_eq!(ep.access_count, 0);
        ep.mark_accessed();
        assert_eq!(ep.access_count, 1);
        ep.mark_accessed();
        assert_eq!(ep.access_count, 2);
    }

    #[test]
    fn test_inmemory_record_and_get() {
        let mut store = InMemoryEpisodicStore::new(100);
        let ep = Episode::new("test content", EpisodeType::Interaction, "user");
        let id = store.record(ep.clone()).unwrap();

        let retrieved = store.get(&id).unwrap();
        assert_eq!(retrieved.content, "test content");
    }

    #[test]
    fn test_inmemory_get_not_found() {
        let store = InMemoryEpisodicStore::new(100);
        let result = store.get(&EventId::new("nonexistent"));
        assert!(matches!(result, Err(MemoryError::EpisodeNotFound(_))));
    }

    #[test]
    fn test_inmemory_get_and_access() {
        let mut store = InMemoryEpisodicStore::new(100);
        let ep = Episode::new("test", EpisodeType::Interaction, "user");
        let id = store.record(ep).unwrap();

        let retrieved = store.get_and_access(&id).unwrap();
        assert_eq!(retrieved.access_count, 1);

        let retrieved2 = store.get_and_access(&id).unwrap();
        assert_eq!(retrieved2.access_count, 2);
    }

    #[test]
    fn test_inmemory_recall_similar() {
        let mut store = InMemoryEpisodicStore::new(100);

        store
            .record(Episode::new(
                "The quick brown fox",
                EpisodeType::Interaction,
                "user",
            ))
            .unwrap();
        store
            .record(Episode::new(
                "The lazy brown dog",
                EpisodeType::Interaction,
                "user",
            ))
            .unwrap();
        store
            .record(Episode::new(
                "Something completely different",
                EpisodeType::Interaction,
                "user",
            ))
            .unwrap();

        let results = store.recall_similar("brown fox", 2).unwrap();
        assert_eq!(results.len(), 2);
        // First result should be most similar
        assert!(results[0].content.contains("fox"));
    }

    #[test]
    fn test_inmemory_temporal_query() {
        let mut store = InMemoryEpisodicStore::new(100);

        store
            .record(Episode::new("event 1", EpisodeType::Interaction, "user"))
            .unwrap();
        store
            .record(Episode::new("event 2", EpisodeType::Interaction, "user"))
            .unwrap();

        let range = TimeRange::last(Duration::from_secs(60));
        let results = store.temporal_query(&range).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_inmemory_query_by_type() {
        let mut store = InMemoryEpisodicStore::new(100);

        store
            .record(Episode::new(
                "interaction",
                EpisodeType::Interaction,
                "user",
            ))
            .unwrap();
        store
            .record(Episode::new("decision", EpisodeType::Decision, "agent"))
            .unwrap();
        store
            .record(Episode::new(
                "interaction 2",
                EpisodeType::Interaction,
                "user",
            ))
            .unwrap();

        let results = store.query_by_type(&EpisodeType::Interaction).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_inmemory_query_by_run() {
        let mut store = InMemoryEpisodicStore::new(100);

        store
            .record(Episode::new("ep1", EpisodeType::Interaction, "user").with_run_id("run-1"))
            .unwrap();
        store
            .record(Episode::new("ep2", EpisodeType::Interaction, "user").with_run_id("run-2"))
            .unwrap();
        store
            .record(Episode::new("ep3", EpisodeType::Interaction, "user").with_run_id("run-1"))
            .unwrap();

        let results = store.query_by_run("run-1").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_inmemory_recent() {
        let mut store = InMemoryEpisodicStore::new(100);

        store
            .record(Episode::new("old", EpisodeType::Interaction, "user"))
            .unwrap();
        store
            .record(Episode::new("newer", EpisodeType::Interaction, "user"))
            .unwrap();
        store
            .record(Episode::new("newest", EpisodeType::Interaction, "user"))
            .unwrap();

        let results = store.recent(2).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].content == "newest");
    }

    #[test]
    fn test_inmemory_most_important() {
        let mut store = InMemoryEpisodicStore::new(100);

        store
            .record(Episode::new("low", EpisodeType::Decision, "agent").with_importance(0.1))
            .unwrap();
        store
            .record(Episode::new("high", EpisodeType::Decision, "agent").with_importance(0.9))
            .unwrap();
        store
            .record(Episode::new("medium", EpisodeType::Decision, "agent").with_importance(0.5))
            .unwrap();

        let results = store.most_important(2).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].importance > results[1].importance);
    }

    #[test]
    fn test_inmemory_update() {
        let mut store = InMemoryEpisodicStore::new(100);
        let ep = Episode::new("original", EpisodeType::Interaction, "user");
        let id = store.record(ep).unwrap();

        let mut updated = store.get(&id).unwrap();
        updated.content = "updated".to_string();
        store.update(updated).unwrap();

        let retrieved = store.get(&id).unwrap();
        assert_eq!(retrieved.content, "updated");
    }

    #[test]
    fn test_inmemory_delete() {
        let mut store = InMemoryEpisodicStore::new(100);
        let ep = Episode::new("test", EpisodeType::Interaction, "user");
        let id = store.record(ep).unwrap();

        assert_eq!(store.count(), 1);
        store.delete(&id).unwrap();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_inmemory_eviction() {
        let mut store = InMemoryEpisodicStore::new(3);

        let id1 = store
            .record(Episode::new("first", EpisodeType::Interaction, "user"))
            .unwrap();
        store
            .record(Episode::new("second", EpisodeType::Interaction, "user"))
            .unwrap();
        store
            .record(Episode::new("third", EpisodeType::Interaction, "user"))
            .unwrap();
        store
            .record(Episode::new("fourth", EpisodeType::Interaction, "user"))
            .unwrap();

        assert_eq!(store.count(), 3);
        // First should be evicted
        assert!(store.get(&id1).is_err());
    }

    #[test]
    fn test_inmemory_clear() {
        let mut store = InMemoryEpisodicStore::new(100);
        store
            .record(Episode::new("test1", EpisodeType::Interaction, "user"))
            .unwrap();
        store
            .record(Episode::new("test2", EpisodeType::Interaction, "user"))
            .unwrap();

        assert_eq!(store.count(), 2);
        store.clear();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_text_similarity() {
        let sim = InMemoryEpisodicStore::text_similarity("hello world", "hello world");
        assert!((sim - 1.0).abs() < 0.001);

        let sim2 = InMemoryEpisodicStore::text_similarity("hello world", "goodbye world");
        assert!(sim2 > 0.0 && sim2 < 1.0);

        let sim3 = InMemoryEpisodicStore::text_similarity("hello", "goodbye");
        assert!((sim3 - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_episode_age_seconds() {
        let ep = Episode::new("test", EpisodeType::Interaction, "user");
        // Just created, should be very close to 0
        assert!(ep.age_seconds() < 2);
    }
}
