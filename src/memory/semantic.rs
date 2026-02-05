//! # Semantic Memory Store
//!
//! Vector embeddings for concepts and knowledge representation.
//! Supports nearest neighbor search and concept relations.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{ConceptId, MemoryError, MemoryResult};

// =============================================================================
// CONCEPT TYPE
// =============================================================================

/// A concept represents a semantic unit in memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Unique identifier
    pub id: ConceptId,
    /// Human-readable name/label
    pub name: String,
    /// Description or definition
    pub description: String,
    /// Vector embedding (may be empty if not computed)
    pub embedding: Vec<f32>,
    /// When this concept was created
    pub created_at: DateTime<Utc>,
    /// When this concept was last updated
    pub updated_at: DateTime<Utc>,
    /// Confidence/certainty score (0.0 - 1.0)
    pub confidence: f32,
    /// Access count for frequency-based retrieval
    pub access_count: u32,
    /// Relations to other concepts
    pub relations: Vec<ConceptRelation>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// A relation between concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptRelation {
    /// Target concept ID
    pub target: ConceptId,
    /// Relation type
    pub relation_type: RelationType,
    /// Strength of the relation (0.0 - 1.0)
    pub strength: f32,
}

/// Types of relations between concepts
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    /// Parent/child hierarchy (is-a)
    IsA,
    /// Part-of relation
    PartOf,
    /// Has-property relation
    HasProperty,
    /// Causes or leads to
    Causes,
    /// Similar/related to
    SimilarTo,
    /// Opposite of
    OppositeOf,
    /// Instance of (class-instance relation)
    InstanceOf,
    /// Used for/purpose
    UsedFor,
    /// Custom relation
    Custom(String),
}

impl Concept {
    /// Create a new concept
    pub fn new(name: impl Into<String>, embedding: Vec<f32>) -> Self {
        let now = Utc::now();
        Self {
            id: ConceptId::generate(),
            name: name.into(),
            description: String::new(),
            embedding,
            created_at: now,
            updated_at: now,
            confidence: 1.0,
            access_count: 0,
            relations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a concept with ID
    pub fn with_id(id: ConceptId, name: impl Into<String>, embedding: Vec<f32>) -> Self {
        let mut concept = Self::new(name, embedding);
        concept.id = id;
        concept
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add a relation
    pub fn with_relation(
        mut self,
        target: ConceptId,
        relation_type: RelationType,
        strength: f32,
    ) -> Self {
        self.relations.push(ConceptRelation {
            target,
            relation_type,
            strength: strength.clamp(0.0, 1.0),
        });
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Mark as accessed
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
    }

    /// Update the concept
    pub fn update(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Check if this concept has an embedding
    pub fn has_embedding(&self) -> bool {
        !self.embedding.is_empty()
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding.len()
    }

    /// Add a relation to another concept
    pub fn add_relation(&mut self, target: ConceptId, relation_type: RelationType, strength: f32) {
        self.relations.push(ConceptRelation {
            target,
            relation_type,
            strength: strength.clamp(0.0, 1.0),
        });
        self.update();
    }

    /// Remove a relation by target
    pub fn remove_relation(&mut self, target: &ConceptId) {
        self.relations.retain(|r| &r.target != target);
        self.update();
    }

    /// Get relations of a specific type
    pub fn relations_of_type(&self, relation_type: &RelationType) -> Vec<&ConceptRelation> {
        self.relations
            .iter()
            .filter(|r| &r.relation_type == relation_type)
            .collect()
    }
}

impl ConceptRelation {
    /// Create a new relation
    pub fn new(target: ConceptId, relation_type: RelationType, strength: f32) -> Self {
        Self {
            target,
            relation_type,
            strength: strength.clamp(0.0, 1.0),
        }
    }
}

// =============================================================================
// SEMANTIC STORE TRAIT
// =============================================================================

/// Trait for semantic memory storage
pub trait SemanticStore: Send + Sync {
    /// Store a new concept
    fn store(&mut self, concept: Concept) -> MemoryResult<ConceptId>;

    /// Retrieve a concept by ID
    fn get(&self, id: &ConceptId) -> MemoryResult<Concept>;

    /// Retrieve a concept by ID and mark as accessed
    fn get_and_access(&mut self, id: &ConceptId) -> MemoryResult<Concept>;

    /// Find a concept by name
    fn find_by_name(&self, name: &str) -> MemoryResult<Option<Concept>>;

    /// Find concepts matching a name pattern
    fn search_by_name(&self, pattern: &str) -> MemoryResult<Vec<Concept>>;

    /// Find nearest concepts by embedding
    fn nearest(&self, embedding: &[f32], k: usize) -> MemoryResult<Vec<(Concept, f32)>>;

    /// Find concepts related to a given concept
    fn related_to(
        &self,
        id: &ConceptId,
        relation_type: Option<&RelationType>,
    ) -> MemoryResult<Vec<Concept>>;

    /// Update a concept
    fn update(&mut self, concept: Concept) -> MemoryResult<()>;

    /// Delete a concept
    fn delete(&mut self, id: &ConceptId) -> MemoryResult<()>;

    /// Get total number of concepts
    fn count(&self) -> usize;

    /// Get all concept IDs
    fn all_ids(&self) -> Vec<ConceptId>;

    /// Clear all concepts
    fn clear(&mut self);
}

// =============================================================================
// IN-MEMORY IMPLEMENTATION
// =============================================================================

/// In-memory semantic store (for testing and lightweight use)
pub struct InMemorySemanticStore {
    /// Concepts indexed by ID
    concepts: HashMap<ConceptId, Concept>,
    /// Name to ID index for fast lookup
    name_index: HashMap<String, ConceptId>,
    /// Expected embedding dimension
    embedding_dim: usize,
    /// Maximum number of concepts
    max_size: usize,
}

impl InMemorySemanticStore {
    /// Create a new in-memory store
    pub fn new(embedding_dim: usize, max_size: usize) -> Self {
        Self {
            concepts: HashMap::new(),
            name_index: HashMap::new(),
            embedding_dim,
            max_size,
        }
    }

    /// Cosine similarity between two embeddings
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Euclidean distance between two embeddings
    #[allow(dead_code)]
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl SemanticStore for InMemorySemanticStore {
    fn store(&mut self, mut concept: Concept) -> MemoryResult<ConceptId> {
        // Validate embedding dimension if provided
        if !concept.embedding.is_empty() && concept.embedding.len() != self.embedding_dim {
            return Err(MemoryError::DimensionMismatch {
                expected: self.embedding_dim,
                got: concept.embedding.len(),
            });
        }

        // Check capacity (simple eviction: reject if full)
        if self.concepts.len() >= self.max_size && !self.concepts.contains_key(&concept.id) {
            return Err(MemoryError::Storage(
                "Semantic store capacity exceeded".to_string(),
            ));
        }

        let id = concept.id.clone();
        let name = concept.name.clone().to_lowercase();

        concept.updated_at = Utc::now();

        self.name_index.insert(name, id.clone());
        self.concepts.insert(id.clone(), concept);

        Ok(id)
    }

    fn get(&self, id: &ConceptId) -> MemoryResult<Concept> {
        self.concepts
            .get(id)
            .cloned()
            .ok_or_else(|| MemoryError::ConceptNotFound(id.clone()))
    }

    fn get_and_access(&mut self, id: &ConceptId) -> MemoryResult<Concept> {
        let concept = self
            .concepts
            .get_mut(id)
            .ok_or_else(|| MemoryError::ConceptNotFound(id.clone()))?;
        concept.mark_accessed();
        Ok(concept.clone())
    }

    fn find_by_name(&self, name: &str) -> MemoryResult<Option<Concept>> {
        let normalized = name.to_lowercase();
        if let Some(id) = self.name_index.get(&normalized) {
            Ok(Some(self.get(id)?))
        } else {
            Ok(None)
        }
    }

    fn search_by_name(&self, pattern: &str) -> MemoryResult<Vec<Concept>> {
        let pattern_lower = pattern.to_lowercase();
        let matches: Vec<_> = self
            .concepts
            .values()
            .filter(|c| c.name.to_lowercase().contains(&pattern_lower))
            .cloned()
            .collect();
        Ok(matches)
    }

    fn nearest(&self, embedding: &[f32], k: usize) -> MemoryResult<Vec<(Concept, f32)>> {
        if embedding.len() != self.embedding_dim {
            return Err(MemoryError::DimensionMismatch {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }

        let mut scored: Vec<_> = self
            .concepts
            .values()
            .filter(|c| c.has_embedding())
            .map(|c| {
                let sim = Self::cosine_similarity(embedding, &c.embedding);
                (c.clone(), sim)
            })
            .collect();

        // Sort by similarity descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored.into_iter().take(k).collect())
    }

    fn related_to(
        &self,
        id: &ConceptId,
        relation_type: Option<&RelationType>,
    ) -> MemoryResult<Vec<Concept>> {
        let concept = self.get(id)?;

        let related_ids: Vec<_> = concept
            .relations
            .iter()
            .filter(|r| relation_type.map_or(true, |rt| &r.relation_type == rt))
            .map(|r| &r.target)
            .collect();

        let mut related = Vec::new();
        for related_id in related_ids {
            if let Ok(c) = self.get(related_id) {
                related.push(c);
            }
        }

        Ok(related)
    }

    fn update(&mut self, concept: Concept) -> MemoryResult<()> {
        if !self.concepts.contains_key(&concept.id) {
            return Err(MemoryError::ConceptNotFound(concept.id.clone()));
        }

        // Validate embedding dimension if provided
        if !concept.embedding.is_empty() && concept.embedding.len() != self.embedding_dim {
            return Err(MemoryError::DimensionMismatch {
                expected: self.embedding_dim,
                got: concept.embedding.len(),
            });
        }

        let name = concept.name.clone().to_lowercase();
        self.name_index.insert(name, concept.id.clone());
        self.concepts.insert(concept.id.clone(), concept);

        Ok(())
    }

    fn delete(&mut self, id: &ConceptId) -> MemoryResult<()> {
        if let Some(concept) = self.concepts.remove(id) {
            self.name_index.remove(&concept.name.to_lowercase());
            Ok(())
        } else {
            Err(MemoryError::ConceptNotFound(id.clone()))
        }
    }

    fn count(&self) -> usize {
        self.concepts.len()
    }

    fn all_ids(&self) -> Vec<ConceptId> {
        self.concepts.keys().cloned().collect()
    }

    fn clear(&mut self) {
        self.concepts.clear();
        self.name_index.clear();
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    #[test]
    fn test_concept_new() {
        let concept = Concept::new("test", vec![0.1, 0.2, 0.3]);
        assert_eq!(concept.name, "test");
        assert_eq!(concept.embedding, vec![0.1, 0.2, 0.3]);
        assert!(concept.confidence > 0.99);
    }

    #[test]
    fn test_concept_with_description() {
        let concept = Concept::new("test", vec![]).with_description("A test concept");
        assert_eq!(concept.description, "A test concept");
    }

    #[test]
    fn test_concept_with_relation() {
        let target_id = ConceptId::new("target-1");
        let concept =
            Concept::new("test", vec![]).with_relation(target_id.clone(), RelationType::IsA, 0.9);
        assert_eq!(concept.relations.len(), 1);
        assert_eq!(concept.relations[0].target, target_id);
    }

    #[test]
    fn test_concept_has_embedding() {
        let with_emb = Concept::new("test", vec![0.1, 0.2]);
        let without_emb = Concept::new("test", vec![]);

        assert!(with_emb.has_embedding());
        assert!(!without_emb.has_embedding());
    }

    #[test]
    fn test_concept_add_relation() {
        let mut concept = Concept::new("test", vec![]);
        let target = ConceptId::new("target");
        concept.add_relation(target.clone(), RelationType::SimilarTo, 0.8);

        assert_eq!(concept.relations.len(), 1);
    }

    #[test]
    fn test_concept_remove_relation() {
        let target = ConceptId::new("target");
        let mut concept =
            Concept::new("test", vec![]).with_relation(target.clone(), RelationType::IsA, 0.9);

        concept.remove_relation(&target);
        assert!(concept.relations.is_empty());
    }

    #[test]
    fn test_concept_relations_of_type() {
        let t1 = ConceptId::new("t1");
        let t2 = ConceptId::new("t2");
        let concept = Concept::new("test", vec![])
            .with_relation(t1, RelationType::IsA, 0.9)
            .with_relation(t2, RelationType::SimilarTo, 0.5);

        let is_a_rels = concept.relations_of_type(&RelationType::IsA);
        assert_eq!(is_a_rels.len(), 1);
    }

    #[test]
    fn test_inmemory_store_and_get() {
        let mut store = InMemorySemanticStore::new(3, 100);
        let concept = Concept::new("test", vec![0.1, 0.2, 0.3]);
        let id = store.store(concept).unwrap();

        let retrieved = store.get(&id).unwrap();
        assert_eq!(retrieved.name, "test");
    }

    #[test]
    fn test_inmemory_dimension_mismatch() {
        let mut store = InMemorySemanticStore::new(3, 100);
        let concept = Concept::new("test", vec![0.1, 0.2]); // Wrong dimension

        let result = store.store(concept);
        assert!(matches!(result, Err(MemoryError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_inmemory_find_by_name() {
        let mut store = InMemorySemanticStore::new(3, 100);
        store
            .store(Concept::new("Hello World", vec![0.1, 0.2, 0.3]))
            .unwrap();

        let found = store.find_by_name("hello world").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "Hello World");
    }

    #[test]
    fn test_inmemory_search_by_name() {
        let mut store = InMemorySemanticStore::new(3, 100);
        store
            .store(Concept::new("apple", vec![0.1, 0.2, 0.3]))
            .unwrap();
        store
            .store(Concept::new("pineapple", vec![0.1, 0.2, 0.3]))
            .unwrap();
        store
            .store(Concept::new("banana", vec![0.1, 0.2, 0.3]))
            .unwrap();

        let results = store.search_by_name("apple").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_inmemory_nearest() {
        let mut store = InMemorySemanticStore::new(4, 100);

        store
            .store(Concept::new("a", sample_embedding(4, 0.0)))
            .unwrap();
        store
            .store(Concept::new("b", sample_embedding(4, 1.0)))
            .unwrap();
        store
            .store(Concept::new("c", sample_embedding(4, 2.0)))
            .unwrap();

        let query = sample_embedding(4, 0.1);
        let results = store.nearest(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // First should be most similar
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn test_inmemory_nearest_dimension_mismatch() {
        let store = InMemorySemanticStore::new(4, 100);
        let result = store.nearest(&[0.1, 0.2], 2);
        assert!(matches!(result, Err(MemoryError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_inmemory_related_to() {
        let mut store = InMemorySemanticStore::new(3, 100);

        let parent = Concept::new("animal", vec![0.1, 0.2, 0.3]);
        let parent_id = parent.id.clone();
        store.store(parent).unwrap();

        let child = Concept::new("dog", vec![0.1, 0.2, 0.3]).with_relation(
            parent_id.clone(),
            RelationType::IsA,
            0.9,
        );
        let child_id = child.id.clone();
        store.store(child).unwrap();

        let related = store
            .related_to(&child_id, Some(&RelationType::IsA))
            .unwrap();
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].name, "animal");
    }

    #[test]
    fn test_inmemory_update() {
        let mut store = InMemorySemanticStore::new(3, 100);
        let concept = Concept::new("test", vec![0.1, 0.2, 0.3]);
        let id = store.store(concept).unwrap();

        let mut updated = store.get(&id).unwrap();
        updated.confidence = 0.5;
        store.update(updated).unwrap();

        let retrieved = store.get(&id).unwrap();
        assert!((retrieved.confidence - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_inmemory_delete() {
        let mut store = InMemorySemanticStore::new(3, 100);
        let concept = Concept::new("test", vec![0.1, 0.2, 0.3]);
        let id = store.store(concept).unwrap();

        assert_eq!(store.count(), 1);
        store.delete(&id).unwrap();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_inmemory_all_ids() {
        let mut store = InMemorySemanticStore::new(3, 100);
        let id1 = store.store(Concept::new("a", vec![0.1, 0.2, 0.3])).unwrap();
        let id2 = store.store(Concept::new("b", vec![0.1, 0.2, 0.3])).unwrap();

        let ids = store.all_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn test_inmemory_clear() {
        let mut store = InMemorySemanticStore::new(3, 100);
        store.store(Concept::new("a", vec![0.1, 0.2, 0.3])).unwrap();
        store.store(Concept::new("b", vec![0.1, 0.2, 0.3])).unwrap();

        assert_eq!(store.count(), 2);
        store.clear();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = InMemorySemanticStore::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        let sim2 = InMemorySemanticStore::cosine_similarity(&a, &c);
        assert!(sim2.abs() < 0.001);
    }

    #[test]
    fn test_capacity_exceeded() {
        let mut store = InMemorySemanticStore::new(3, 2);
        store.store(Concept::new("a", vec![0.1, 0.2, 0.3])).unwrap();
        store.store(Concept::new("b", vec![0.1, 0.2, 0.3])).unwrap();

        let result = store.store(Concept::new("c", vec![0.1, 0.2, 0.3]));
        assert!(matches!(result, Err(MemoryError::Storage(_))));
    }

    #[test]
    fn test_get_and_access() {
        let mut store = InMemorySemanticStore::new(3, 100);
        let concept = Concept::new("test", vec![0.1, 0.2, 0.3]);
        let id = store.store(concept).unwrap();

        let retrieved = store.get_and_access(&id).unwrap();
        assert_eq!(retrieved.access_count, 1);

        let retrieved2 = store.get_and_access(&id).unwrap();
        assert_eq!(retrieved2.access_count, 2);
    }
}
