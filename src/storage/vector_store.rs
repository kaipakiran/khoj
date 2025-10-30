//! Simple offline vector store for semantic search
//!
//! Uses a flat index with exact nearest neighbor search.
//! Perfect for offline operation with datasets up to ~100k vectors.
//! For larger datasets, consider adding HNSW or other ANN algorithms.

use crate::types::{Embedding, FileId};
use crate::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Simple flat vector store for offline semantic search
///
/// Uses exact nearest neighbor search with cosine similarity.
/// All data stored in memory and can be persisted to disk.
pub struct VectorStore {
    /// Map from file ID to embedding vector
    vectors: Arc<RwLock<HashMap<FileId, Vec<f32>>>>,
    /// Dimension of embeddings (e.g., 384 for all-MiniLM-L6-v2)
    dimension: usize,
}

impl VectorStore {
    /// Create a new vector store
    ///
    /// # Arguments
    /// * `dimension` - Dimension of embeddings (e.g., 384 for all-MiniLM-L6-v2)
    pub fn new(dimension: usize) -> Result<Self> {
        Ok(Self {
            vectors: Arc::new(RwLock::new(HashMap::new())),
            dimension,
        })
    }

    /// Insert or update a vector for a file
    ///
    /// # Arguments
    /// * `file_id` - File ID
    /// * `embedding` - Embedding vector (must be normalized)
    pub fn upsert(&self, file_id: FileId, embedding: &Embedding) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(crate::Error::Embedding(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len()
            )));
        }

        let mut vectors = self.vectors.write().unwrap();
        vectors.insert(file_id, embedding.clone());

        Ok(())
    }

    /// Search for similar vectors using cosine similarity
    ///
    /// # Arguments
    /// * `query_embedding` - Query embedding vector (should be normalized)
    /// * `limit` - Number of results to return
    ///
    /// # Returns
    /// List of (file_id, similarity_score) tuples, sorted by score descending
    pub fn search(&self, query_embedding: &Embedding, limit: usize) -> Result<Vec<(FileId, f32)>> {
        if query_embedding.len() != self.dimension {
            return Err(crate::Error::Embedding(format!(
                "Query embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                query_embedding.len()
            )));
        }

        let vectors = self.vectors.read().unwrap();

        // Calculate cosine similarity for all vectors
        let mut scores: Vec<(FileId, f32)> = vectors
            .iter()
            .map(|(&file_id, embedding)| {
                let similarity = cosine_similarity(query_embedding, embedding);
                (file_id, similarity)
            })
            .collect();

        // Sort by similarity descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k results
        scores.truncate(limit);

        Ok(scores)
    }

    /// Delete a vector for a file
    ///
    /// # Arguments
    /// * `file_id` - File ID to delete
    pub fn delete(&self, file_id: FileId) -> Result<()> {
        let mut vectors = self.vectors.write().unwrap();
        vectors.remove(&file_id);
        Ok(())
    }

    /// Get the number of vectors in the store
    pub fn len(&self) -> usize {
        self.vectors.read().unwrap().len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Save the vector store to disk
    ///
    /// # Arguments
    /// * `path` - Path to save the vector store
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let vectors = self.vectors.read().unwrap();

        // Convert to a serializable format
        let data = VectorStoreData {
            dimension: self.dimension,
            vectors: vectors.clone(),
        };

        let json = serde_json::to_string(&data)?;
        fs::write(path, json)?;

        Ok(())
    }

    /// Load a vector store from disk
    ///
    /// # Arguments
    /// * `path` - Path to load the vector store from
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let data: VectorStoreData = serde_json::from_str(&json)?;

        Ok(Self {
            vectors: Arc::new(RwLock::new(data.vectors)),
            dimension: data.dimension,
        })
    }
}

/// Serializable vector store data
#[derive(serde::Serialize, serde::Deserialize)]
struct VectorStoreData {
    dimension: usize,
    vectors: HashMap<FileId, Vec<f32>>,
}

/// Compute cosine similarity between two vectors
///
/// Assumes vectors are normalized (L2 norm = 1).
/// Returns value in range [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

    // For normalized vectors, cosine similarity is just the dot product
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_vector_store() {
        let store = VectorStore::new(384).unwrap();
        assert_eq!(store.dimension, 384);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_insert_and_search() {
        let store = VectorStore::new(128).unwrap();

        // Create some test embeddings (normalized)
        let embedding1: Vec<f32> = normalize(&(0..128).map(|i| i as f32).collect::<Vec<_>>());
        let embedding2: Vec<f32> = normalize(&(0..128).map(|i| (i + 10) as f32).collect::<Vec<_>>());
        let embedding3: Vec<f32> = normalize(&(0..128).map(|i| (i + 50) as f32).collect::<Vec<_>>());

        // Insert embeddings
        store.upsert(1, &embedding1).unwrap();
        store.upsert(2, &embedding2).unwrap();
        store.upsert(3, &embedding3).unwrap();

        assert_eq!(store.len(), 3);

        // Search with a query similar to embedding1
        let query: Vec<f32> = normalize(&(0..128).map(|i| i as f32).collect::<Vec<_>>());
        let results = store.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // First result should be file_id=1 with highest similarity
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 > 0.99); // Should be very similar
    }

    #[test]
    fn test_dimension_mismatch() {
        let store = VectorStore::new(128).unwrap();

        let wrong_dim: Vec<f32> = vec![0.1, 0.2, 0.3]; // Only 3 dimensions
        let result = store.upsert(1, &wrong_dim);

        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let store = VectorStore::new(128).unwrap();

        let embedding: Vec<f32> = normalize(&(0..128).map(|i| i as f32).collect::<Vec<_>>());
        store.upsert(1, &embedding).unwrap();

        assert_eq!(store.len(), 1);

        store.delete(1).unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![0.6, 0.8];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);

        // Opposite vectors
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let save_path = temp_dir.path().join("vectors.json");

        // Create and populate store
        let store = VectorStore::new(64).unwrap();
        let embedding1: Vec<f32> = normalize(&(0..64).map(|i| i as f32).collect::<Vec<_>>());
        let embedding2: Vec<f32> = normalize(&(0..64).map(|i| (i + 10) as f32).collect::<Vec<_>>());

        store.upsert(1, &embedding1).unwrap();
        store.upsert(2, &embedding2).unwrap();

        // Save to disk
        store.save(&save_path).unwrap();

        // Load from disk
        let loaded_store = VectorStore::load(&save_path).unwrap();

        assert_eq!(loaded_store.dimension, 64);
        assert_eq!(loaded_store.len(), 2);

        // Verify search works on loaded store
        let query: Vec<f32> = normalize(&(0..64).map(|i| i as f32).collect::<Vec<_>>());
        let results = loaded_store.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_multiple_searches() {
        let store = VectorStore::new(128).unwrap();

        // Insert multiple embeddings
        for i in 0..10 {
            let embedding: Vec<f32> = normalize(&(0..128).map(|j| ((j + i * 10) as f32)).collect::<Vec<_>>());
            store.upsert(i as i64, &embedding).unwrap();
        }

        assert_eq!(store.len(), 10);

        // Search multiple times
        for _ in 0..5 {
            let query: Vec<f32> = normalize(&(0..128).map(|i| i as f32).collect::<Vec<_>>());
            let results = store.search(&query, 3).unwrap();
            assert_eq!(results.len(), 3);
        }
    }

    // Helper function to normalize a vector
    fn normalize(vec: &[f32]) -> Vec<f32> {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vec.iter().map(|x| x / norm).collect()
        } else {
            vec.to_vec()
        }
    }
}