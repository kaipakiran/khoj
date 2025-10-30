//! Hybrid search combining keyword (BM25) and semantic (vector) search

use crate::storage::{TantivyIndex, VectorStore};
use crate::types::{Embedding, FileId, SearchResult};
use crate::Result;
use std::collections::HashMap;

/// Hybrid search engine combining BM25 and vector search
pub struct HybridSearch {
    tantivy_index: TantivyIndex,
    vector_store: VectorStore,
}

impl HybridSearch {
    /// Create a new hybrid search engine
    pub fn new(tantivy_index: TantivyIndex, vector_store: VectorStore) -> Self {
        Self {
            tantivy_index,
            vector_store,
        }
    }

    /// Search using keyword search only (BM25)
    pub fn keyword_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.tantivy_index.search(query, limit)
    }

    /// Search using semantic search only (vector similarity)
    pub fn semantic_search(&self, query_embedding: &Embedding, limit: usize) -> Result<Vec<(FileId, f32)>> {
        self.vector_store.search(query_embedding, limit)
    }

    /// Hybrid search combining keyword and semantic search using Reciprocal Rank Fusion
    ///
    /// # Arguments
    /// * `query` - Text query for keyword search
    /// * `query_embedding` - Optional embedding for semantic search
    /// * `limit` - Number of results to return
    /// * `keyword_weight` - Weight for keyword search results (0.0 to 1.0, default 0.7)
    ///
    /// # Returns
    /// Combined and re-ranked search results
    pub fn hybrid_search(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
        keyword_weight: f32,
    ) -> Result<Vec<SearchResult>> {
        // Get keyword search results
        let keyword_results = self.tantivy_index.search(query, limit * 2)?;

        // Get semantic search results if embedding provided
        let semantic_results = if let Some(embedding) = query_embedding {
            let embedding_vec = embedding.to_vec();
            self.vector_store.search(&embedding_vec, limit * 2)?
        } else {
            Vec::new()
        };

        // If no semantic results, return keyword results only
        if semantic_results.is_empty() {
            let mut results = keyword_results;
            results.truncate(limit);
            return Ok(results);
        }

        // Use Reciprocal Rank Fusion to combine results
        let combined = reciprocal_rank_fusion(
            &keyword_results,
            &semantic_results,
            keyword_weight,
            limit,
        )?;

        // Fetch file metadata for combined results
        let mut final_results = Vec::new();
        for (file_id, score) in combined {
            // Try to find existing result from keyword search
            if let Some(result) = keyword_results.iter().find(|r| r.file_id == file_id) {
                final_results.push(SearchResult {
                    file_id,
                    path: result.path.clone(),
                    filename: result.filename.clone(),
                    score,
                    snippet: result.snippet.clone(),
                });
            } else {
                // If not in keyword results, create result without snippet
                final_results.push(SearchResult {
                    file_id,
                    path: format!("file_{}", file_id), // Placeholder - would fetch from DB in production
                    filename: format!("file_{}", file_id),
                    score,
                    snippet: None,
                });
            }
        }

        Ok(final_results)
    }
}

/// Reciprocal Rank Fusion (RRF) algorithm
///
/// Combines rankings from multiple sources using the formula:
/// RRF_score(d) = Σ 1 / (k + rank(d))
///
/// where k is a constant (typically 60) and rank(d) is the rank of document d in each list
fn reciprocal_rank_fusion(
    keyword_results: &[SearchResult],
    semantic_results: &[(FileId, f32)],
    keyword_weight: f32,
    limit: usize,
) -> Result<Vec<(FileId, f32)>> {
    const K: f32 = 60.0; // RRF constant

    let semantic_weight = 1.0 - keyword_weight;
    let mut scores: HashMap<FileId, f32> = HashMap::new();

    // Add keyword search scores
    for (rank, result) in keyword_results.iter().enumerate() {
        let rrf_score = keyword_weight / (K + (rank as f32) + 1.0);
        *scores.entry(result.file_id).or_insert(0.0) += rrf_score;
    }

    // Add semantic search scores
    for (rank, &(file_id, _similarity)) in semantic_results.iter().enumerate() {
        let rrf_score = semantic_weight / (K + (rank as f32) + 1.0);
        *scores.entry(file_id).or_insert(0.0) += rrf_score;
    }

    // Sort by combined score
    let mut combined: Vec<(FileId, f32)> = scores.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined.truncate(limit);

    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reciprocal_rank_fusion() {
        // Create mock keyword results
        let keyword_results = vec![
            SearchResult {
                file_id: 1,
                path: "file1.txt".to_string(),
                filename: "file1.txt".to_string(),
                score: 10.0,
                snippet: None,
            },
            SearchResult {
                file_id: 2,
                path: "file2.txt".to_string(),
                filename: "file2.txt".to_string(),
                score: 8.0,
                snippet: None,
            },
            SearchResult {
                file_id: 3,
                path: "file3.txt".to_string(),
                filename: "file3.txt".to_string(),
                score: 6.0,
                snippet: None,
            },
        ];

        // Create mock semantic results
        let semantic_results = vec![
            (2, 0.95), // file2 ranks high in semantic search
            (4, 0.90), // file4 only in semantic search
            (1, 0.85), // file1 also appears
        ];

        // Test with equal weights
        let combined = reciprocal_rank_fusion(&keyword_results, &semantic_results, 0.5, 10).unwrap();

        // File 2 should rank highest (appears in both)
        assert_eq!(combined[0].0, 2);

        // Should combine unique results
        assert!(combined.len() >= 4);
    }

    #[test]
    fn test_rrf_keyword_only() {
        let keyword_results = vec![
            SearchResult {
                file_id: 1,
                path: "file1.txt".to_string(),
                filename: "file1.txt".to_string(),
                score: 10.0,
                snippet: None,
            },
        ];

        let semantic_results = vec![];

        let combined = reciprocal_rank_fusion(&keyword_results, &semantic_results, 1.0, 10).unwrap();

        assert_eq!(combined.len(), 1);
        assert_eq!(combined[0].0, 1);
    }

    #[test]
    fn test_rrf_semantic_only() {
        let keyword_results = vec![];
        let semantic_results = vec![(1, 0.95), (2, 0.90)];

        let combined = reciprocal_rank_fusion(&keyword_results, &semantic_results, 0.0, 10).unwrap();

        assert_eq!(combined.len(), 2);
        assert_eq!(combined[0].0, 1); // Higher similarity ranks first
    }
}