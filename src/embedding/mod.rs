//! Embedding generation using ONNX models for text and images

pub mod tokenizer;
pub mod image;

use crate::types::Embedding;
use crate::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::path::Path;
use tokenizer::Tokenizer;

/// Text embedding model using ONNX Runtime
pub struct EmbeddingModel {
    session: Session,
    tokenizer: Tokenizer,
    max_length: usize,
}

impl EmbeddingModel {
    /// Create a new embedding model
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `vocab_path` - Path to the vocabulary file
    ///
    /// # Returns
    /// An embedding model ready to generate embeddings
    pub fn new<P: AsRef<Path>>(model_path: P, tokenizer_path: P) -> Result<Self> {
        // Load ONNX model file
        let model_bytes = std::fs::read(model_path.as_ref())?;

        // Load ONNX model with optimizations
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_memory(&model_bytes)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        Ok(Self {
            session,
            tokenizer,
            max_length: 512, // all-MiniLM-L6-v2 max sequence length
        })
    }

    /// Create a new embedding model from Hugging Face model identifier
    ///
    /// # Arguments
    /// * `model_identifier` - Model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    pub fn from_pretrained(model_identifier: &str) -> Result<Self> {
        // For now, this is a placeholder. In a real implementation, you would download
        // the ONNX model from Hugging Face Hub
        // Load tokenizer from Hugging Face
        let _tokenizer = Tokenizer::from_pretrained(model_identifier)?;

        // This would need to download the actual ONNX model file
        // For now, we return an error indicating this needs the model file
        Err(crate::Error::Embedding(
            "from_pretrained requires manual model download. Use new() with model_path instead".to_string()
        ))
    }

    /// Generate embedding for a text string
    ///
    /// # Arguments
    /// * `text` - Input text to embed
    ///
    /// # Returns
    /// 384-dimensional embedding vector for all-MiniLM-L6-v2
    pub fn embed(&mut self, text: &str) -> Result<Embedding> {
        // Tokenize input text
        let tokens = self.tokenizer.encode(text, self.max_length)?;

        // Prepare input tensors as ort Values
        // ort 2.0 expects (shape, data) tuples
        let seq_len = tokens.input_ids.len();
        let shape = vec![1, seq_len];

        let input_ids_value = Value::from_array((shape.clone(), tokens.input_ids.clone()))?;
        let attention_mask_value = Value::from_array((shape.clone(), tokens.attention_mask.clone()))?;
        let token_type_ids_value = Value::from_array((shape.clone(), tokens.token_type_ids.clone()))?;

        // Run inference with proper input format
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_value,
            "attention_mask" => attention_mask_value,
            "token_type_ids" => token_type_ids_value,
        ])?;

        // Extract embeddings from output
        // all-MiniLM-L6-v2 outputs shape: [batch_size, sequence_length, hidden_size]
        // We need to mean pool over the sequence length dimension
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        // Convert to ndarray for easier manipulation
        let embeddings = ndarray::ArrayView3::from_shape(
            (shape[0] as usize, shape[1] as usize, shape[2] as usize),
            data
        ).unwrap().to_owned();

        // Drop outputs to release the mutable borrow
        drop(outputs);

        // Mean pooling: average over sequence length (dim 1)
        let pooled = self.mean_pool(&embeddings, &tokens.attention_mask)?;

        // Normalize the embedding
        let normalized = self.normalize(&pooled);

        Ok(normalized)
    }

    /// Mean pooling over sequence dimension with attention mask
    fn mean_pool(&self, embeddings: &ndarray::Array3<f32>, attention_mask: &[i64]) -> Result<Vec<f32>> {
        let batch_size = embeddings.shape()[0];
        let seq_len = embeddings.shape()[1];
        let hidden_size = embeddings.shape()[2];

        assert_eq!(batch_size, 1, "Only batch size 1 is supported");

        let mut pooled = vec![0.0f32; hidden_size];
        let mut mask_sum = 0i64;

        for i in 0..seq_len {
            let mask_value = attention_mask[i];
            mask_sum += mask_value;

            if mask_value > 0 {
                for j in 0..hidden_size {
                    pooled[j] += embeddings[[0, i, j]] * mask_value as f32;
                }
            }
        }

        // Average
        for val in &mut pooled {
            *val /= mask_sum as f32;
        }

        Ok(pooled)
    }

    /// L2 normalize the embedding vector
    fn normalize(&self, embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding.to_vec()
        }
    }

    /// Compute cosine similarity between two embeddings
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Embeddings must have same dimension");

        // Since embeddings are already normalized, cosine similarity is just the dot product
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let vec = vec![3.0, 4.0];
        let normalized = normalize_test_helper(&vec);

        // 3-4-5 triangle: normalized should be [0.6, 0.8]
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let vec = vec![0.0, 0.0];
        let normalized = normalize_test_helper(&vec);

        assert_eq!(normalized, vec![0.0, 0.0]);
    }

    #[test]
    fn test_cosine_similarity() {
        // Same vector
        let a = vec![0.6, 0.8];
        let sim = EmbeddingModel::cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = EmbeddingModel::cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    // Helper function for testing normalize without needing a full model
    fn normalize_test_helper(embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding.to_vec()
        }
    }
}