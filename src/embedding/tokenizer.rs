//! BERT-style tokenizer using Hugging Face tokenizers library

use crate::Result;
use std::path::Path;
use tokenizers::Tokenizer as HFTokenizer;

/// Tokenized output with input tensors
#[derive(Debug, Clone)]
pub struct TokenizedInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
}

/// BERT-style tokenizer wrapper around Hugging Face tokenizers
#[derive(Clone)]
pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    /// Load tokenizer from file
    ///
    /// # Arguments
    /// * `tokenizer_path` - Path to tokenizer.json file
    pub fn from_file<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(tokenizer_path)
            .map_err(|e| crate::Error::Embedding(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self { tokenizer })
    }

    /// Load tokenizer from pretrained model (requires downloading tokenizer file)
    ///
    /// # Arguments
    /// * `identifier` - Model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    ///
    /// Note: You need to download the tokenizer.json file from Hugging Face Hub manually
    /// and use `from_file` instead. This is a placeholder for future implementation.
    pub fn from_pretrained(_identifier: &str) -> Result<Self> {
        Err(crate::Error::Embedding(
            "from_pretrained not supported. Download tokenizer.json and use from_file() instead".to_string()
        ))
    }

    /// Encode text to token IDs
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `max_length` - Maximum sequence length
    pub fn encode(&self, text: &str, max_length: usize) -> Result<TokenizedInput> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| crate::Error::Embedding(format!("Tokenization failed: {}", e)))?;

        // Get token IDs and convert to i64
        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        // Get attention mask (1 for real tokens, 0 for padding)
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        // Get token type IDs (defaults to 0 for single sequence)
        let mut token_type_ids: Vec<i64> = encoding
            .get_type_ids()
            .iter()
            .map(|&t| t as i64)
            .collect();

        // Truncate to max_length if needed
        if input_ids.len() > max_length {
            input_ids.truncate(max_length);
            attention_mask.truncate(max_length);
            token_type_ids.truncate(max_length);
        }

        // Pad to max_length
        let padding_length = max_length.saturating_sub(input_ids.len());
        if padding_length > 0 {
            let pad_token_id = self
                .tokenizer
                .get_padding()
                .map(|p| p.pad_id as i64)
                .unwrap_or(0);

            input_ids.extend(vec![pad_token_id; padding_length]);
            attention_mask.extend(vec![0i64; padding_length]);
            token_type_ids.extend(vec![0i64; padding_length]);
        }

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenized_input_structure() {
        // Just test the structure works
        let input = TokenizedInput {
            input_ids: vec![101, 102, 103],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };

        assert_eq!(input.input_ids.len(), 3);
        assert_eq!(input.attention_mask.len(), 3);
        assert_eq!(input.token_type_ids.len(), 3);
    }

    #[test]
    fn test_load_tokenizer_from_file() {
        let tokenizer_path = "models/tokenizer.json";
        if std::path::Path::new(tokenizer_path).exists() {
            let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

            // Test basic encoding
            let encoded = tokenizer.encode("hello world", 128).unwrap();

            assert_eq!(encoded.input_ids.len(), 128);
            assert_eq!(encoded.attention_mask.len(), 128);
            assert_eq!(encoded.token_type_ids.len(), 128);

            // First token should be [CLS] (101 for BERT-based models)
            assert_eq!(encoded.input_ids[0], 101);
            assert_eq!(encoded.attention_mask[0], 1);
        }
    }

    #[test]
    fn test_encode_and_pad() {
        let tokenizer_path = "models/tokenizer.json";
        if std::path::Path::new(tokenizer_path).exists() {
            let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

            let encoded = tokenizer.encode("test", 64).unwrap();

            // Should be padded to 64
            assert_eq!(encoded.input_ids.len(), 64);

            // Padding tokens should have attention_mask = 0
            // Find first padding position (where attention_mask = 0)
            let pad_start = encoded
                .attention_mask
                .iter()
                .position(|&m| m == 0)
                .unwrap_or(64);

            if pad_start < 64 {
                assert_eq!(encoded.input_ids[pad_start], 0); // PAD token
                assert_eq!(encoded.attention_mask[pad_start], 0);
            }
        }
    }

    #[test]
    fn test_encode_truncation() {
        let tokenizer_path = "models/tokenizer.json";
        if std::path::Path::new(tokenizer_path).exists() {
            let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

            let long_text = "word ".repeat(200);
            let encoded = tokenizer.encode(&long_text, 32).unwrap();

            // Should be truncated to 32
            assert_eq!(encoded.input_ids.len(), 32);
            assert_eq!(encoded.attention_mask.len(), 32);
        }
    }
}