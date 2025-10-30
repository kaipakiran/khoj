//! File Search - A fast, offline, hybrid search engine for files
//!
//! This library provides a hybrid search engine that combines traditional
//! keyword-based search (BM25) with semantic search using embeddings.
//!
//! # Example
//!
//! ```no_run
//! use file_search::{
//!     storage::{TantivyIndex, VectorStore},
//!     search::HybridSearch,
//! };
//!
//! // Initialize components
//! let tantivy_index = TantivyIndex::new("~/.file-search/tantivy")?;
//! let vector_store = VectorStore::new(384)?;
//!
//! // Create search engine
//! let engine = HybridSearch::new(tantivy_index, vector_store);
//! let results = engine.keyword_search("rust async", 10)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod cli;
pub mod config;
pub mod embedding;
pub mod extractors;
pub mod indexer;
pub mod search;
pub mod storage;
pub mod watcher;
pub mod web;

pub mod error;
pub use error::{Error, Result};

/// Common types used throughout the library
pub mod types {
    use serde::{Deserialize, Serialize};

    /// Unique identifier for a file in the index
    pub type FileId = i64;

    /// Vector embedding (typically 384 dimensions for text)
    pub type Embedding = Vec<f32>;

    /// File type classification
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum FileType {
        Text,
        Code,
        Markdown,
        Pdf,
        Docx,
        Xlsx,
        Image,
        Archive,
        Unknown,
    }

    impl FileType {
        /// Get file type from extension
        pub fn from_extension(ext: &str) -> Self {
            match ext.to_lowercase().as_str() {
                "txt" | "text" => FileType::Text,
                "md" | "markdown" => FileType::Markdown,
                "rs" | "py" | "js" | "ts" | "java" | "c" | "cpp" | "go" | "rb" => FileType::Code,
                "pdf" => FileType::Pdf,
                "docx" | "doc" => FileType::Docx,
                "xlsx" | "xls" => FileType::Xlsx,
                "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" => FileType::Image,
                "zip" | "tar" | "gz" | "7z" => FileType::Archive,
                _ => FileType::Unknown,
            }
        }

        /// Convert to string representation
        pub fn as_str(&self) -> &'static str {
            match self {
                FileType::Text => "text",
                FileType::Code => "code",
                FileType::Markdown => "markdown",
                FileType::Pdf => "pdf",
                FileType::Docx => "docx",
                FileType::Xlsx => "xlsx",
                FileType::Image => "image",
                FileType::Archive => "archive",
                FileType::Unknown => "unknown",
            }
        }
    }

    /// Metadata about an indexed file
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FileMetadata {
        pub id: FileId,
        pub path: String,
        pub filename: String,
        pub file_type: FileType,
        pub mime_type: Option<String>,
        pub size: u64,
        pub hash: String,
        pub created_at: i64,
        pub modified_at: i64,
        pub indexed_at: i64,
    }

    /// Search result with score
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SearchResult {
        pub file_id: FileId,
        pub path: String,
        pub filename: String,
        pub score: f32,
        pub snippet: Option<String>,
    }
}