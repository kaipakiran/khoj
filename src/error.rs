//! Error types for the file-search library

use thiserror::Error;

/// Result type alias for file-search operations
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for file-search
#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Search index error: {0}")]
    SearchIndex(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("File extraction error: {0}")]
    Extraction(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Unsupported file type: {0}")]
    UnsupportedFileType(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// Convert tantivy errors to our Error type
impl From<tantivy::TantivyError> for Error {
    fn from(err: tantivy::TantivyError) -> Self {
        Error::SearchIndex(err.to_string())
    }
}

// Convert ort errors to our Error type
impl From<ort::Error> for Error {
    fn from(err: ort::Error) -> Self {
        Error::Embedding(err.to_string())
    }
}

// Convert ignore errors to our Error type
impl From<ignore::Error> for Error {
    fn from(err: ignore::Error) -> Self {
        Error::Io(std::io::Error::new(std::io::ErrorKind::Other, err))
    }
}

// Convert tantivy query parser errors to our Error type
impl From<tantivy::query::QueryParserError> for Error {
    fn from(err: tantivy::query::QueryParserError) -> Self {
        Error::SearchIndex(err.to_string())
    }
}

// Convert serde_json errors to our Error type
impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Other(err.into())
    }
}