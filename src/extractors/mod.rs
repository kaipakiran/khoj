//! Content extractors for different file types

pub mod text;

pub use text::{extract_text, extract_snippet, ExtractedContent};