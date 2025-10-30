//! File indexing module

pub mod walker;
pub mod metadata;

use crate::Result;

/// File indexer
pub struct Indexer {
    // TODO: implement
}

impl Indexer {
    pub fn new(_index_path: &str) -> Result<Self> {
        todo!("Implement Indexer::new")
    }

    pub fn index_directory(&self, _path: &str) -> Result<()> {
        todo!("Implement index_directory")
    }
}