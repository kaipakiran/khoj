//! Tantivy full-text search index

use crate::types::{FileId, SearchResult};
use crate::Result;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy};

/// Tantivy search index for BM25 keyword search
pub struct TantivyIndex {
    index: Index,
    reader: IndexReader,
    writer: IndexWriter,
    file_id_field: Field,
    path_field: Field,
    filename_field: Field,
    content_field: Field,
}

impl TantivyIndex {
    /// Create a new Tantivy index
    ///
    /// # Arguments
    /// * `index_path` - Directory to store the index
    pub fn new<P: AsRef<Path>>(index_path: P) -> Result<Self> {
        let index_path = index_path.as_ref();

        // Create schema
        let mut schema_builder = Schema::builder();
        let file_id_field = schema_builder.add_i64_field("file_id", STORED | FAST | INDEXED);
        let path_field = schema_builder.add_text_field("path", STRING | STORED);
        let filename_field = schema_builder.add_text_field("filename", TEXT | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT);
        let schema = schema_builder.build();

        // Create or open index
        let index = if index_path.exists() {
            Index::open_in_dir(index_path)?
        } else {
            std::fs::create_dir_all(index_path)?;
            Index::create_in_dir(index_path, schema.clone())?
        };

        // Create writer with 50MB buffer
        let writer = index.writer(50_000_000)?;

        // Create reader with auto-reload
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            writer,
            file_id_field,
            path_field,
            filename_field,
            content_field,
        })
    }

    /// Add or update a document in the index
    ///
    /// # Arguments
    /// * `file_id` - File ID
    /// * `path` - File path
    /// * `filename` - Filename
    /// * `content` - File content
    pub fn upsert_document(
        &mut self,
        file_id: FileId,
        path: &str,
        filename: &str,
        content: &str,
    ) -> Result<()> {
        // Delete existing document with this file_id
        let term = Term::from_field_i64(self.file_id_field, file_id);
        self.writer.delete_term(term);

        // Add new document
        let doc = doc!(
            self.file_id_field => file_id,
            self.path_field => path,
            self.filename_field => filename,
            self.content_field => content,
        );

        self.writer.add_document(doc)?;
        Ok(())
    }

    /// Commit changes to the index
    pub fn commit(&mut self) -> Result<()> {
        self.writer.commit()?;
        // Reload reader to see new documents
        self.reader.reload()?;
        Ok(())
    }

    /// Search the index with BM25 ranking
    ///
    /// # Arguments
    /// * `query` - Search query string
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    /// List of search results with scores
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let searcher = self.reader.searcher();

        // Parse query (searches in filename and content fields)
        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.filename_field, self.content_field],
        );

        let query = query_parser.parse_query(query)?;

        // Execute search
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        // Convert results
        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let doc = searcher.doc::<tantivy::TantivyDocument>(doc_address)?;

            let file_id = doc
                .get_first(self.file_id_field)
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            let path = doc
                .get_first(self.path_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let filename = doc
                .get_first(self.filename_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            results.push(SearchResult {
                file_id,
                path,
                filename,
                score,
                snippet: None, // Will be added by search engine
            });
        }

        Ok(results)
    }

    /// Delete a document from the index
    ///
    /// # Arguments
    /// * `file_id` - File ID to delete
    pub fn delete_document(&mut self, file_id: FileId) -> Result<()> {
        let term = Term::from_field_i64(self.file_id_field, file_id);
        self.writer.delete_term(term);
        self.writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Get the number of documents in the index
    pub fn num_docs(&self) -> u64 {
        let searcher = self.reader.searcher();
        searcher.num_docs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_index() -> (TantivyIndex, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("tantivy");
        let index = TantivyIndex::new(&index_path).unwrap();
        (index, temp_dir)
    }

    #[test]
    fn test_create_index() {
        let (index, _temp_dir) = create_test_index();
        assert_eq!(index.num_docs(), 0);
    }

    #[test]
    fn test_upsert_document() {
        let (mut index, _temp_dir) = create_test_index();

        index
            .upsert_document(1, "/test/file.txt", "file.txt", "Hello world")
            .unwrap();

        index.commit().unwrap();
        assert_eq!(index.num_docs(), 1);
    }

    #[test]
    fn test_search() {
        let (mut index, _temp_dir) = create_test_index();

        // Add test documents
        index
            .upsert_document(
                1,
                "/test/rust.rs",
                "rust.rs",
                "Rust is a systems programming language",
            )
            .unwrap();

        index
            .upsert_document(
                2,
                "/test/python.py",
                "python.py",
                "Python is a high-level programming language",
            )
            .unwrap();

        index
            .upsert_document(3, "/test/hello.txt", "hello.txt", "Hello world")
            .unwrap();

        index.commit().unwrap();

        // Search for "programming"
        let results = index.search("programming", 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.filename == "rust.rs"));
        assert!(results.iter().any(|r| r.filename == "python.py"));

        // Search for "Rust"
        let results = index.search("Rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].filename, "rust.rs");

        // Search for "hello"
        let results = index.search("hello", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].filename, "hello.txt");
    }

    #[test]
    fn test_upsert_updates_existing() {
        let (mut index, _temp_dir) = create_test_index();

        // Add document
        index
            .upsert_document(1, "/test/file.txt", "file.txt", "apple orange")
            .unwrap();
        index.commit().unwrap();

        // Update same document - use completely different words
        index
            .upsert_document(1, "/test/file.txt", "file.txt", "banana grape")
            .unwrap();
        index.commit().unwrap();

        // Search should find updated content
        let results = index.search("banana", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_id, 1);
    }

    #[test]
    fn test_delete_document() {
        let (mut index, _temp_dir) = create_test_index();

        // Add documents with completely distinct words
        index
            .upsert_document(1, "/test/file1.txt", "file1.txt", "apple orange pear")
            .unwrap();
        index
            .upsert_document(2, "/test/file2.txt", "file2.txt", "banana grape melon")
            .unwrap();
        index.commit().unwrap();

        // Delete one document
        index.delete_document(1).unwrap();

        // Verify it's gone from search results
        let results = index.search("apple", 10).unwrap();
        assert_eq!(results.len(), 0, "Found {} documents with 'apple', expected 0. Results: {:?}", results.len(), results);

        // Other document still there
        let results = index.search("banana", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_filename() {
        let (mut index, _temp_dir) = create_test_index();

        index
            .upsert_document(1, "/test/important.txt", "important.txt", "some content")
            .unwrap();
        index.commit().unwrap();

        // Search by filename
        let results = index.search("important", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].filename, "important.txt");
    }

    #[test]
    fn test_search_limit() {
        let (mut index, _temp_dir) = create_test_index();

        // Add many documents
        for i in 0..10 {
            index
                .upsert_document(
                    i,
                    &format!("/test/file{}.txt", i),
                    &format!("file{}.txt", i),
                    "test content",
                )
                .unwrap();
        }
        index.commit().unwrap();

        // Search with limit
        let results = index.search("test", 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_empty_search() {
        let (index, _temp_dir) = create_test_index();

        // Search empty index
        let results = index.search("anything", 10).unwrap();
        assert_eq!(results.len(), 0);
    }
}