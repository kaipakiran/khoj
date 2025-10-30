//! Storage layer for metadata and content

pub mod tantivy_index;
pub mod vector_store;

use crate::extractors::ExtractedContent;
use crate::types::{FileId, FileMetadata, FileType};
use crate::Result;
use sqlx::{sqlite::SqlitePool, Row};
use std::path::Path;

pub use tantivy_index::TantivyIndex;
pub use vector_store::VectorStore;

/// Database connection pool
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    /// Create a new database connection
    ///
    /// # Arguments
    /// * `db_path` - Path to the SQLite database file
    pub async fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let db_url = format!("sqlite://{}?mode=rwc", db_path.as_ref().display());
        let pool = SqlitePool::connect(&db_url).await?;

        let db = Self { pool };
        db.initialize_schema().await?;

        Ok(db)
    }

    /// Initialize database schema
    async fn initialize_schema(&self) -> Result<()> {
        // Enable WAL mode for concurrent reads/writes
        sqlx::query("PRAGMA journal_mode=WAL").execute(&self.pool).await?;

        let schema = include_str!("schema.sql");
        sqlx::query(schema).execute(&self.pool).await?;
        Ok(())
    }

    /// Insert or update a file's metadata
    ///
    /// # Arguments
    /// * `metadata` - File metadata to store
    ///
    /// # Returns
    /// File ID (new or existing)
    pub async fn upsert_file(&self, metadata: &FileMetadata) -> Result<FileId> {
        let file_type_str = metadata.file_type.as_str();

        let result = sqlx::query(
            r#"
            INSERT INTO files (path, filename, file_type, mime_type, size, hash, created_at, modified_at, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                filename = excluded.filename,
                file_type = excluded.file_type,
                mime_type = excluded.mime_type,
                size = excluded.size,
                hash = excluded.hash,
                modified_at = excluded.modified_at,
                indexed_at = excluded.indexed_at
            RETURNING id
            "#,
        )
        .bind(&metadata.path)
        .bind(&metadata.filename)
        .bind(file_type_str)
        .bind(&metadata.mime_type)
        .bind(metadata.size as i64)
        .bind(&metadata.hash)
        .bind(metadata.created_at)
        .bind(metadata.modified_at)
        .bind(metadata.indexed_at)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.get("id"))
    }

    /// Insert or update file content
    ///
    /// # Arguments
    /// * `file_id` - File ID
    /// * `content` - Extracted content
    pub async fn upsert_content(&self, file_id: FileId, content: &ExtractedContent) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO content (file_id, text, word_count, language)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file_id) DO UPDATE SET
                text = excluded.text,
                word_count = excluded.word_count,
                language = excluded.language
            "#,
        )
        .bind(file_id)
        .bind(&content.text)
        .bind(content.word_count as i64)
        .bind(&content.language)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get file metadata by file ID
    pub async fn get_file(&self, file_id: FileId) -> Result<Option<FileMetadata>> {
        let result = sqlx::query_as::<_, FileMetadataRow>(
            "SELECT id, path, filename, file_type, mime_type, size, hash, created_at, modified_at, indexed_at
             FROM files WHERE id = ?"
        )
        .bind(file_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(result.map(|row| row.into()))
    }

    /// Get file metadata by path
    pub async fn get_file_by_path(&self, path: &str) -> Result<Option<FileMetadata>> {
        let result = sqlx::query_as::<_, FileMetadataRow>(
            "SELECT id, path, filename, file_type, mime_type, size, hash, created_at, modified_at, indexed_at
             FROM files WHERE path = ?"
        )
        .bind(path)
        .fetch_optional(&self.pool)
        .await?;

        Ok(result.map(|row| row.into()))
    }

    /// Get file content by file ID
    pub async fn get_content(&self, file_id: FileId) -> Result<Option<ExtractedContent>> {
        let result = sqlx::query(
            "SELECT text, word_count, language FROM content WHERE file_id = ?"
        )
        .bind(file_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(result.map(|row| ExtractedContent {
            text: row.get("text"),
            word_count: row.get::<i64, _>("word_count") as usize,
            language: row.get("language"),
        }))
    }

    /// Check if a file needs reindexing (hash changed)
    pub async fn needs_reindex(&self, path: &str, current_hash: &str) -> Result<bool> {
        let result = sqlx::query("SELECT hash FROM files WHERE path = ?")
            .bind(path)
            .fetch_optional(&self.pool)
            .await?;

        match result {
            Some(row) => {
                let stored_hash: String = row.get("hash");
                Ok(stored_hash != current_hash)
            }
            None => Ok(true), // File not in index, needs indexing
        }
    }

    /// Get total number of indexed files
    pub async fn count_files(&self) -> Result<i64> {
        let result = sqlx::query("SELECT COUNT(*) as count FROM files")
            .fetch_one(&self.pool)
            .await?;

        Ok(result.get("count"))
    }

    /// Get statistics about indexed files
    pub async fn get_stats(&self) -> Result<IndexStats> {
        let total_files: i64 = sqlx::query("SELECT COUNT(*) as count FROM files")
            .fetch_one(&self.pool)
            .await?
            .get("count");

        let total_size: i64 = sqlx::query("SELECT COALESCE(SUM(size), 0) as total FROM files")
            .fetch_one(&self.pool)
            .await?
            .get("total");

        let by_type = sqlx::query("SELECT file_type, COUNT(*) as count FROM files GROUP BY file_type")
            .fetch_all(&self.pool)
            .await?
            .into_iter()
            .map(|row| (row.get::<String, _>("file_type"), row.get::<i64, _>("count")))
            .collect();

        Ok(IndexStats {
            total_files,
            total_size,
            by_type,
        })
    }

    /// Delete a file from the index
    pub async fn delete_file(&self, path: &str) -> Result<()> {
        sqlx::query("DELETE FROM files WHERE path = ?")
            .bind(path)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

/// Statistics about the index
#[derive(Debug)]
pub struct IndexStats {
    pub total_files: i64,
    pub total_size: i64,
    pub by_type: Vec<(String, i64)>,
}

/// Helper struct for deserializing file metadata from database
#[derive(sqlx::FromRow)]
struct FileMetadataRow {
    id: i64,
    path: String,
    filename: String,
    file_type: String,
    mime_type: Option<String>,
    size: i64,
    hash: String,
    created_at: i64,
    modified_at: i64,
    indexed_at: i64,
}

impl From<FileMetadataRow> for FileMetadata {
    fn from(row: FileMetadataRow) -> Self {
        FileMetadata {
            id: row.id,
            path: row.path,
            filename: row.filename,
            file_type: FileType::from_extension(&row.file_type),
            mime_type: row.mime_type,
            size: row.size as u64,
            hash: row.hash,
            created_at: row.created_at,
            modified_at: row.modified_at,
            indexed_at: row.indexed_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FileType;
    use tempfile::TempDir;

    async fn create_test_db() -> (Database, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let db = Database::new(&db_path).await.unwrap();
        (db, temp_dir)
    }

    fn create_test_metadata() -> FileMetadata {
        FileMetadata {
            id: 0,
            path: "/test/file.txt".to_string(),
            filename: "file.txt".to_string(),
            file_type: FileType::Text,
            mime_type: Some("text/plain".to_string()),
            size: 100,
            hash: "abc123".to_string(),
            created_at: 1000,
            modified_at: 2000,
            indexed_at: 3000,
        }
    }

    #[tokio::test]
    async fn test_upsert_file() {
        let (db, _temp_dir) = create_test_db().await;
        let metadata = create_test_metadata();

        let file_id = db.upsert_file(&metadata).await.unwrap();
        assert!(file_id > 0);

        // Verify file was inserted
        let retrieved = db.get_file_by_path(&metadata.path).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.path, metadata.path);
        assert_eq!(retrieved.hash, metadata.hash);
    }

    #[tokio::test]
    async fn test_upsert_content() {
        let (db, _temp_dir) = create_test_db().await;
        let metadata = create_test_metadata();
        let file_id = db.upsert_file(&metadata).await.unwrap();

        let content = ExtractedContent {
            text: "Hello, world!".to_string(),
            word_count: 2,
            language: None,
        };

        db.upsert_content(file_id, &content).await.unwrap();

        // Verify content was inserted
        let retrieved = db.get_content(file_id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.text, content.text);
        assert_eq!(retrieved.word_count, content.word_count);
    }

    #[tokio::test]
    async fn test_needs_reindex() {
        let (db, _temp_dir) = create_test_db().await;
        let metadata = create_test_metadata();

        // File not in index yet
        assert!(db.needs_reindex(&metadata.path, &metadata.hash).await.unwrap());

        // Insert file
        db.upsert_file(&metadata).await.unwrap();

        // Same hash, doesn't need reindex
        assert!(!db.needs_reindex(&metadata.path, &metadata.hash).await.unwrap());

        // Different hash, needs reindex
        assert!(db.needs_reindex(&metadata.path, "different_hash").await.unwrap());
    }

    #[tokio::test]
    async fn test_count_files() {
        let (db, _temp_dir) = create_test_db().await;

        assert_eq!(db.count_files().await.unwrap(), 0);

        let metadata = create_test_metadata();
        db.upsert_file(&metadata).await.unwrap();

        assert_eq!(db.count_files().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_get_stats() {
        let (db, _temp_dir) = create_test_db().await;

        let metadata1 = FileMetadata {
            path: "/test/file1.txt".to_string(),
            ..create_test_metadata()
        };
        let metadata2 = FileMetadata {
            path: "/test/file2.rs".to_string(),
            file_type: FileType::Code,
            ..create_test_metadata()
        };

        db.upsert_file(&metadata1).await.unwrap();
        db.upsert_file(&metadata2).await.unwrap();

        let stats = db.get_stats().await.unwrap();
        assert_eq!(stats.total_files, 2);
        assert_eq!(stats.total_size, 200);
        assert_eq!(stats.by_type.len(), 2);
    }

    #[tokio::test]
    async fn test_delete_file() {
        let (db, _temp_dir) = create_test_db().await;
        let metadata = create_test_metadata();

        let file_id = db.upsert_file(&metadata).await.unwrap();
        assert_eq!(db.count_files().await.unwrap(), 1);

        db.delete_file(&metadata.path).await.unwrap();
        assert_eq!(db.count_files().await.unwrap(), 0);

        // Verify content was also deleted (CASCADE)
        let content = db.get_content(file_id).await.unwrap();
        assert!(content.is_none());
    }
}