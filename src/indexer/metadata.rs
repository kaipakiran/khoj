//! File metadata extraction

use crate::types::{FileMetadata, FileType};
use crate::Result;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::time::SystemTime;

/// Extract metadata from a file
///
/// # Arguments
/// * `path` - Path to the file
/// * `file_type` - Detected file type
///
/// # Returns
/// File metadata including hash, size, timestamps
pub fn extract_metadata(path: &Path, file_type: FileType) -> Result<FileMetadata> {
    let metadata = fs::metadata(path)?;

    // Get file size
    let size = metadata.len();

    // Compute file hash (SHA256)
    let hash = compute_file_hash(path)?;

    // Get timestamps
    let created_at = metadata
        .created()
        .or_else(|_| metadata.modified())
        .unwrap_or(SystemTime::now())
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let modified_at = metadata
        .modified()
        .unwrap_or(SystemTime::now())
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let indexed_at = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    // Extract filename
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Detect MIME type
    let mime_type = mime_guess::from_path(path)
        .first()
        .map(|m| m.to_string());

    Ok(FileMetadata {
        id: 0, // Will be set by database
        path: path.to_string_lossy().to_string(),
        filename,
        file_type,
        mime_type,
        size,
        hash,
        created_at,
        modified_at,
        indexed_at,
    })
}

/// Compute SHA256 hash of a file
fn compute_file_hash(path: &Path) -> Result<String> {
    let contents = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// Check if a file has been modified since last index
///
/// # Arguments
/// * `path` - Path to the file
/// * `stored_hash` - Previously stored hash
///
/// # Returns
/// True if file has been modified (hash differs)
pub fn is_modified(path: &Path, stored_hash: &str) -> Result<bool> {
    let current_hash = compute_file_hash(path)?;
    Ok(current_hash != stored_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_extract_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // Create test file
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"Hello, world!").unwrap();
        file.sync_all().unwrap();

        let metadata = extract_metadata(&file_path, FileType::Text).unwrap();

        assert_eq!(metadata.filename, "test.txt");
        assert_eq!(metadata.file_type, FileType::Text);
        assert_eq!(metadata.size, 13);
        assert!(!metadata.hash.is_empty());
        assert_eq!(metadata.hash.len(), 64); // SHA256 hex string length
        assert!(metadata.modified_at > 0);
        assert!(metadata.indexed_at > 0);
    }

    #[test]
    fn test_compute_file_hash() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // Create file with known content
        fs::write(&file_path, "test content").unwrap();

        let hash = compute_file_hash(&file_path).unwrap();

        // Verify hash is consistent
        let hash2 = compute_file_hash(&file_path).unwrap();
        assert_eq!(hash, hash2);

        // Verify hash changes with content
        fs::write(&file_path, "different content").unwrap();
        let hash3 = compute_file_hash(&file_path).unwrap();
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_is_modified() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // Create initial file
        fs::write(&file_path, "original").unwrap();
        let original_hash = compute_file_hash(&file_path).unwrap();

        // File should not be modified
        assert!(!is_modified(&file_path, &original_hash).unwrap());

        // Modify file
        fs::write(&file_path, "modified").unwrap();

        // File should be detected as modified
        assert!(is_modified(&file_path, &original_hash).unwrap());
    }

    #[test]
    fn test_mime_type_detection() {
        let temp_dir = TempDir::new().unwrap();

        // Test various file types - note: mime_guess may return different mime types
        // depending on the system and version, so we just check that mime_type is detected
        let test_cases = vec![
            ("test.txt", FileType::Text),
            ("test.md", FileType::Markdown),
            ("test.rs", FileType::Code),
            ("test.py", FileType::Code),
        ];

        for (filename, file_type) in test_cases {
            let file_path = temp_dir.path().join(filename);
            fs::write(&file_path, "content").unwrap();

            let metadata = extract_metadata(&file_path, file_type).unwrap();

            // Just verify that some mime type was detected
            assert!(
                metadata.mime_type.is_some(),
                "Expected mime type for {}, got None",
                filename
            );
        }
    }

    #[test]
    fn test_extract_metadata_nonexistent_file() {
        let result = extract_metadata(Path::new("/nonexistent/file.txt"), FileType::Text);
        assert!(result.is_err());
    }
}