//! File system walker for traversing directories

use crate::config::PrivacyConfig;
use crate::types::FileType;
use crate::Result;
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};

/// Represents a discovered file during traversal
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    pub path: PathBuf,
    pub file_type: FileType,
    pub size: u64,
}

/// File system walker that respects .gitignore and privacy settings
pub struct FileWalker {
    privacy_config: PrivacyConfig,
}

impl FileWalker {
    /// Create a new file walker with privacy configuration
    pub fn new(privacy_config: PrivacyConfig) -> Self {
        Self { privacy_config }
    }

    /// Walk a directory and return all discovered files
    ///
    /// # Arguments
    /// * `root_path` - Root directory to start walking from
    ///
    /// # Returns
    /// Iterator over discovered files
    pub fn walk<P: AsRef<Path>>(&self, root_path: P) -> Result<Vec<DiscoveredFile>> {
        let root_path = root_path.as_ref();

        if !root_path.exists() {
            return Err(crate::Error::FileNotFound(
                root_path.display().to_string(),
            ));
        }

        let mut builder = WalkBuilder::new(root_path);

        // Respect .gitignore and .searchignore files
        if self.privacy_config.respect_ignore_files.contains(&".gitignore".to_string()) {
            builder.git_ignore(true);
        }

        let mut files = Vec::new();

        for result in builder.build() {
            let entry = match result {
                Ok(e) => e,
                Err(e) => {
                    // Gracefully skip permission denied errors (e.g., Photos Library on macOS)
                    if let Some(io_err) = e.io_error() {
                        if io_err.kind() == std::io::ErrorKind::PermissionDenied {
                            tracing::debug!("Skipping protected folder/file (permission denied): {}", e);
                            continue;
                        }
                    }
                    // For other errors, propagate them
                    return Err(e.into());
                }
            };
            let path = entry.path();

            // Skip directories
            if path.is_dir() {
                continue;
            }

            // Check exclusion patterns
            let should_exclude = self.privacy_config.exclude_patterns.iter().any(|pattern| {
                self.matches_pattern(path, pattern)
            });

            if should_exclude {
                tracing::debug!("Skipping excluded file: {}", path.display());
                continue;
            }

            // Get file metadata
            let metadata = match std::fs::metadata(path) {
                Ok(m) => m,
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    tracing::debug!("Skipping file (permission denied): {}", path.display());
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
            let size = metadata.len();

            // Skip files that are too large
            if size > self.privacy_config.max_file_size {
                tracing::debug!("Skipping large file: {} ({} bytes)", path.display(), size);
                continue;
            }

            // Determine file type
            let file_type = self.detect_file_type(path);

            // Skip archives, but include all other types (even Unknown)
            // We'll at least store metadata even if we can't extract text
            if matches!(file_type, FileType::Archive) {
                continue;
            }

            files.push(DiscoveredFile {
                path: path.to_path_buf(),
                file_type,
                size,
            });
        }

        Ok(files)
    }

    /// Check if a path matches an exclusion pattern
    fn matches_pattern(&self, path: &Path, pattern: &str) -> bool {
        // Simple glob-like pattern matching
        let path_str = path.to_string_lossy();

        if pattern.starts_with("**/") {
            let suffix = &pattern[3..];
            if suffix.starts_with("*") {
                // Pattern like "**/*.key" - match file extension
                let ext_pattern = &suffix[2..]; // Skip "*."
                path_str.ends_with(ext_pattern)
            } else {
                // Pattern like "**/.git" - match anywhere in path
                path_str.contains(suffix)
            }
        } else if pattern.starts_with("**") {
            // Match at end
            let suffix = &pattern[2..];
            path_str.ends_with(suffix)
        } else {
            // Exact match
            path_str.contains(pattern)
        }
    }

    /// Detect file type from path
    fn detect_file_type(&self, path: &Path) -> FileType {
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy();
            FileType::from_extension(&ext_str)
        } else {
            FileType::Unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_config() -> PrivacyConfig {
        PrivacyConfig {
            exclude_patterns: vec![
                "**/.git".to_string(),
                "**/node_modules".to_string(),
                "**/*.key".to_string(),
            ],
            respect_ignore_files: vec![".gitignore".to_string()],
            max_file_size: 10 * 1024 * 1024, // 10MB for tests
        }
    }

    #[test]
    fn test_walk_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let walker = FileWalker::new(create_test_config());

        let files = walker.walk(temp_dir.path()).unwrap();
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_walk_with_text_files() {
        let temp_dir = TempDir::new().unwrap();
        let walker = FileWalker::new(create_test_config());

        // Create test files
        fs::write(temp_dir.path().join("test.txt"), "hello").unwrap();
        fs::write(temp_dir.path().join("test.md"), "# Title").unwrap();
        fs::write(temp_dir.path().join("test.rs"), "fn main() {}").unwrap();

        let files = walker.walk(temp_dir.path()).unwrap();
        assert_eq!(files.len(), 3);

        // Check file types
        let txt_file = files.iter().find(|f| f.path.ends_with("test.txt")).unwrap();
        assert_eq!(txt_file.file_type, FileType::Text);

        let md_file = files.iter().find(|f| f.path.ends_with("test.md")).unwrap();
        assert_eq!(md_file.file_type, FileType::Markdown);

        let rs_file = files.iter().find(|f| f.path.ends_with("test.rs")).unwrap();
        assert_eq!(rs_file.file_type, FileType::Code);
    }

    #[test]
    fn test_walk_respects_exclusions() {
        let temp_dir = TempDir::new().unwrap();
        let walker = FileWalker::new(create_test_config());

        // Create files in excluded directory
        let node_modules = temp_dir.path().join("node_modules");
        fs::create_dir(&node_modules).unwrap();
        fs::write(node_modules.join("package.json"), "{}").unwrap();

        // Create normal file
        fs::write(temp_dir.path().join("test.txt"), "hello").unwrap();

        // Create excluded file type
        fs::write(temp_dir.path().join("secret.key"), "password").unwrap();

        let files = walker.walk(temp_dir.path()).unwrap();

        // Should only find test.txt
        assert_eq!(files.len(), 1);
        assert!(files[0].path.ends_with("test.txt"));
    }

    #[test]
    fn test_walk_nonexistent_directory() {
        let walker = FileWalker::new(create_test_config());
        let result = walker.walk("/nonexistent/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_file_type() {
        let walker = FileWalker::new(create_test_config());

        assert_eq!(
            walker.detect_file_type(Path::new("test.txt")),
            FileType::Text
        );
        assert_eq!(
            walker.detect_file_type(Path::new("test.rs")),
            FileType::Code
        );
        assert_eq!(
            walker.detect_file_type(Path::new("test.md")),
            FileType::Markdown
        );
        assert_eq!(
            walker.detect_file_type(Path::new("test.unknown")),
            FileType::Unknown
        );
    }

    #[test]
    fn test_matches_pattern() {
        let walker = FileWalker::new(create_test_config());

        // Test wildcard patterns
        assert!(walker.matches_pattern(Path::new("/path/to/.git/file"), "**/.git"));
        assert!(walker.matches_pattern(Path::new("/path/node_modules/pkg"), "**/node_modules"));
        assert!(walker.matches_pattern(Path::new("/path/secret.key"), "**/*.key"));

        // Test non-matches
        assert!(!walker.matches_pattern(Path::new("/path/to/file.txt"), "**/.git"));
    }
}