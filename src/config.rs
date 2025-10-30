//! Configuration management for file-search

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for file-search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub storage: StorageConfig,
    pub search: SearchConfig,
    pub privacy: PrivacyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Path to store the index
    pub index_path: PathBuf,
    /// Enable index encryption
    pub encrypt: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Default number of results to return
    pub default_limit: usize,
    /// Fuzzy search edit distance
    pub fuzzy_distance: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Patterns to exclude from indexing
    pub exclude_patterns: Vec<String>,
    /// Respect these ignore files
    pub respect_ignore_files: Vec<String>,
    /// Maximum file size to index (in bytes)
    pub max_file_size: u64,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            exclude_patterns: vec![
                "**/.git".to_string(),
                "**/.ssh".to_string(),
                "**/passwords".to_string(),
                "**/.gnupg".to_string(),
                "**/node_modules".to_string(),
                "**/target".to_string(),
                "**/*.key".to_string(),
                "**/*.pem".to_string(),
            ],
            respect_ignore_files: vec![".gitignore".to_string(), ".searchignore".to_string()],
            max_file_size: 100 * 1024 * 1024, // 100MB
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            storage: StorageConfig {
                index_path: dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".file-search/index"),
                encrypt: false,
            },
            search: SearchConfig {
                default_limit: 20,
                fuzzy_distance: 2,
            },
            privacy: PrivacyConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.search.default_limit, 20);
        assert_eq!(config.search.fuzzy_distance, 2);
        assert!(!config.storage.encrypt);
        assert!(!config.privacy.exclude_patterns.is_empty());
    }
}