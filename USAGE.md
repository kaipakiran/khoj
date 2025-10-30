# File Search - Usage Guide

A fast, offline, hybrid search engine for indexing and searching your files.

## Quick Start

### 1. Run the Demo

Index and search the source code:

```bash
cargo run --example index_and_search
```

This will index the `src/` directory and run example searches.

### 2. Index Your Own Folder

```bash
cargo run --example index_and_search -- /path/to/your/folder
```

For example, to index your Documents folder:

```bash
cargo run --example index_and_search -- ~/Documents
```

### 3. Interactive Search

For an interactive search experience:

```bash
cargo run --example index_folder -- /path/to/your/folder
```

This will index the folder and then let you enter search queries interactively.

## What Gets Indexed

The system automatically indexes:

- **Text files**: `.txt`, `.md`, etc.
- **Code files**: `.rs`, `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.go`, `.rb`, etc.
- **Markdown files**: Full content with syntax awareness

### Privacy-Aware Exclusions

By default, the following are excluded for privacy:
- `.git` directories
- `.ssh` directories
- `node_modules`
- `target` (Rust build artifacts)
- `passwords` directories
- `.gnupg` directories
- `*.key` files
- `*.pem` files

## Search Features

### Keyword Search (BM25)

The system uses Tantivy's BM25 algorithm for keyword-based search:

```rust
let results = search_engine.keyword_search("your query", 10)?;
```

- Searches both filename and file content
- Ranks results by relevance
- Fast full-text search

### Example Searches

```
Query: "embedding"
   1. mod.rs (score: 2.09)
   2. vector_store.rs (score: 2.09)

Query: "tantivy"
   1. tantivy_index.rs (score: 4.42)
   2. error.rs (score: 2.38)
```

## Programming API

### Basic Usage

```rust
use file_search::{
    config::PrivacyConfig,
    extractors::text,
    indexer::{metadata, walker},
    search::HybridSearch,
    storage::{Database, TantivyIndex, VectorStore},
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup paths
    let folder_to_index = PathBuf::from("/path/to/folder");
    let db_path = PathBuf::from("index/db.sqlite");
    let tantivy_path = PathBuf::from("index/tantivy");

    // 2. Initialize storage
    let db = Database::new(&db_path).await?;
    let mut tantivy_index = TantivyIndex::new(&tantivy_path)?;
    let vector_store = VectorStore::new(384)?;

    // 3. Configure file walker
    let privacy_config = PrivacyConfig::default();
    let walker = walker::FileWalker::new(privacy_config);

    // 4. Discover and index files
    let discovered = walker.walk(&folder_to_index)?;

    for disc_file in discovered {
        let metadata = metadata::extract_metadata(&disc_file.path, disc_file.file_type)?;
        let file_id = db.upsert_file(&metadata).await?;

        if let Ok(content) = text::extract_text(&disc_file.path, disc_file.file_type) {
            db.upsert_content(file_id, &content).await?;
            tantivy_index.upsert_document(
                file_id,
                &disc_file.path.to_string_lossy(),
                &metadata.filename,
                &content.text,
            )?;
        }
    }

    tantivy_index.commit()?;

    // 5. Create search engine
    let search_engine = HybridSearch::new(tantivy_index, vector_store);

    // 6. Search!
    let results = search_engine.keyword_search("rust async", 10)?;

    for result in results {
        println!("{}: {}", result.filename, result.score);
    }

    Ok(())
}
```

### Indexing Steps

1. **File Discovery**: Walk directory tree with privacy filters
2. **Metadata Extraction**: Extract file size, hash, timestamps
3. **Content Extraction**: Extract text from supported file types
4. **Indexing**: Store in SQLite + Tantivy for fast search
5. **Commit**: Finalize index for searching

### Search Options

```rust
// Keyword search only (BM25)
let results = search_engine.keyword_search("query", 10)?;

// Hybrid search (keyword + semantic) - requires embeddings
let embedding = embedding_model.embed("query")?;
let results = search_engine.hybrid_search(
    "query",
    Some(&embedding),
    10,
    0.7,  // keyword weight (0.0 to 1.0)
)?;

// Semantic search only - requires embeddings
let results = search_engine.semantic_search(&embedding, 10)?;
```

## Index Storage

Indexes are stored in:
- **SQLite database**: File metadata and content
- **Tantivy index**: Full-text search index
- **Vector store**: Embeddings (if semantic search enabled)

The index can be saved and loaded for persistence:

```rust
// Save vector store
vector_store.save("index/vectors.json")?;

// Load vector store
let vector_store = VectorStore::load("index/vectors.json")?;
```

## Performance

- **Indexing Speed**: ~100-1000 files/second (depends on file size)
- **Search Speed**: Sub-millisecond for keyword search
- **Memory**: Minimal (index stored on disk)
- **Disk Space**: ~10-20% of original file sizes

## Next Steps

### Enable Semantic Search

To enable full semantic search capabilities:

1. Download the ONNX model for `all-MiniLM-L6-v2`
2. Generate embeddings during indexing:
   ```rust
   let mut embedding_model = EmbeddingModel::new("model.onnx", "tokenizer.json")?;
   let embedding = embedding_model.embed(&content.text)?;
   vector_store.upsert(file_id, &embedding)?;
   ```
3. Use hybrid search for best results

### Build a CLI

A full CLI is planned with features like:
- `file-search index /path/to/folder`
- `file-search search "query"`
- `file-search watch /path/to/folder` (auto-reindex on changes)

## Examples

The `examples/` directory contains:

- `simple_search.rs`: Basic search demonstration
- `index_and_search.rs`: Index a folder and run searches
- `index_folder.rs`: Interactive indexing and search

Run any example with:
```bash
cargo run --example <example_name>
```

## Testing

Run the full test suite:

```bash
cargo test
```

All 53 tests should pass, including:
- File walker tests
- Metadata extraction tests
- Text extraction tests
- Tantivy indexing tests
- Vector store tests
- Hybrid search tests