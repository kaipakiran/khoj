# File Search

A fast, offline, lightweight hybrid search engine for files with semantic understanding.

## Features

**Phase 1 (Current): Hybrid Text Search**
- ⚡ Fast keyword search (BM25 via Tantivy)
- 🧠 Semantic search (embeddings via all-MiniLM-L6-v2)
- 🔀 Hybrid ranking (combines both approaches)
- 📁 Text file indexing (txt, md, source code)
- 🔍 Fuzzy matching
- ⏱️ Real-time file watching

**Coming Soon**:
- 📄 Document support (PDF, DOCX, XLSX)
- 🖼️ Image search with visual embeddings (CLIP)
- 📍 Location-based search (GPS from EXIF)
- 🔤 OCR for images and scanned PDFs

## Installation

```bash
# Clone and build
git clone <repo>
cd file-search
cargo build --release

# The binary will be at target/release/file-search
```

## Usage

```bash
# Index a directory
file-search index /path/to/folder

# Search (keyword)
file-search search "query"

# Semantic search
file-search search "machine learning" --semantic

# Hybrid search (best of both)
file-search search "rust async" --hybrid

# With filters
file-search search "TODO" --type rs --after 2024-01-01

# Watch for changes
file-search watch /path/to/folder

# Statistics
file-search stats
```

## Architecture

Built in Rust for maximum performance and safety:

- **Tantivy**: Full-text search with BM25
- **SQLite**: Metadata storage
- **ONNX Runtime**: Embedding models
- **HNSW**: Fast vector similarity search

## Development

```bash
# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run -- search "test"

# Format code
cargo fmt

# Lint
cargo clippy -- -D warnings

# Build optimized release
cargo build --release
```

## Performance Targets

- Indexing: ~1000 files/sec
- Search (keyword): <10ms
- Search (semantic): <50ms
- Memory: <200MB
- Binary: ~50MB (with models)

## Privacy

100% offline, no telemetry or external API calls. All processing happens locally on your machine.

## License

MIT
