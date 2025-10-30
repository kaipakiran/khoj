# File Search - Multi-Modal Search System

A state-of-the-art, offline, lightweight file search system built in Rust with hybrid search capabilities (keyword + semantic).

## Project Goals

- **Fast**: Sub-50ms search latency
- **Lightweight**: <100MB binary (with models)
- **Offline**: 100% local, no cloud dependencies
- **Smart**: Semantic understanding via embeddings
- **Comprehensive**: Text, images, documents, code, spreadsheets

## Code Quality Standards

- **Clean & Readable**: Well-documented, clear variable names, logical structure
- **Tested**: Unit tests for each module, integration tests for workflows
- **No Dead Code**: Remove unused code, no commented-out blocks
- **Error Handling**: Proper error types with context using `anyhow::Result`
- **Modular**: Single responsibility principle, small focused functions
- **Documented**: Public APIs have doc comments with examples
- **Idiomatic Rust**: Follow Rust best practices and clippy suggestions

## Technology Stack

### Language
**Rust** - For maximum performance, safety, and single-binary deployment

### Core Dependencies
```toml
# Search & Indexing
tantivy = "0.22"              # Full-text search (BM25)
sqlx = { version = "0.7", features = ["sqlite", "runtime-tokio"] }

# Vector/Embeddings (Phase 2+)
ort = "2.0"                   # ONNX Runtime
hnsw = "0.11"                 # Vector similarity search

# Document Extraction
pdf-extract = "0.7"           # PDF text extraction
docx-rs = "0.4"               # Word documents
calamine = "0.24"             # Excel files

# Image Processing (Phase 3)
image = "0.24"                # Image loading
kamadak-exif = "0.5"          # EXIF metadata
rusty-tesseract = "1.1"       # OCR (optional)

# File Handling
walkdir = "2.4"               # Directory traversal
notify = "6.1"                # File watching
ignore = "0.4"                # .gitignore support
mime_guess = "2.0"
infer = "0.15"                # File type detection

# CLI
clap = { version = "4", features = ["derive"] }
indicatif = "0.17"            # Progress bars
colored = "2.0"
tabled = "0.15"               # Pretty tables

# Async & Parallelism
tokio = { version = "1", features = ["full"] }
rayon = "1.8"                 # Parallel processing

# Utilities
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"                 # File hashing
chrono = "0.4"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface                             │
│         (Search text, images, code, docs...)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────▼───────────┐
         │   Query Router         │
         │ (text/image/multimodal)│
         └────┬──────────────┬────┘
              │              │
    ┌─────────▼────┐    ┌───▼──────────┐
    │   Indexer    │    │Search Engine │
    │              │    │              │
    └──────┬───────┘    └───┬──────────┘
           │                │
    ┌──────▼─────────────────▼──────────────────┐
    │       Content Extractors                  │
    │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │
    │  │ Text │ │ PDF  │ │DOCX  │ │Code  │    │
    │  │      │ │+OCR  │ │XLSX  │ │Parser│    │
    │  └──────┘ └──────┘ └──────┘ └──────┘    │
    └───────────────────┬───────────────────────┘
                        │
    ┌───────────────────▼───────────────────────┐
    │         Embedding Models                  │
    │  ┌─────────────────────────────────┐     │
    │  │ Text: all-MiniLM-L6-v2 (22MB)   │     │
    │  └─────────────────────────────────┘     │
    │  ┌─────────────────────────────────┐     │
    │  │ Image: CLIP or MobileCLIP       │     │
    │  └─────────────────────────────────┘     │
    └───────────────────┬───────────────────────┘
                        │
    ┌───────────────────▼───────────────────────┐
    │            Storage Layer                  │
    │                                            │
    │  ┌────────────────────────────────────┐   │
    │  │  SQLite (metadata)                 │   │
    │  │  - File info, EXIF, tags           │   │
    │  │  - Extracted text, OCR             │   │
    │  │  - Code symbols                    │   │
    │  └────────────────────────────────────┘   │
    │                                            │
    │  ┌────────────────────────────────────┐   │
    │  │  Tantivy Index (BM25)              │   │
    │  │  - Full-text search                │   │
    │  └────────────────────────────────────┘   │
    │                                            │
    │  ┌────────────────────────────────────┐   │
    │  │  HNSW Vector Index                 │   │
    │  │  - Text embeddings (384D)          │   │
    │  │  - Image embeddings (512D)         │   │
    │  └────────────────────────────────────┘   │
    └────────────────────────────────────────────┘
```

## Project Structure

```
file-search/
├── Cargo.toml
├── README.md
├── PLAN.md (this file)
├── .gitignore
├── models/                       # AI models (downloaded separately)
│   ├── all-MiniLM-L6-v2.onnx
│   ├── clip-vit-b32-visual.onnx
│   └── tokenizers/
├── src/
│   ├── main.rs                   # CLI entry point
│   ├── lib.rs                    # Library exports
│   ├── config.rs                 # Configuration management
│   │
│   ├── indexer/
│   │   ├── mod.rs                # Indexing orchestration
│   │   ├── walker.rs             # File system traversal
│   │   ├── metadata.rs           # Metadata extraction
│   │   ├── chunker.rs            # Text chunking for embeddings
│   │   └── pipeline.rs           # Multi-threaded indexing
│   │
│   ├── extractors/
│   │   ├── mod.rs                # Extractor trait & factory
│   │   ├── text.rs               # Plain text files
│   │   ├── code.rs               # Source code (tree-sitter)
│   │   ├── pdf.rs                # PDF extraction
│   │   ├── docx.rs               # Word documents
│   │   ├── xlsx.rs               # Excel spreadsheets
│   │   ├── image.rs              # Images + EXIF
│   │   └── archive.rs            # ZIP, TAR
│   │
│   ├── ocr/
│   │   ├── mod.rs
│   │   └── tesseract.rs          # OCR engine
│   │
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── schema.sql            # SQLite schema
│   │   ├── sqlite.rs             # SQLite operations
│   │   ├── tantivy_index.rs      # Full-text index
│   │   └── vector_store.rs       # HNSW vector index
│   │
│   ├── embedding/
│   │   ├── mod.rs
│   │   ├── text_model.rs         # MiniLM embeddings
│   │   ├── image_model.rs        # CLIP embeddings
│   │   ├── tokenizer.rs
│   │   └── cache.rs              # Embedding cache
│   │
│   ├── search/
│   │   ├── mod.rs
│   │   ├── keyword.rs            # BM25 search
│   │   ├── semantic.rs           # Vector search
│   │   ├── image_search.rs       # Image similarity
│   │   ├── hybrid.rs             # Hybrid ranking (RRF)
│   │   ├── filters.rs            # Filters (date, size, type)
│   │   └── query.rs              # Query DSL parsing
│   │
│   ├── watcher/
│   │   └── mod.rs                # File system watcher
│   │
│   └── cli/
│       ├── mod.rs
│       ├── index.rs              # Index command
│       ├── search.rs             # Search command
│       ├── similar.rs            # Find similar files
│       ├── stats.rs              # Statistics
│       └── export.rs             # Export results
│
└── tests/
    ├── integration_test.rs
    └── fixtures/
        ├── sample.txt
        ├── sample.pdf
        ├── sample.docx
        ├── sample.xlsx
        ├── sample.jpg
        └── sample.py
```

## Database Schema

```sql
-- Files table
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,         -- 'text', 'pdf', 'image', 'code', etc.
    mime_type TEXT,
    size INTEGER,
    hash TEXT,                        -- SHA256 for change detection
    created_at INTEGER,
    modified_at INTEGER,
    indexed_at INTEGER
);

-- Content table (extracted text)
CREATE TABLE content (
    file_id INTEGER PRIMARY KEY,
    text TEXT,                        -- Extracted text content
    ocr_text TEXT,                    -- OCR'd text from images/PDFs
    language TEXT,                    -- Detected language
    word_count INTEGER,
    FOREIGN KEY (file_id) REFERENCES files(id)
);

-- Image metadata
CREATE TABLE image_metadata (
    file_id INTEGER PRIMARY KEY,
    width INTEGER,
    height INTEGER,
    camera_make TEXT,
    camera_model TEXT,
    lens TEXT,
    focal_length INTEGER,
    aperture REAL,
    iso INTEGER,
    gps_lat REAL,
    gps_lon REAL,
    gps_altitude REAL,
    location_name TEXT,               -- Reverse geocoded location
    location_country TEXT,
    location_region TEXT,
    taken_at INTEGER,
    timezone TEXT,
    detected_text TEXT,               -- OCR'd text
    dominant_color_hex TEXT,
    is_landscape BOOLEAN,
    FOREIGN KEY (file_id) REFERENCES files(id)
);

-- Code metadata
CREATE TABLE code_metadata (
    file_id INTEGER PRIMARY KEY,
    language TEXT,
    loc INTEGER,                      -- Lines of code
    functions TEXT,                   -- JSON array of function names
    classes TEXT,                     -- JSON array of class names
    imports TEXT,                     -- JSON array of imports
    FOREIGN KEY (file_id) REFERENCES files(id)
);

-- Document metadata
CREATE TABLE document_metadata (
    file_id INTEGER PRIMARY KEY,
    title TEXT,
    author TEXT,
    pages INTEGER,
    word_count INTEGER,
    created_date INTEGER,
    FOREIGN KEY (file_id) REFERENCES files(id)
);

-- Vectors (references to HNSW index)
CREATE TABLE vectors (
    file_id INTEGER,
    vector_type TEXT,                 -- 'text' or 'image'
    vector_id INTEGER,                -- ID in HNSW index
    chunk_index INTEGER DEFAULT 0,   -- For chunked documents
    FOREIGN KEY (file_id) REFERENCES files(id)
);

-- Tags (user-defined or auto-generated)
CREATE TABLE tags (
    file_id INTEGER,
    tag TEXT,
    source TEXT,                      -- 'user', 'auto', 'exif'
    FOREIGN KEY (file_id) REFERENCES files(id)
);

-- Indexes for performance
CREATE INDEX idx_files_type ON files(file_type);
CREATE INDEX idx_files_modified ON files(modified_at);
CREATE INDEX idx_files_hash ON files(hash);
CREATE INDEX idx_tags_tag ON tags(tag);
CREATE INDEX idx_image_location ON image_metadata(gps_lat, gps_lon);
```

## Phased Implementation Plan

### Phase 1: Text Search (Tonight) 🎯

**Goal**: Fast keyword search for text files

**Scope**:
- Text file indexing (txt, md, code files)
- Tantivy full-text search (BM25)
- SQLite metadata storage
- Basic CLI (index, search)
- File watcher for incremental updates

**File Types**: .txt, .md, .rs, .py, .js, .ts, .java, .c, .cpp, .go, .rb, etc.

**Features**:
- Fast keyword search
- Fuzzy matching
- Boolean queries (AND, OR, NOT)
- Basic filters (file type, date, size)

**CLI Commands**:
```bash
# Initialize index
file-search init /path/to/folder

# Index files
file-search index /path/to/folder

# Search
file-search search "query"
file-search search "rust async" --type rs
file-search search "TODO" --fuzzy

# Watch mode
file-search watch /path/to/folder

# Stats
file-search stats
```

**Deliverables**:
- Working CLI tool
- Fast text search
- ~5-10MB binary
- <10ms search latency

**Time Estimate**: 3-4 hours

---

### Phase 2: Document Support (Next)

**Goal**: Add PDF, DOCX, XLSX support

**Scope**:
- PDF text extraction
- Word document (.docx) extraction
- Excel spreadsheet (.xlsx) extraction
- Enhanced metadata extraction

**File Types**: .pdf, .docx, .doc, .xlsx, .xls, .pptx

**Features**:
- Extract text from documents
- Search inside PDFs and Word docs
- Search spreadsheet content
- Document metadata (author, title, pages)

**Time Estimate**: 2-3 hours

---

### Phase 3: Semantic Search (Text Embeddings)

**Goal**: Add semantic understanding for text

**Scope**:
- Integrate ONNX runtime
- Load all-MiniLM-L6-v2 model
- Generate text embeddings
- HNSW vector index
- Hybrid search (BM25 + semantic)

**Features**:
- Semantic similarity search
- "Find similar documents"
- Better ranking with hybrid search
- Reciprocal Rank Fusion (RRF)

**CLI Commands**:
```bash
file-search search "machine learning tutorial" --semantic
file-search search "ML algorithms" --hybrid
file-search similar /path/to/document.txt
```

**Time Estimate**: 3-4 hours

---

### Phase 4: Image Support + Visual Search

**Goal**: Index images with CLIP embeddings

**Scope**:
- Image metadata extraction (EXIF)
- CLIP image embeddings
- Visual similarity search
- GPS/location support

**File Types**: .jpg, .png, .gif, .bmp, .webp, .tiff

**Features**:
- Search images by visual content
- EXIF metadata extraction
- GPS/location search
- Find similar images

**CLI Commands**:
```bash
file-search search "mountain" --type image
file-search search "sunset beach" --type image
file-search similar photo.jpg
file-search search --near "Swiss Alps" --radius 50km
```

**Time Estimate**: 4-5 hours

---

### Phase 5: OCR Support

**Goal**: Extract text from images and scanned PDFs

**Scope**:
- Integrate Tesseract OCR
- OCR for images
- OCR for scanned PDFs
- Index OCR'd text

**Features**:
- Search text in images
- Find receipts, screenshots, business cards
- Scanned document search

**CLI Commands**:
```bash
file-search search "invoice" --type image --ocr
file-search search "receipt with total $500" --ocr
```

**Time Estimate**: 2-3 hours

---

### Phase 6: Advanced Features

**Goal**: Polish and advanced capabilities

**Scope**:
- Multi-modal search (text query → find images)
- Advanced filters and query DSL
- Export results (JSON, CSV)
- Web UI (optional)
- Performance optimization
- Encrypted index support

**Features**:
- Complex query language
- Batch operations
- API server mode
- Better ranking algorithms
- Caching and optimization

**Time Estimate**: 4-6 hours

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Indexing Speed | ~1000 files/sec (text) |
| Search Latency (keyword) | <10ms |
| Search Latency (semantic) | <50ms |
| Search Latency (hybrid) | <100ms |
| Memory Usage | <200MB typical |
| Binary Size | ~50MB (with models) |
| Startup Time | <100ms |

## Privacy & Security

**Privacy Controls**:
- All processing happens locally (offline)
- No telemetry or external API calls
- Configurable exclusion patterns
- Respect .gitignore and .searchignore
- Optional index encryption

**Default Exclusions**:
```
**/.git
**/.ssh
**/passwords
**/.gnupg
**/node_modules
**/target
**/*.key
**/*.pem
```

**Configuration** (`~/.config/file-search/config.toml`):
```toml
[privacy]
exclude_patterns = [
    "**/.git",
    "**/.ssh",
    "**/passwords",
    "**/.gnupg"
]
respect_ignore_files = [".gitignore", ".searchignore"]
max_file_size = "100MB"

[storage]
index_path = "~/.file-search/index"
encrypt = false

[search]
default_limit = 20
fuzzy_distance = 2
```

## Search Capabilities

### Text Search
```bash
# Simple keyword
file-search search "function parseJSON"

# Boolean queries
file-search search "rust AND async"
file-search search "TODO OR FIXME"
file-search search "bug NOT fixed"

# Fuzzy search
file-search search "algoritm" --fuzzy  # finds "algorithm"

# Filters
file-search search "class" --type py --after 2024-01-01
```

### Semantic Search
```bash
# Semantic similarity
file-search search "machine learning tutorial" --semantic

# Find similar documents
file-search similar document.pdf
```

### Image Search
```bash
# Visual content
file-search search "mountain sunset" --type image

# Location-based
file-search search --near "Paris" --radius 10km --type image

# EXIF filters
file-search search --camera "Canon" --date "2024-06"

# OCR in images
file-search search "invoice" --type image --ocr
```

### Multi-Modal Search
```bash
# Text query, any result type
file-search search "beach vacation"  # finds docs, photos, etc.

# Complex filters
file-search search "quarterly report" \
  --type pdf,docx \
  --after 2024-01-01 \
  --size ">1MB"
```

## Embedding Models

### Text Embeddings
- **all-MiniLM-L6-v2**: 22MB, 384 dimensions (recommended)
- **all-MiniLM-L12-v2**: 43MB, 384 dimensions (more accurate)
- **bge-small-en-v1.5**: 33MB, 384 dimensions (good quality)

### Image Embeddings
- **MobileCLIP-S2**: ~40MB, 512 dimensions (lightweight)
- **CLIP-ViT-B-32**: ~350MB, 512 dimensions (balanced)
- **CLIP-ViT-L-14**: ~900MB, 768 dimensions (best quality)

## Success Criteria

**Phase 1 Success**:
- ✅ Can index 10,000+ text files
- ✅ Search returns results in <10ms
- ✅ Fuzzy matching works
- ✅ File watcher updates index automatically
- ✅ Binary size <10MB

**Overall Project Success**:
- ✅ All file types supported
- ✅ Hybrid search working
- ✅ Sub-50ms semantic search
- ✅ Image search by content
- ✅ OCR functional
- ✅ Privacy controls in place
- ✅ Production-ready documentation

## Next Steps

1. **Tonight**: Implement Phase 1 (Text Search)
2. **Tomorrow**: Test Phase 1, start Phase 2 (Documents)
3. **Week 1**: Complete Phases 1-3 (Text + Semantic)
4. **Week 2**: Complete Phases 4-5 (Images + OCR)
5. **Week 3**: Phase 6 (Polish + Advanced)

---

**Started**: 2025-10-27
**Target Completion**: ~3 weeks for full system
**Phase 1 Target**: Tonight (3-4 hours)