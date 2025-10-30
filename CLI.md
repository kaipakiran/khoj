# File Search CLI

A fast, offline hybrid search engine for your files with AI-powered semantic search.

## Installation

```bash
# Build the CLI
cargo build --release

# The binary will be at: target/release/file-search

# Optionally, install it globally
cargo install --path .
```

## Quick Start

### 1. Index Your Files

**Basic indexing (keyword search only):**
```bash
file-search index ~/Documents
```

**With semantic AI search:**
```bash
file-search index ~/Documents --semantic
```

### 2. Search

**Keyword search (fast):**
```bash
file-search search "tax form"
```

**Semantic search (finds by meaning):**
```bash
file-search search "employment documents" --semantic
```

**Hybrid search with custom weighting:**
```bash
file-search search "passport" --semantic --keyword-weight 0.5
```

### 3. Manage Index

**View statistics:**
```bash
file-search stats
```

**List indexed files:**
```bash
file-search list
```

**Clear index:**
```bash
file-search clear
```

## Commands

### `file-search index <PATH>`

Index a folder for searching.

**Options:**
- `-s, --semantic` - Enable AI semantic search (requires ONNX model)
- `-v, --verbose` - Show progress for each file
- `--index-dir <DIR>` - Custom index location (default: `~/.file-search`)

**Examples:**
```bash
# Basic indexing
file-search index ~/Documents

# With semantic search
file-search index ~/Documents --semantic

# Verbose output
file-search index ~/Documents --verbose

# Custom index location
file-search index ~/Documents --index-dir /path/to/index
```

### `file-search search <QUERY>`

Search indexed files.

**Options:**
- `-l, --limit <N>` - Number of results (default: 10)
- `-s, --semantic` - Use semantic AI search
- `--keyword-weight <0.0-1.0>` - Balance between keyword/semantic (default: 0.7)

**Examples:**
```bash
# Keyword search
file-search search "tax 2024"

# Semantic search
file-search search "financial documents" --semantic

# More semantic, less keyword
file-search search "work history" --semantic --keyword-weight 0.3

# Show more results
file-search search "pdf" --limit 50
```

### `file-search stats`

Show index statistics.

### `file-search list`

List all indexed files.

### `file-search clear`

Delete the index.

**Options:**
- `-y, --yes` - Skip confirmation

## Supported File Types

### Fully Indexed (Text Searchable)

- **Text files**: `.txt`, `.md`
- **Code files**: `.rs`, `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.go`, etc.
- **Documents**: `.pdf`, `.docx`
- **Web files**: `.html`, `.css`, `.js`, `.json`, `.xml`

### Partially Supported

- **Excel files** (`.xlsx`) - Metadata only (extraction coming soon)
- **Images** (`.jpg`, `.png`) - Metadata only (OCR planned)

### Not Supported

- **Videos** (`.mp4`, `.avi`)
- **Archives** (`.zip`, `.tar`)
- **Encrypted PDFs** (require password)

## How It Works

### Keyword Search (BM25)

Uses Tantivy's BM25 algorithm for fast, exact keyword matching:
```bash
file-search search "invoice 2024"
```
- Finds documents containing those exact words
- Very fast (milliseconds)
- Best for known terms

### Semantic Search (AI Embeddings)

Uses `all-MiniLM-L6-v2` model for understanding meaning:
```bash
file-search search "employment records" --semantic
```
- Finds documents by meaning, not just keywords
- Finds "resume", "W2", "job history" even without those exact words
- Requires indexing with `--semantic` flag

### Hybrid Search (Best of Both)

Combines both approaches with Reciprocal Rank Fusion:
```bash
file-search search "tax documents" --semantic --keyword-weight 0.7
```
- 70% keyword matching, 30% semantic similarity
- Best overall results
- Adjust `--keyword-weight` to tune

## Index Storage

By default, the index is stored at `~/.file-search/`:

```
~/.file-search/
├── db.sqlite        # File metadata and content
├── tantivy/         # Keyword search index
└── vectors.json     # Semantic embeddings (if --semantic used)
```

**Size:** About 10-20% of your original file sizes.

## Performance

- **Indexing**: 100-1000 files/second (depends on file size)
- **Keyword search**: < 1ms (instant)
- **Semantic search**: ~100ms (includes AI inference)
- **Hybrid search**: ~100ms

## Privacy

- **100% offline** - No data sent to external servers
- **Local storage** - All indexes stored on your machine
- **Respects `.gitignore`** - Won't index excluded files
- **Privacy filters** - Excludes `.ssh`, passwords, keys by default

## Semantic Search Setup

To use semantic search, download the AI model:

```bash
mkdir -p models
curl -L -o models/model.onnx https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
curl -L -o models/tokenizer.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
```

Model size: ~86 MB

## Tips

### Re-indexing

To update the index after files change:
```bash
file-search index ~/Documents
```
This will update existing files and add new ones.

### Multiple Folders

You can index multiple folders:
```bash
file-search index ~/Documents
file-search index ~/Projects
file-search index ~/Downloads
```

All will be searchable with a single `search` command.

### Search Tips

**For best results:**
- Use keyword search for exact terms: `file-search search "invoice-2024.pdf"`
- Use semantic for concepts: `file-search search "tax documents" --semantic`
- Use hybrid for everything else: `file-search search "work contracts" --semantic`

## Troubleshooting

### "No index found" error

You haven't indexed any folders yet:
```bash
file-search index ~/Documents
```

### Semantic search not working

1. Make sure you indexed with `--semantic`:
   ```bash
   file-search index ~/Documents --semantic
   ```

2. Check if the model is downloaded:
   ```bash
   ls -lh models/
   ```

### Slow indexing

- Normal for first-time indexing with semantic search
- PDF extraction can be slow for large/complex files
- Use `--verbose` to see progress

### Files not found

Check if they're excluded:
- `.gitignore` files are respected
- Private directories (`.ssh`, etc.) are excluded
- Check `file-search stats` to see what was indexed

## Examples

### Daily Usage

```bash
# Morning: index new downloads
file-search index ~/Downloads

# Find that tax document
file-search search "w2 2024"

# Find resume for job application
file-search search "resume software engineer" --semantic

# Find bank statements
file-search search "bank statement march" --semantic
```

### Advanced Usage

```bash
# Index everything with semantic search
file-search index ~/ --semantic

# Pure semantic search (no keywords)
file-search search "identity documents" --semantic --keyword-weight 0.0

# Pure keyword search (no AI)
file-search search "exact filename.pdf" --keyword-weight 1.0

# Show all PDFs
file-search search "pdf" --limit 100
```