-- File Search Database Schema

-- Files table: Core metadata for all indexed files
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,  -- 'text', 'code', 'markdown', etc.
    mime_type TEXT,
    size INTEGER NOT NULL,
    hash TEXT NOT NULL,       -- SHA256 hash for change detection
    created_at INTEGER NOT NULL,
    modified_at INTEGER NOT NULL,
    indexed_at INTEGER NOT NULL
);

-- Content table: Extracted text content
CREATE TABLE IF NOT EXISTS content (
    file_id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    language TEXT,            -- For code files (rust, python, etc.)
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);

-- Vectors table: References to HNSW vector index
CREATE TABLE IF NOT EXISTS vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    vector_type TEXT NOT NULL, -- 'text' or 'image'
    vector_id INTEGER NOT NULL, -- ID in HNSW index
    chunk_index INTEGER DEFAULT 0, -- For chunked documents
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type);
CREATE INDEX IF NOT EXISTS idx_files_modified ON files(modified_at);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
CREATE INDEX IF NOT EXISTS idx_vectors_file_id ON vectors(file_id);
CREATE INDEX IF NOT EXISTS idx_vectors_type ON vectors(vector_type);

-- Full-text search index on content
CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
    text,
    content=content,
    content_rowid=file_id
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS content_ai AFTER INSERT ON content BEGIN
    INSERT INTO content_fts(rowid, text) VALUES (new.file_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS content_ad AFTER DELETE ON content BEGIN
    DELETE FROM content_fts WHERE rowid = old.file_id;
END;

CREATE TRIGGER IF NOT EXISTS content_au AFTER UPDATE ON content BEGIN
    UPDATE content_fts SET text = new.text WHERE rowid = new.file_id;
END;