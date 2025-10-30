//! Hybrid search demo with semantic embeddings
//!
//! Run with: cargo run --example hybrid_search_demo -- /path/to/folder

use file_search::{
    config::PrivacyConfig,
    embedding::EmbeddingModel,
    extractors::text,
    indexer::{metadata, walker},
    search::HybridSearch,
    storage::{Database, TantivyIndex, VectorStore},
};
use std::env;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 File Search - Hybrid Search Demo (BM25 + Semantic)\n");

    // Get folder path from command line argument
    let args: Vec<String> = env::args().collect();
    let folder_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("src")
    };

    if !folder_path.exists() {
        eprintln!("Error: Path does not exist: {}", folder_path.display());
        std::process::exit(1);
    }

    println!("📁 Indexing folder: {}", folder_path.display());

    // Setup index paths
    let index_base = env::temp_dir().join("file-search-hybrid");
    std::fs::create_dir_all(&index_base)?;

    let db_path = index_base.join("db.sqlite");
    let tantivy_path = index_base.join("tantivy");

    println!("💾 Index location: {}", index_base.display());
    println!();

    // Initialize embedding model
    println!("🤖 Loading embedding model (all-MiniLM-L6-v2)...");
    let model_path = PathBuf::from("models/model.onnx");
    let tokenizer_path = PathBuf::from("models/tokenizer.json");

    if !model_path.exists() {
        eprintln!("Error: ONNX model not found at {}", model_path.display());
        eprintln!("Please download it first:");
        eprintln!("  mkdir -p models");
        eprintln!("  curl -L -o models/model.onnx https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx");
        std::process::exit(1);
    }

    let mut embedding_model = EmbeddingModel::new(&model_path, &tokenizer_path)?;
    println!("   ✓ Model loaded successfully");
    println!();

    // Initialize storage components
    println!("🔧 Initializing storage...");
    let db = Database::new(&db_path).await?;
    let mut tantivy_index = TantivyIndex::new(&tantivy_path)?;
    let vector_store = VectorStore::new(384)?; // 384-dim for all-MiniLM-L6-v2

    // Configure file walker
    let privacy_config = PrivacyConfig::default();
    let walker = walker::FileWalker::new(privacy_config);

    // Discover files
    println!("🔎 Discovering files...");
    let discovered = walker.walk(&folder_path)?;
    println!("   Found {} files", discovered.len());
    println!();

    // Index each file with embeddings
    println!("📚 Indexing files with embeddings...");
    let mut indexed_count = 0;
    let mut skipped_count = 0;

    for disc_file in discovered {
        let metadata = match metadata::extract_metadata(&disc_file.path, disc_file.file_type) {
            Ok(m) => m,
            Err(e) => {
                println!("   ⚠️  Metadata error: {} - {}", disc_file.path.file_name().unwrap().to_string_lossy(), e);
                skipped_count += 1;
                continue;
            }
        };

        let file_id = db.upsert_file(&metadata).await?;

        match text::extract_text(&disc_file.path, disc_file.file_type) {
            Ok(content) => {
                db.upsert_content(file_id, &content).await?;

                // Index in Tantivy for keyword search
                tantivy_index.upsert_document(
                    file_id,
                    &disc_file.path.to_string_lossy(),
                    &metadata.filename,
                    &content.text,
                )?;

                // Generate embedding for semantic search
                let text_for_embedding = if content.text.len() > 5000 {
                    // Truncate very long texts to first 5000 chars for embedding
                    &content.text[..5000]
                } else {
                    &content.text
                };

                match embedding_model.embed(text_for_embedding) {
                    Ok(embedding) => {
                        vector_store.upsert(file_id, &embedding)?;
                        println!("   ✓ {} ({}, embedded)", metadata.filename, disc_file.file_type.as_str());
                        indexed_count += 1;
                    }
                    Err(e) => {
                        println!("   ⚠️  Embedding failed for {}: {}", metadata.filename, e);
                        // Still indexed in Tantivy, just no semantic search
                        indexed_count += 1;
                    }
                }
            }
            Err(e) => {
                println!("   ○ Skipped: {} ({}) - {}", metadata.filename, disc_file.file_type.as_str(), e);
                skipped_count += 1;
            }
        }
    }

    tantivy_index.commit()?;

    println!();
    println!("✅ Indexed {} files with embeddings", indexed_count);
    if skipped_count > 0 {
        println!("⚠️  Skipped {} files", skipped_count);
    }
    println!();

    // Save vector store for future use
    let vector_path = index_base.join("vectors.json");
    vector_store.save(&vector_path)?;
    println!("💾 Saved vector store to {}", vector_path.display());
    println!();

    // Create hybrid search engine
    let search_engine = HybridSearch::new(tantivy_index, vector_store);

    // Perform hybrid searches
    println!("🔎 Hybrid Search Examples (BM25 + Semantic):\n");

    let queries = vec![
        ("embedding model", "Exact keyword match"),
        ("vector similarity", "Semantic concept"),
        ("database storage", "Related concepts"),
        ("text processing", "Broad topic"),
    ];

    for (query, description) in queries {
        println!("Query: \"{}\" ({})", query, description);

        // Generate query embedding
        let query_embedding = embedding_model.embed(query)?;

        // Hybrid search (70% keyword, 30% semantic)
        match search_engine.hybrid_search(query, Some(&query_embedding), 5, 0.7) {
            Ok(results) => {
                if results.is_empty() {
                    println!("   No results found.");
                } else {
                    for (i, result) in results.iter().enumerate() {
                        println!("   {}. {} (score: {:.2})", i + 1, result.filename, result.score);
                    }
                }
            }
            Err(e) => {
                eprintln!("   Error: {}", e);
            }
        }
        println!();
    }

    println!("✨ Hybrid search demo complete!");
    println!("\n💡 Tips:");
    println!("   - Hybrid search combines keyword matching + semantic similarity");
    println!("   - Finds documents even if exact words don't match");
    println!("   - keyword_weight: 1.0 = pure keyword, 0.0 = pure semantic");
    println!("   - Default 0.7 gives 70% keyword, 30% semantic");

    Ok(())
}