//! Semantic search example - search by meaning, not just keywords
//!
//! Run with: cargo run --example semantic_search -- ~/Documents

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
    println!("🔍 Semantic Search - Find documents by meaning, not just keywords\n");

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

    println!("📁 Folder: {}", folder_path.display());

    let index_base = env::temp_dir().join("file-search-semantic");
    std::fs::create_dir_all(&index_base)?;

    // Initialize
    println!("🤖 Loading AI model...");
    let mut embedding_model = EmbeddingModel::new("models/model.onnx", "models/tokenizer.json")?;

    let db = Database::new(&index_base.join("db.sqlite")).await?;
    let mut tantivy_index = TantivyIndex::new(&index_base.join("tantivy"))?;
    let vector_store = VectorStore::new(384)?;

    // Index files
    println!("📚 Indexing (this may take a minute)...");
    let walker = walker::FileWalker::new(PrivacyConfig::default());
    let discovered = walker.walk(&folder_path)?;

    let mut count = 0;
    for disc_file in discovered {
        if let Ok(metadata) = metadata::extract_metadata(&disc_file.path, disc_file.file_type) {
            let file_id = db.upsert_file(&metadata).await?;

            if let Ok(content) = text::extract_text(&disc_file.path, disc_file.file_type) {
                db.upsert_content(file_id, &content).await?;
                tantivy_index.upsert_document(
                    file_id,
                    &disc_file.path.to_string_lossy(),
                    &metadata.filename,
                    &content.text,
                )?;

                let text_chunk = if content.text.len() > 5000 {
                    &content.text[..5000]
                } else {
                    &content.text
                };

                if let Ok(embedding) = embedding_model.embed(text_chunk) {
                    vector_store.upsert(file_id, &embedding)?;
                    count += 1;
                    if count % 10 == 0 {
                        print!(".");
                        use std::io::Write;
                        std::io::stdout().flush()?;
                    }
                }
            }
        }
    }

    tantivy_index.commit()?;
    println!("\n✅ Indexed {} documents\n", count);

    let search_engine = HybridSearch::new(tantivy_index, vector_store);

    // Semantic search examples
    println!("🔎 Try these semantic searches:\n");

    let semantic_queries = vec![
        "tax documents and financial statements",
        "passport and identity documents",
        "employment and work history",
        "banking and credit information",
    ];

    for query in semantic_queries {
        println!("🔍 Query: \"{}\"", query);
        let query_embedding = embedding_model.embed(query)?;

        // Pure semantic search (keyword_weight = 0.0)
        match search_engine.hybrid_search(query, Some(&query_embedding), 3, 0.3) {
            Ok(results) => {
                if results.is_empty() {
                    println!("   No results found.");
                } else {
                    for (i, result) in results.iter().enumerate() {
                        println!("   {}. {}", i + 1, result.filename);
                    }
                }
            }
            Err(e) => eprintln!("   Error: {}", e),
        }
        println!();
    }

    println!("💡 Notice how it finds relevant documents even without exact keyword matches!");
    println!("   This is the power of semantic search with AI embeddings.\n");

    Ok(())
}