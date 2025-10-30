//! Example demonstrating how to index a folder and search it
//!
//! Run with: cargo run --example index_and_search

use file_search::{
    config::PrivacyConfig,
    extractors::text,
    indexer::{metadata, walker},
    search::HybridSearch,
    storage::{Database, TantivyIndex, VectorStore},
};
use std::env;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 File Search - Index and Search Demo\n");

    // Get folder path from command line argument or use current directory
    let args: Vec<String> = env::args().collect();
    let folder_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        // Default to the src directory of this project
        PathBuf::from("src")
    };

    if !folder_path.exists() {
        eprintln!("Error: Path does not exist: {}", folder_path.display());
        std::process::exit(1);
    }

    println!("📁 Indexing folder: {}", folder_path.display());

    // Setup index paths (in temp directory)
    let index_base = env::temp_dir().join("file-search-demo");
    std::fs::create_dir_all(&index_base)?;

    let db_path = index_base.join("db.sqlite");
    let tantivy_path = index_base.join("tantivy");

    println!("💾 Index location: {}", index_base.display());
    println!();

    // Initialize storage components
    println!("🔧 Initializing storage...");
    let db = Database::new(&db_path).await?;
    let mut tantivy_index = TantivyIndex::new(&tantivy_path)?;
    let vector_store = VectorStore::new(384)?;

    // Configure file walker with privacy settings
    let privacy_config = PrivacyConfig::default();
    let walker = walker::FileWalker::new(privacy_config);

    // Discover files
    println!("🔎 Discovering files...");
    let discovered = walker.walk(&folder_path)?;
    println!("   Found {} files", discovered.len());
    println!();

    // Index each file
    println!("📚 Indexing files...");
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

                tantivy_index.upsert_document(
                    file_id,
                    &disc_file.path.to_string_lossy(),
                    &metadata.filename,
                    &content.text,
                )?;

                println!("   ✓ {} ({})", metadata.filename, disc_file.file_type.as_str());
                indexed_count += 1;
            }
            Err(e) => {
                println!("   ○ Skipped: {} ({}) - {}", metadata.filename, disc_file.file_type.as_str(), e);
                skipped_count += 1;
            }
        }
    }

    tantivy_index.commit()?;

    println!();
    println!("✅ Indexed {} files", indexed_count);
    if skipped_count > 0 {
        println!("⚠️  Skipped {} files (unsupported types or errors)", skipped_count);
    }
    println!();

    // Create search engine
    let search_engine = HybridSearch::new(tantivy_index, vector_store);

    // Perform some example searches
    println!("🔎 Example Searches:\n");

    let queries = vec![
        "kiran",
        "infosys",
        "geeta",
        "aditya",
        "siddhanth",
    ];

    for query in queries {
        println!("Query: \"{}\"", query);
        match search_engine.keyword_search(query, 5) {
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

    println!("✨ Demo complete!");
    println!("\n💡 To search your own folder:");
    println!("   cargo run --example index_and_search -- /path/to/your/folder");

    Ok(())
}