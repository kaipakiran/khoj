//! Example demonstrating how to index a folder
//!
//! Run with: cargo run --example index_folder -- /path/to/folder

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
    println!("🔍 File Search - Folder Indexing Example\n");

    // Get folder path from command line argument
    let args: Vec<String> = env::args().collect();
    let folder_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        // Default to current directory
        env::current_dir()?
    };

    if !folder_path.exists() {
        eprintln!("Error: Path does not exist: {}", folder_path.display());
        eprintln!("Usage: cargo run --example index_folder -- /path/to/folder");
        std::process::exit(1);
    }

    println!("📁 Indexing folder: {}", folder_path.display());

    // Setup index paths (in temp directory for this example)
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
    let vector_store = VectorStore::new(384)?; // 384-dim for all-MiniLM-L6-v2

    // Configure file walker with privacy settings
    let privacy_config = PrivacyConfig::default();
    let walker = walker::FileWalker::new(privacy_config);

    // Discover files
    println!("🔎 Discovering files...");
    let discovered = walker.walk(&folder_path)?;
    println!("   Found {} files to index", discovered.len());
    println!();

    // Index each file
    println!("📚 Indexing files...");
    let mut indexed_count = 0;
    let mut skipped_count = 0;

    for disc_file in discovered {
        // Extract metadata
        let metadata = match metadata::extract_metadata(&disc_file.path, disc_file.file_type) {
            Ok(m) => m,
            Err(e) => {
                println!("   ⚠️  Skipped {}: {}", disc_file.path.display(), e);
                skipped_count += 1;
                continue;
            }
        };

        // Store in database
        let file_id = db.upsert_file(&metadata).await?;

        // Extract and index text content
        match text::extract_text(&disc_file.path, disc_file.file_type) {
            Ok(content) => {
                // Store content in database
                db.upsert_content(file_id, &content).await?;

                // Index in Tantivy for keyword search
                tantivy_index.upsert_document(
                    file_id,
                    &disc_file.path.to_string_lossy(),
                    &metadata.filename,
                    &content.text,
                )?;

                println!(
                    "   ✓ Indexed: {} ({} bytes, {} words)",
                    metadata.filename,
                    metadata.size,
                    content.word_count
                );
                indexed_count += 1;
            }
            Err(_) => {
                // For files we can't extract text from (images, etc.), just store metadata
                println!("   ○ Metadata only: {} ({})", metadata.filename, disc_file.file_type.as_str());
                skipped_count += 1;
            }
        }
    }

    // Commit the index
    println!();
    println!("💾 Committing index...");
    tantivy_index.commit()?;

    println!();
    println!("✅ Indexing complete!");
    println!("   {} files indexed", indexed_count);
    println!("   {} files skipped", skipped_count);
    println!();

    // Create search engine
    let search_engine = HybridSearch::new(tantivy_index, vector_store);

    // Interactive search loop
    println!("🔎 Search is ready! Enter queries (or 'quit' to exit):");
    println!();

    loop {
        print!("Search> ");
        use std::io::Write;
        std::io::stdout().flush()?;

        let mut query = String::new();
        std::io::stdin().read_line(&mut query)?;
        let query = query.trim();

        if query.is_empty() {
            continue;
        }

        if query == "quit" || query == "exit" {
            println!("👋 Goodbye!");
            break;
        }

        // Perform search
        match search_engine.keyword_search(query, 10) {
            Ok(results) => {
                if results.is_empty() {
                    println!("   No results found.");
                } else {
                    println!("   Found {} results:", results.len());
                    for (i, result) in results.iter().enumerate() {
                        println!("   {}. {} (score: {:.2})", i + 1, result.filename, result.score);
                        println!("      Path: {}", result.path);
                    }
                }
                println!();
            }
            Err(e) => {
                eprintln!("   Search error: {}", e);
                println!();
            }
        }
    }

    Ok(())
}