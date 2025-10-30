//! Simple example demonstrating keyword search
//!
//! Run with: cargo run --example simple_search

use file_search::{
    indexer::{metadata, walker},
    storage::{Database, TantivyIndex, VectorStore},
    search::HybridSearch,
    config::PrivacyConfig,
    extractors::text,
};
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 File Search - Simple Example\n");

    // Create temporary directories for testing
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("db.sqlite");
    let tantivy_path = temp_dir.path().join("tantivy");

    // Create test files
    let test_files_dir = temp_dir.path().join("files");
    std::fs::create_dir_all(&test_files_dir)?;

    println!("📝 Creating test files...");
    std::fs::write(
        test_files_dir.join("rust_intro.txt"),
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.",
    )?;

    std::fs::write(
        test_files_dir.join("python_intro.txt"),
        "Python is an interpreted high-level general-purpose programming language known for its simplicity.",
    )?;

    std::fs::write(
        test_files_dir.join("javascript_intro.txt"),
        "JavaScript is a programming language that conforms to the ECMAScript specification and runs in web browsers.",
    )?;

    // Initialize components
    println!("\n🔧 Initializing search engine...");
    let db = Database::new(&db_path).await?;
    let mut tantivy_index = TantivyIndex::new(&tantivy_path)?;
    let vector_store = VectorStore::new(384)?; // 384-dim for all-MiniLM-L6-v2

    // Index the files
    println!("\n📚 Indexing files...");
    let privacy_config = PrivacyConfig::default();
    let walker = walker::FileWalker::new(privacy_config);
    let discovered = walker.walk(&test_files_dir)?;

    println!("   Found {} files", discovered.len());

    for disc_file in discovered {
        // Extract metadata
        let metadata = metadata::extract_metadata(&disc_file.path, disc_file.file_type)?;

        // Store in database
        let file_id = db.upsert_file(&metadata).await?;

        // Extract text content
        if let Ok(content) = text::extract_text(&disc_file.path, disc_file.file_type) {
            // Store content in database
            db.upsert_content(file_id, &content).await?;

            // Index in Tantivy
            tantivy_index.upsert_document(
                file_id,
                &disc_file.path.to_string_lossy(),
                &metadata.filename,
                &content.text,
            )?;

            println!("   ✓ Indexed: {}", metadata.filename);
        }
    }

    tantivy_index.commit()?;
    println!("\n✅ Indexing complete!");

    // Create hybrid search engine (keyword only for now)
    let search_engine = HybridSearch::new(tantivy_index, vector_store);

    // Perform searches
    println!("\n🔎 Searching...\n");

    // Search 1: Programming languages
    println!("Query: \"programming language\"");
    let results = search_engine.keyword_search("programming language", 5)?;
    for (i, result) in results.iter().enumerate() {
        println!("  {}. {} (score: {:.2})", i + 1, result.filename, result.score);
    }

    // Search 2: Specific language
    println!("\nQuery: \"rust\"");
    let results = search_engine.keyword_search("rust", 5)?;
    for (i, result) in results.iter().enumerate() {
        println!("  {}. {} (score: {:.2})", i + 1, result.filename, result.score);
        if let Some(snippet) = &result.snippet {
            println!("     Preview: {}...", &snippet[..snippet.len().min(60)]);
        }
    }

    // Search 3: Another specific term
    println!("\nQuery: \"blazingly fast\"");
    let results = search_engine.keyword_search("blazingly fast", 5)?;
    for (i, result) in results.iter().enumerate() {
        println!("  {}. {} (score: {:.2})", i + 1, result.filename, result.score);
    }

    println!("\n✨ Search demo complete!");
    println!("\n💡 Next steps:");
    println!("   - Download ONNX model for semantic search");
    println!("   - Use hybrid_search() to combine keyword + semantic");
    println!("   - Index your actual documents!");

    Ok(())
}