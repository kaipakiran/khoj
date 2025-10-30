use clap::{Parser, Subcommand};
use khoj::{
    config::PrivacyConfig,
    embedding::{EmbeddingModel, image::{ImageEmbedding, ClipTextEmbedding}},
    extractors::text,
    indexer::{metadata, walker},
    search::HybridSearch,
    storage::{Database, TantivyIndex, VectorStore},
    types::FileType,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "khoj")]
#[command(about = "खोज - A fast, offline hybrid search engine for files", long_about = None)]
struct Cli {
    /// Search query (if no subcommand provided)
    query: Option<String>,

    /// Number of results to return
    #[arg(long, short, default_value = "10")]
    limit: usize,

    /// Use semantic search (requires indexed with --semantic)
    #[arg(long, short)]
    semantic: bool,

    /// Keyword weight for hybrid search (0.0-1.0, default 0.7)
    #[arg(long, default_value = "0.7")]
    keyword_weight: f32,

    /// Index directory (default: ~/.khoj)
    #[arg(long, global = true)]
    index_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Index a folder for searching
    Index {
        /// Folder to index
        path: PathBuf,

        /// Enable semantic search (requires ONNX model)
        #[arg(long, short)]
        semantic: bool,

        /// Show progress for each file
        #[arg(long, short)]
        verbose: bool,
    },

    /// Start web interface
    Serve {
        /// Port to listen on
        #[arg(long, short, default_value = "3000")]
        port: u16,
    },

    /// Show statistics about the index
    Stats,

    /// List all indexed files
    List {
        /// Number of files to show
        #[arg(long, short, default_value = "20")]
        limit: usize,
    },

    /// Clear the index
    Clear {
        /// Skip confirmation prompt
        #[arg(long, short)]
        yes: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Get index directory
    let index_dir = cli.index_dir.unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".khoj")
    });

    std::fs::create_dir_all(&index_dir)?;

    match cli.command {
        Some(Commands::Index { path, semantic, verbose }) => {
            index_folder(&path, &index_dir, semantic, verbose).await?;
        }
        Some(Commands::Serve { port }) => {
            khoj::web::serve(index_dir, port).await?;
        }
        Some(Commands::Stats) => {
            show_stats(&index_dir).await?;
        }
        Some(Commands::List { limit }) => {
            list_files(&index_dir, limit).await?;
        }
        Some(Commands::Clear { yes }) => {
            clear_index(&index_dir, yes)?;
        }
        None => {
            // Default action: search
            if let Some(query) = cli.query {
                search_index(&query, &index_dir, cli.limit, cli.semantic, cli.keyword_weight).await?;
            } else {
                eprintln!("Error: Please provide a search query or use a subcommand");
                eprintln!("");
                eprintln!("Examples:");
                eprintln!("  khoj \"tax forms\"              # Search for tax forms");
                eprintln!("  khoj index ~/Documents        # Index your documents");
                eprintln!("  khoj stats                    # Show index stats");
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

/// Find a model file in common locations
fn find_model_path(filename: &str) -> Option<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    let mut possible_paths = vec![
        // Current directory
        Some(PathBuf::from(format!("models/{}", filename))),
        // Project directory (if running from subdirectory)
        Some(PathBuf::from(format!("../models/{}", filename))),
        // Home directory
        dirs::home_dir().map(|h| h.join(".khoj/models").join(filename)),
    ];

    // Add executable directory paths
    if let Some(ref d) = exe_dir {
        possible_paths.push(Some(d.join("models").join(filename)));
        possible_paths.push(d.parent().map(|p| p.join("models").join(filename)));
        possible_paths.push(d.parent().and_then(|p| p.parent()).map(|p| p.join("models").join(filename)));
        possible_paths.push(d.parent().and_then(|p| p.parent()).and_then(|p| p.parent()).map(|p| p.join("models").join(filename)));
    }

    for path in possible_paths.into_iter().flatten() {
        if path.exists() {
            return Some(path);
        }
    }

    None
}

async fn index_folder(
    path: &PathBuf,
    index_dir: &PathBuf,
    enable_semantic: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use colored::Colorize;
    use indicatif::{ProgressBar, ProgressStyle};

    if !path.exists() {
        eprintln!("{} Path does not exist: {}", "Error:".red().bold(), path.display());
        std::process::exit(1);
    }

    println!("{} {}", "Indexing:".cyan().bold(), path.display());
    println!("{} {}", "Index location:".cyan(), index_dir.display());
    println!();

    // Initialize storage
    let db_path = index_dir.join("db.sqlite");
    let tantivy_path = index_dir.join("tantivy");
    let vector_path = index_dir.join("vectors.json");
    let image_vector_path = index_dir.join("image_vectors.json");

    let db = Database::new(&db_path).await?;
    let mut tantivy_index = TantivyIndex::new(&tantivy_path)?;
    let vector_store = VectorStore::new(384)?;
    let image_vector_store = VectorStore::new(512)?; // CLIP image embeddings are 512-dim

    // Initialize embedding model if semantic search is enabled
    let mut embedding_model = if enable_semantic {
        println!("{}", "Loading AI model for semantic search...".cyan());

        let model_path = find_model_path("model.onnx").unwrap_or_else(|| {
            eprintln!("{}", "Error: ONNX model not found!".red().bold());
            eprintln!("Searched in:");
            eprintln!("  - ./models/model.onnx");
            eprintln!("  - ~/.khoj/models/model.onnx");
            eprintln!("");
            eprintln!("Download with:");
            eprintln!("  mkdir -p models");
            eprintln!("  curl -L -o models/model.onnx https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx");
            std::process::exit(1);
        });

        let tokenizer_path = find_model_path("tokenizer.json").unwrap_or_else(|| {
            eprintln!("{}", "Error: Tokenizer not found!".red().bold());
            std::process::exit(1);
        });

        Some(EmbeddingModel::new(&model_path, &tokenizer_path)?)
    } else {
        None
    };

    // Initialize image embedding model for visual search
    let mut image_embedding_model = if enable_semantic {
        if let Some(clip_model_path) = find_model_path("clip_vision.onnx") {
            println!("{}", "Loading CLIP model for image search...".cyan());
            match ImageEmbedding::new(&clip_model_path) {
                Ok(model) => Some(model),
                Err(e) => {
                    eprintln!("{} Failed to load CLIP model: {}", "Warning:".yellow().bold(), e);
                    eprintln!("Image search will be disabled. Continuing with text search only.");
                    None
                }
            }
        } else {
            println!("{}", "Note: CLIP model not found. Image search will be disabled.".yellow());
            println!("Download with:");
            println!("  curl -L -o models/clip_vision.onnx https://huggingface.co/Qdrant/clip-ViT-B-32-vision/resolve/main/model.onnx");
            None
        }
    } else {
        None
    };

    // Discover files
    let privacy_config = PrivacyConfig::default();
    let walker = walker::FileWalker::new(privacy_config);
    let discovered = walker.walk(path)?;

    println!("{} {} files", "Discovered:".green(), discovered.len());
    println!();

    // Setup progress bar
    let pb = ProgressBar::new(discovered.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut indexed_count = 0;
    let mut skipped_count = 0;

    for disc_file in discovered {
        let filename = disc_file.path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        pb.set_message(filename.clone());

        let metadata = match metadata::extract_metadata(&disc_file.path, disc_file.file_type) {
            Ok(m) => m,
            Err(_) => {
                skipped_count += 1;
                pb.inc(1);
                continue;
            }
        };

        let file_id = db.upsert_file(&metadata).await?;

        // Handle images separately
        if disc_file.file_type == FileType::Image {
            // Try to generate image embedding
            if let Some(ref mut img_model) = image_embedding_model {
                match img_model.embed_image(&disc_file.path) {
                    Ok(embedding) => {
                        image_vector_store.upsert(file_id, &embedding)?;

                        // Add basic metadata to tantivy for filtering
                        tantivy_index.upsert_document(
                            file_id,
                            &disc_file.path.to_string_lossy(),
                            &metadata.filename,
                            &format!("image file: {}", metadata.filename),
                        )?;

                        if verbose {
                            pb.println(format!("  {} {} [image]", "✓".green(), filename));
                        }

                        indexed_count += 1;
                    }
                    Err(e) => {
                        if verbose {
                            pb.println(format!("  {} {} [image error: {}]", "✗".red(), filename, e));
                        }
                        skipped_count += 1;
                    }
                }
            } else {
                // Just index metadata without embedding
                tantivy_index.upsert_document(
                    file_id,
                    &disc_file.path.to_string_lossy(),
                    &metadata.filename,
                    &format!("image file: {}", metadata.filename),
                )?;
                indexed_count += 1;
            }
        } else {
            // Handle text/document files
            match text::extract_text(&disc_file.path, disc_file.file_type) {
                Ok(content) => {
                    db.upsert_content(file_id, &content).await?;

                    tantivy_index.upsert_document(
                        file_id,
                        &disc_file.path.to_string_lossy(),
                        &metadata.filename,
                        &content.text,
                    )?;

                    // Generate embedding if semantic search is enabled
                    if let Some(ref mut model) = embedding_model {
                        let text_chunk = if content.text.len() > 5000 {
                            &content.text[..5000]
                        } else {
                            &content.text
                        };

                        if let Ok(embedding) = model.embed(text_chunk) {
                            vector_store.upsert(file_id, &embedding)?;
                        }
                    }

                    if verbose {
                        pb.println(format!("  {} {}", "✓".green(), filename));
                    }

                    indexed_count += 1;
                }
                Err(_) => {
                    skipped_count += 1;
                }
            }
        }

        pb.inc(1);
    }

    pb.finish_with_message("Done!");
    println!();

    tantivy_index.commit()?;

    // Save vector stores if semantic search was enabled
    if embedding_model.is_some() {
        vector_store.save(&vector_path)?;
    }

    if image_embedding_model.is_some() && !image_vector_store.is_empty() {
        image_vector_store.save(&image_vector_path)?;
    }

    println!("{}", "Indexing complete!".green().bold());
    println!("  {} {} files indexed", "✓".green(), indexed_count);
    if skipped_count > 0 {
        println!("  {} {} files skipped", "⚠".yellow(), skipped_count);
    }
    if !image_vector_store.is_empty() {
        println!("  {} {} images with embeddings", "🖼️ ".cyan(), image_vector_store.len());
    }
    println!();

    Ok(())
}

async fn search_index(
    query: &str,
    index_dir: &PathBuf,
    limit: usize,
    use_semantic: bool,
    keyword_weight: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    use colored::Colorize;

    let db_path = index_dir.join("db.sqlite");
    let tantivy_path = index_dir.join("tantivy");
    let vector_path = index_dir.join("vectors.json");
    let image_vector_path = index_dir.join("image_vectors.json");

    if !tantivy_path.exists() {
        eprintln!("{}", "Error: No index found!".red().bold());
        eprintln!("Run: khoj index <folder>");
        std::process::exit(1);
    }

    let db = Database::new(&db_path).await?;
    let tantivy_index = TantivyIndex::new(&tantivy_path)?;

    let vector_store = if use_semantic && vector_path.exists() {
        VectorStore::load(&vector_path)?
    } else {
        VectorStore::new(384)?
    };

    let image_vector_store = if use_semantic && image_vector_path.exists() {
        VectorStore::load(&image_vector_path)?
    } else {
        VectorStore::new(512)?
    };

    let search_engine = HybridSearch::new(tantivy_index, vector_store);

    let results = if use_semantic {
        // Load embedding model
        let model_path = find_model_path("model.onnx").ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "model.onnx not found")
        })?;
        let tokenizer_path = find_model_path("tokenizer.json").ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "tokenizer.json not found")
        })?;

        let mut embedding_model = EmbeddingModel::new(&model_path, &tokenizer_path)?;

        let query_embedding = embedding_model.embed(query)?;
        search_engine.hybrid_search(query, Some(&query_embedding), limit, keyword_weight)?
    } else {
        search_engine.keyword_search(query, limit)?
    };

    // Also search images if image vectors are available
    let mut image_results = Vec::new();
    if use_semantic && !image_vector_store.is_empty() {
        if let (Some(clip_text_path), Some(clip_tokenizer_path)) =
            (find_model_path("clip_text.onnx"), find_model_path("clip_tokenizer.json")) {
            let mut clip_text_model = ClipTextEmbedding::new(&clip_text_path, &clip_tokenizer_path)?;
            let image_query_embedding = clip_text_model.embed_text(query)?;
            image_results = image_vector_store.search(&image_query_embedding, limit)?;
        }
    }

    println!();
    println!("{} \"{}\"", "Results for:".cyan().bold(), query);
    println!();

    // Display text/document results
    if !results.is_empty() {
        println!("{}", "Documents:".green().bold());
        for (i, result) in results.iter().enumerate() {
            println!("{}", format!("{}. {}", i + 1, result.filename).green());
            println!("   {}: {}", "Path".dimmed(), result.path);
            println!("   {}: {:.2}", "Score".dimmed(), result.score);

            // Get snippet from database
            if let Ok(Some(content)) = db.get_content(result.file_id).await {
                if let Some(snippet) = khoj::extractors::text::extract_snippet(&content.text, query, 100) {
                    let truncated = if snippet.len() > 150 {
                        format!("{}...", &snippet[..150])
                    } else {
                        snippet
                    };
                    println!("   {}: {}", "Preview".dimmed(), truncated);
                }
            }

            println!();
        }
    }

    // Display image results
    if !image_results.is_empty() {
        println!("{}", "Images:".cyan().bold());
        for (i, (file_id, score)) in image_results.iter().enumerate() {
            // Get file metadata from database
            if let Ok(Some(metadata)) = db.get_file(*file_id).await {
                println!("{}", format!("{}. {}", i + 1, metadata.filename).cyan());
                println!("   {}: {}", "Path".dimmed(), metadata.path);
                println!("   {}: {:.2}", "Similarity".dimmed(), score);
                println!();
            }
        }
    }

    if results.is_empty() && image_results.is_empty() {
        println!("{}", "No results found.".yellow());
    }

    Ok(())
}

async fn show_stats(index_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use colored::Colorize;

    let db_path = index_dir.join("db.sqlite");
    let tantivy_path = index_dir.join("tantivy");
    let vector_path = index_dir.join("vectors.json");

    if !db_path.exists() {
        println!("{}", "No index found.".yellow());
        return Ok(());
    }

    let db = Database::new(&db_path).await?;
    let stats = db.get_stats().await?;

    println!();
    println!("{}", "Index Statistics".cyan().bold());
    println!("─────────────────");
    println!("  {}: {}", "Total files".green(), stats.total_files);
    println!("  {}: {}", "Location".dimmed(), index_dir.display());
    println!();

    if tantivy_path.exists() {
        println!("  {} Keyword search index (Tantivy)", "✓".green());
    }

    if vector_path.exists() {
        println!("  {} Semantic search index (Vectors)", "✓".green());
    }
    println!();

    Ok(())
}

async fn list_files(index_dir: &PathBuf, limit: usize) -> Result<(), Box<dyn std::error::Error>> {
    use colored::Colorize;

    let db_path = index_dir.join("db.sqlite");

    if !db_path.exists() {
        println!("{}", "No index found.".yellow());
        return Ok(());
    }

    let db = Database::new(&db_path).await?;

    // This is a simplified version - you'd need to add a list method to Database
    println!();
    println!("{} (showing up to {})", "Indexed files:".cyan().bold(), limit);
    println!();

    // For now, just show stats
    let stats = db.get_stats().await?;
    println!("  {} {} total files indexed", "ℹ".cyan(), stats.total_files);
    println!();
    println!("  {} Use 'khoj \"query\"' to search", "💡".yellow());
    println!();

    Ok(())
}

fn clear_index(index_dir: &PathBuf, skip_confirm: bool) -> Result<(), Box<dyn std::error::Error>> {
    use colored::Colorize;

    if !index_dir.exists() {
        println!("{}", "No index found.".yellow());
        return Ok(());
    }

    if !skip_confirm {
        println!("{}", "This will delete the entire index.".yellow());
        print!("Are you sure? [y/N]: ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    std::fs::remove_dir_all(index_dir)?;
    println!("{}", "Index cleared!".green());

    Ok(())
}