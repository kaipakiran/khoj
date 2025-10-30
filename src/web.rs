//! Web server for khoj - search from browser on any device

use axum::{
    extract::{Path as AxumPath, Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Json, Response},
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::{
    embedding::{EmbeddingModel, image::ClipTextEmbedding},
    search::HybridSearch,
    storage::{Database, TantivyIndex, VectorStore},
};

#[derive(Clone)]
pub struct AppState {
    pub index_dir: PathBuf,
}

#[derive(Deserialize)]
pub struct SearchParams {
    q: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    semantic: bool,
    #[serde(default = "default_keyword_weight")]
    keyword_weight: f32,
}

fn default_limit() -> usize {
    10
}

fn default_keyword_weight() -> f32 {
    0.7
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub documents: Vec<SearchResult>,
    pub images: Vec<SearchResult>,
    pub took_ms: u64,
}

#[derive(Serialize, Clone)]
pub struct SearchResult {
    pub file_id: i64,
    pub filename: String,
    pub path: String,
    pub score: f32,
    pub snippet: Option<String>,
    pub file_type: String,
}

#[derive(Serialize)]
pub struct StatsResponse {
    pub total_files: i64,
    pub index_location: String,
    pub has_keyword_index: bool,
    pub has_semantic_index: bool,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Start the web server
pub async fn serve(index_dir: PathBuf, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState {
        index_dir: index_dir.clone(),
    };

    let app = Router::new()
        .route("/", get(serve_index))
        .route("/api/search", get(handle_search))
        .route("/api/stats", get(handle_stats))
        .route("/api/file/:file_id", get(handle_file))
        .layer(CorsLayer::permissive())
        .with_state(Arc::new(state));

    let addr = format!("0.0.0.0:{}", port);
    println!("\n🌐 khoj web server starting...");
    println!("   Local:    http://localhost:{}", port);
    println!("   Network:  http://<your-ip>:{}", port);
    println!("\n💡 Open in browser to search!");
    println!("   Press Ctrl+C to stop\n");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Serve the HTML interface
async fn serve_index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

/// Handle search requests
async fn handle_search(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();

    // Check if index exists
    let tantivy_path = state.index_dir.join("tantivy");
    if !tantivy_path.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "No index found. Run 'khoj index <folder>' first.".to_string(),
            })
            .into_response(),
        )
            .into_response();
    }

    // Initialize search components
    let db_path = state.index_dir.join("db.sqlite");
    let vector_path = state.index_dir.join("vectors.json");

    let db = match Database::new(&db_path).await {
        Ok(db) => db,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Database error: {}", e),
                })
                .into_response(),
            )
                .into_response()
        }
    };

    let tantivy_index = match TantivyIndex::new(&tantivy_path) {
        Ok(idx) => idx,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Index error: {}", e),
                })
                .into_response(),
            )
                .into_response()
        }
    };

    let vector_store = if params.semantic && vector_path.exists() {
        match VectorStore::load(&vector_path) {
            Ok(vs) => vs,
            Err(_) => VectorStore::new(384).unwrap(),
        }
    } else {
        VectorStore::new(384).unwrap()
    };

    let search_engine = HybridSearch::new(tantivy_index, vector_store);

    // Perform search
    let results = if params.semantic {
        // Load embedding model
        let model_path = PathBuf::from("models/model.onnx");
        let tokenizer_path = PathBuf::from("models/tokenizer.json");

        if !model_path.exists() {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Semantic search requires ONNX model. Index with --semantic first."
                        .to_string(),
                })
                .into_response(),
            )
                .into_response();
        }

        let mut embedding_model = match EmbeddingModel::new(&model_path, &tokenizer_path) {
            Ok(model) => model,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Failed to load embedding model: {}", e),
                    })
                    .into_response(),
                )
                    .into_response()
            }
        };

        let query_embedding = match embedding_model.embed(&params.q) {
            Ok(emb) => emb,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Failed to generate query embedding: {}", e),
                    })
                    .into_response(),
                )
                    .into_response()
            }
        };

        match search_engine.hybrid_search(
            &params.q,
            Some(&query_embedding),
            params.limit,
            params.keyword_weight,
        ) {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Search error: {}", e),
                    })
                    .into_response(),
                )
                    .into_response()
            }
        }
    } else {
        match search_engine.keyword_search(&params.q, params.limit) {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Search error: {}", e),
                    })
                    .into_response(),
                )
                    .into_response()
            }
        }
    };

    // Also search images if semantic search is enabled
    let image_vector_path = state.index_dir.join("image_vectors.json");
    let mut image_results = Vec::new();

    if params.semantic && image_vector_path.exists() {
        // Load image vector store
        let image_vector_store = match VectorStore::load(&image_vector_path) {
            Ok(vs) => vs,
            Err(_) => VectorStore::new(512).unwrap(),
        };

        if !image_vector_store.is_empty() {
            // Load CLIP text model for text-to-image search
            let clip_text_path = PathBuf::from("models/clip_text.onnx");
            let clip_tokenizer_path = PathBuf::from("models/clip_tokenizer.json");

            if clip_text_path.exists() && clip_tokenizer_path.exists() {
                if let Ok(mut clip_model) = ClipTextEmbedding::new(&clip_text_path, &clip_tokenizer_path) {
                    if let Ok(image_embedding) = clip_model.embed_text(&params.q) {
                        image_results = image_vector_store.search(&image_embedding, params.limit).unwrap_or_default();
                    }
                }
            }
        }
    }

    // Get snippets and file info for results
    let mut search_results = Vec::new();
    for result in results {
        // Get file metadata to determine type
        let file_metadata = db.get_file(result.file_id).await.ok().flatten();
        let file_type = file_metadata
            .as_ref()
            .map(|m| m.file_type.as_str().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let snippet = if let Ok(Some(content)) = db.get_content(result.file_id).await {
            crate::extractors::text::extract_snippet(&content.text, &params.q, 100)
        } else {
            None
        };

        search_results.push(SearchResult {
            file_id: result.file_id,
            filename: result.filename,
            path: result.path,
            score: result.score,
            snippet,
            file_type,
        });
    }

    // Add image results
    for (file_id, similarity) in image_results {
        if let Ok(Some(metadata)) = db.get_file(file_id).await {
            search_results.push(SearchResult {
                file_id,
                filename: metadata.filename,
                path: metadata.path,
                score: similarity,
                snippet: None,
                file_type: "image".to_string(),
            });
        }
    }

    // Separate results into documents and images, sorted by score (highest first)
    let mut documents: Vec<SearchResult> = search_results
        .iter()
        .filter(|r| r.file_type != "image")
        .cloned()
        .collect();

    let mut images: Vec<SearchResult> = search_results
        .iter()
        .filter(|r| r.file_type == "image")
        .cloned()
        .collect();

    // Sort by score descending (highest similarity/relevance first)
    documents.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    images.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let took_ms = start.elapsed().as_millis() as u64;

    (
        StatusCode::OK,
        Json(SearchResponse {
            query: params.q,
            documents,
            images,
            took_ms,
        })
        .into_response(),
    )
        .into_response()
}

/// Handle stats requests
async fn handle_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db_path = state.index_dir.join("db.sqlite");
    let tantivy_path = state.index_dir.join("tantivy");
    let vector_path = state.index_dir.join("vectors.json");

    if !db_path.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "No index found".to_string(),
            })
            .into_response(),
        )
            .into_response();
    }

    let db = match Database::new(&db_path).await {
        Ok(db) => db,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Database error: {}", e),
                })
                .into_response(),
            )
                .into_response()
        }
    };

    let stats = match db.get_stats().await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to get stats: {}", e),
                })
                .into_response(),
            )
                .into_response()
        }
    };

    (
        StatusCode::OK,
        Json(StatsResponse {
            total_files: stats.total_files,
            index_location: state.index_dir.display().to_string(),
            has_keyword_index: tantivy_path.exists(),
            has_semantic_index: vector_path.exists(),
        })
        .into_response(),
    )
        .into_response()
}

/// Serve a file by ID
async fn handle_file(
    State(state): State<Arc<AppState>>,
    AxumPath(file_id): AxumPath<i64>,
) -> impl IntoResponse {
    let db_path = state.index_dir.join("db.sqlite");

    let db = match Database::new(&db_path).await {
        Ok(db) => db,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Database error: {}", e),
                })
                .into_response(),
            )
                .into_response()
        }
    };

    // Get file metadata
    let file_metadata = match db.get_file(file_id).await {
        Ok(Some(metadata)) => metadata,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "File not found".to_string(),
                })
                .into_response(),
            )
                .into_response()
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Database error: {}", e),
                })
                .into_response(),
            )
                .into_response()
        }
    };

    // Read the file
    let file_path = std::path::Path::new(&file_metadata.path);
    let file_bytes = match tokio::fs::read(file_path).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("Failed to read file: {}", e),
                })
                .into_response(),
            )
                .into_response()
        }
    };

    // Determine content type from file extension
    let content_type = file_metadata
        .mime_type
        .unwrap_or_else(|| "application/octet-stream".to_string());

    // Return file with appropriate headers
    use axum::body::Body;

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(
            header::CONTENT_DISPOSITION,
            format!("inline; filename=\"{}\"", file_metadata.filename),
        )
        .body(Body::from(file_bytes))
        .unwrap()
        .into_response()
}