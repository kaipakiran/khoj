//! Image embedding using CLIP for visual search
//! Enables searching images by content: "garden", "car", "family", etc.

use crate::Result;
use crate::embedding::Tokenizer;
use image::DynamicImage;
use ndarray::{Array3, Array4, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::path::Path;

/// CLIP image encoder for visual search
pub struct ImageEmbedding {
    session: Session,
    image_size: u32,
}

impl ImageEmbedding {
    /// Create a new image embedding model
    ///
    /// Expects CLIP ViT-B/32 vision model in ONNX format
    pub fn new(model_path: &Path) -> Result<Self> {
        // Load ONNX model file
        let model_bytes = std::fs::read(model_path)?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_memory(&model_bytes)?;

        Ok(Self {
            session,
            image_size: 224, // CLIP standard input size
        })
    }

    /// Generate embedding for an image file
    pub fn embed_image(&mut self, image_path: &Path) -> Result<Vec<f32>> {
        // Load and preprocess image
        let img = image::open(image_path)
            .map_err(|e| crate::Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        self.embed_dynamic_image(&img)
    }

    /// Generate embedding from a DynamicImage
    pub fn embed_dynamic_image(&mut self, img: &DynamicImage) -> Result<Vec<f32>> {
        // Preprocess: resize, normalize
        let preprocessed = self.preprocess_image(img)?;

        // Convert ndarray to ort Value
        let shape = preprocessed.shape().to_vec();
        let data: Vec<f32> = preprocessed.iter().copied().collect();
        let input_value = Value::from_array((shape, data))?;

        // Run inference
        // CLIP vision models typically use "pixel_values" as input name
        let outputs = self.session.run(ort::inputs!["pixel_values" => input_value])?;

        // Extract embeddings from output
        let (_emb_shape, emb_data) = outputs[0].try_extract_tensor::<f32>()?;

        // Convert to Vec (typically CLIP image encoder outputs 512-dim)
        let embedding_vec: Vec<f32> = emb_data.iter().copied().collect();

        // Drop outputs to release mutable borrow before calling normalize
        drop(outputs);

        // Normalize embeddings for cosine similarity
        Ok(self.normalize(&embedding_vec))
    }

    /// Preprocess image for CLIP:
    /// 1. Resize to 224x224
    /// 2. Convert to RGB
    /// 3. Normalize with ImageNet stats
    /// 4. Convert to CHW format (channels, height, width)
    fn preprocess_image(&self, img: &DynamicImage) -> Result<Array4<f32>> {
        // Resize to model input size
        let img = img.resize_exact(
            self.image_size,
            self.image_size,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to RGB
        let rgb_img = img.to_rgb8();

        // ImageNet normalization parameters (used by CLIP)
        let mean = [0.48145466, 0.4578275, 0.40821073];
        let std = [0.26862954, 0.26130258, 0.27577711];

        // Convert to ndarray and normalize
        let mut array = Array3::<f32>::zeros((3, self.image_size as usize, self.image_size as usize));

        for (x, y, pixel) in rgb_img.enumerate_pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            array[[0, y as usize, x as usize]] = (r - mean[0]) / std[0];
            array[[1, y as usize, x as usize]] = (g - mean[1]) / std[1];
            array[[2, y as usize, x as usize]] = (b - mean[2]) / std[2];
        }

        // Add batch dimension: (1, 3, 224, 224)
        Ok(array.insert_axis(Axis(0)))
    }

    /// Normalize embedding vector to unit length for cosine similarity
    fn normalize(&self, embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding.to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use std::path::PathBuf;

    #[test]
    #[ignore] // Only run when model is available
    fn test_image_embedding() {
        let model_path = PathBuf::from("models/clip_vision.onnx");
        if !model_path.exists() {
            return;
        }

        let mut embedder = ImageEmbedding::new(&model_path).unwrap();

        // Create a test image
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(224, 224, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        }));

        let embedding = embedder.embed_dynamic_image(&img).unwrap();

        // CLIP embeddings are typically 512-dimensional
        assert_eq!(embedding.len(), 512);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}

/// CLIP text encoder for text-to-image search
/// Enables searching images with text queries like "garden", "sunset", "car"
pub struct ClipTextEmbedding {
    session: Session,
    tokenizer: Tokenizer,
}

impl ClipTextEmbedding {
    /// Create a new CLIP text embedding model
    ///
    /// Expects CLIP ViT-B/32 text model in ONNX format
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        // Load ONNX model file
        let model_bytes = std::fs::read(model_path)?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_memory(&model_bytes)?;

        // Load CLIP tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        Ok(Self { session, tokenizer })
    }

    /// Generate embedding for a text query
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        // Tokenize input text (CLIP uses max length 77)
        let tokens = self.tokenizer.encode(text, 77)?;

        // Prepare input tensors
        let seq_len = tokens.input_ids.len();
        let shape = vec![1, seq_len];

        let input_ids_value = Value::from_array((shape.clone(), tokens.input_ids.clone()))?;
        let attention_mask_value = Value::from_array((shape.clone(), tokens.attention_mask.clone()))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_value,
            "attention_mask" => attention_mask_value,
        ])?;

        // Extract embeddings from output
        let (_emb_shape, emb_data) = outputs[0].try_extract_tensor::<f32>()?;

        // Convert to Vec (CLIP text encoder outputs 512-dim)
        let embedding_vec: Vec<f32> = emb_data.iter().copied().collect();

        // Drop outputs to release mutable borrow
        drop(outputs);

        // Normalize embeddings for cosine similarity
        Ok(self.normalize(&embedding_vec))
    }

    /// Normalize embedding vector to unit length for cosine similarity
    fn normalize(&self, embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding.to_vec()
        }
    }
}