//! Text file content extractor

use crate::types::FileType;
use crate::Result;
use std::fs;
use std::path::Path;

/// Extracted text content from a file
#[derive(Debug, Clone)]
pub struct ExtractedContent {
    pub text: String,
    pub word_count: usize,
    pub language: Option<String>,
}

/// Extract text content from a file
///
/// # Arguments
/// * `path` - Path to the file
/// * `file_type` - Type of the file
///
/// # Returns
/// Extracted text content with metadata
pub fn extract_text(path: &Path, file_type: FileType) -> Result<ExtractedContent> {
    let content = match file_type {
        FileType::Pdf => extract_pdf(path)?,
        FileType::Docx => extract_docx(path)?,
        FileType::Image => {
            // For now, we just store the filename for images
            // Later we can add OCR or image embedding
            return Err(crate::Error::UnsupportedFileType(
                "Image text extraction not yet implemented (OCR planned)".to_string()
            ));
        }
        FileType::Text | FileType::Code | FileType::Markdown | FileType::Unknown => {
            // Read file content as UTF-8
            fs::read_to_string(path)?
        }
        FileType::Xlsx => {
            return Err(crate::Error::UnsupportedFileType(
                "Excel extraction not yet implemented".to_string()
            ));
        }
        FileType::Archive => {
            return Err(crate::Error::UnsupportedFileType(
                "Archive extraction not supported".to_string()
            ));
        }
    };

    // Count words (simple whitespace-based counting)
    let word_count = content.split_whitespace().count();

    // Detect language based on file type
    let language = detect_language(path, file_type);

    Ok(ExtractedContent {
        text: content,
        word_count,
        language,
    })
}

/// Extract text from PDF files
fn extract_pdf(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;

    // Try to extract text, but handle errors AND panics gracefully
    // Some PDFs may be corrupted, encrypted, or have complex structures
    // The pdf-extract library can panic on malformed PDFs
    let result = std::panic::catch_unwind(|| {
        pdf_extract::extract_text_from_mem(&bytes)
    });

    match result {
        Ok(Ok(text)) => Ok(text),
        Ok(Err(e)) => {
            // PDF extraction error
            Err(crate::Error::Extraction(format!(
                "PDF extraction failed: {}",
                e
            )))
        }
        Err(_) => {
            // PDF extraction panicked
            Err(crate::Error::Extraction(
                "PDF extraction failed (file may be corrupted or use unsupported features)".to_string()
            ))
        }
    }
}

/// Extract text from DOCX files
fn extract_docx(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    let docx = docx_rs::read_docx(&bytes)
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("DOCX extraction failed: {}", e)))?;

    // Extract all text from paragraphs
    let mut text = String::new();
    for child in docx.document.children {
        if let docx_rs::DocumentChild::Paragraph(para) = child {
            for child in para.children {
                if let docx_rs::ParagraphChild::Run(run) = child {
                    for child in run.children {
                        if let docx_rs::RunChild::Text(t) = child {
                            text.push_str(&t.text);
                            text.push(' ');
                        }
                    }
                }
            }
            text.push('\n');
        }
    }

    Ok(text)
}

/// Detect programming language from file extension
fn detect_language(path: &Path, file_type: FileType) -> Option<String> {
    if file_type != FileType::Code {
        return None;
    }

    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| match ext.to_lowercase().as_str() {
            "rs" => "rust",
            "py" => "python",
            "js" => "javascript",
            "ts" => "typescript",
            "java" => "java",
            "c" => "c",
            "cpp" | "cc" | "cxx" => "cpp",
            "go" => "go",
            "rb" => "ruby",
            "php" => "php",
            "cs" => "csharp",
            "swift" => "swift",
            "kt" => "kotlin",
            "scala" => "scala",
            "sh" | "bash" => "shell",
            _ => "unknown",
        })
        .map(String::from)
}

/// Extract a snippet from text around a search term
///
/// # Arguments
/// * `text` - Full text content
/// * `query` - Search term
/// * `context_chars` - Number of characters to include before/after match
///
/// # Returns
/// A snippet of text with context around the match
pub fn extract_snippet(text: &str, query: &str, context_chars: usize) -> Option<String> {
    let query_lower = query.to_lowercase();
    let text_lower = text.to_lowercase();

    if let Some(pos) = text_lower.find(&query_lower) {
        let start = pos.saturating_sub(context_chars);
        let end = (pos + query.len() + context_chars).min(text.len());

        let snippet = &text[start..end];
        let prefix = if start > 0 { "..." } else { "" };
        let suffix = if end < text.len() { "..." } else { "" };

        Some(format!("{}{}{}", prefix, snippet, suffix))
    } else {
        // If no exact match, return the first N characters
        if text.len() > context_chars * 2 {
            Some(format!("{}...", &text[..context_chars * 2]))
        } else {
            Some(text.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_extract_text_from_txt_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        let content = "Hello world!\nThis is a test file.\n";
        fs::write(&file_path, content).unwrap();

        let extracted = extract_text(&file_path, FileType::Text).unwrap();

        assert_eq!(extracted.text, content);
        assert_eq!(extracted.word_count, 7);
        assert_eq!(extracted.language, None);
    }

    #[test]
    fn test_extract_text_from_markdown() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.md");

        let content = "# Title\n\nThis is **bold** text.\n";
        fs::write(&file_path, content).unwrap();

        let extracted = extract_text(&file_path, FileType::Markdown).unwrap();

        assert_eq!(extracted.text, content);
        assert!(extracted.word_count > 0);
        assert_eq!(extracted.language, None);
    }

    #[test]
    fn test_extract_text_from_code() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.rs");

        let content = "fn main() {\n    println!(\"Hello, world!\");\n}\n";
        fs::write(&file_path, content).unwrap();

        let extracted = extract_text(&file_path, FileType::Code).unwrap();

        assert_eq!(extracted.text, content);
        assert!(extracted.word_count > 0);
        assert_eq!(extracted.language, Some("rust".to_string()));
    }

    #[test]
    fn test_detect_language() {
        let test_cases = vec![
            ("test.rs", FileType::Code, Some("rust")),
            ("test.py", FileType::Code, Some("python")),
            ("test.js", FileType::Code, Some("javascript")),
            ("test.ts", FileType::Code, Some("typescript")),
            ("test.java", FileType::Code, Some("java")),
            ("test.go", FileType::Code, Some("go")),
            ("test.cpp", FileType::Code, Some("cpp")),
            ("test.txt", FileType::Text, None),
        ];

        for (filename, file_type, expected) in test_cases {
            let path = Path::new(filename);
            let lang = detect_language(path, file_type);
            assert_eq!(
                lang.as_deref(),
                expected,
                "Failed for {}",
                filename
            );
        }
    }

    #[test]
    fn test_extract_snippet_with_match() {
        let text = "This is a test file with some content. We want to find the word test.";
        let snippet = extract_snippet(text, "test", 20).unwrap();

        assert!(snippet.contains("test"));
        assert!(snippet.len() <= text.len());
    }

    #[test]
    fn test_extract_snippet_no_match() {
        let text = "This is some content without the search term.";
        let snippet = extract_snippet(text, "nonexistent", 20).unwrap();

        // Should return beginning of text
        assert!(snippet.starts_with("This is"));
    }

    #[test]
    fn test_extract_snippet_short_text() {
        let text = "Short text";
        let snippet = extract_snippet(text, "query", 50).unwrap();

        assert_eq!(snippet, text);
    }

    #[test]
    fn test_word_count() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        fs::write(&file_path, "one two three four five").unwrap();
        let extracted = extract_text(&file_path, FileType::Text).unwrap();
        assert_eq!(extracted.word_count, 5);

        fs::write(&file_path, "one\ntwo\nthree").unwrap();
        let extracted = extract_text(&file_path, FileType::Text).unwrap();
        assert_eq!(extracted.word_count, 3);
    }

    #[test]
    fn test_extract_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("empty.txt");

        fs::write(&file_path, "").unwrap();
        let extracted = extract_text(&file_path, FileType::Text).unwrap();

        assert_eq!(extracted.text, "");
        assert_eq!(extracted.word_count, 0);
    }

    #[test]
    fn test_extract_nonexistent_file() {
        let result = extract_text(Path::new("/nonexistent/file.txt"), FileType::Text);
        assert!(result.is_err());
    }
}