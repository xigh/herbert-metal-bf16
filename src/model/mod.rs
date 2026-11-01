//! The checkpoint, read in place: Gemma-4 (E2B, E4B) and Qwen3.5, text only.

use crate::core::error::{HerbertError, Result};
use std::path::{Path, PathBuf};

pub mod gemma;
pub mod qwen;

/// A Hugging Face cache directory points at a snapshot; a plain directory is
/// used as is.
pub fn resolve_model_dir(model_dir: &Path) -> Result<PathBuf> {
    if model_dir.join("config.json").exists() {
        return Ok(model_dir.to_path_buf());
    }
    let snapshots = model_dir.join("snapshots");
    if !snapshots.is_dir() {
        return Ok(model_dir.to_path_buf());
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&snapshots)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    candidates.sort();
    candidates.pop().ok_or_else(|| HerbertError::ModelLoad(format!("no snapshot under {}", snapshots.display())))
}
