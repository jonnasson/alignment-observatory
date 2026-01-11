//! Streaming Activation Capture Module
//!
//! This module provides memory-efficient streaming capture for large models:
//!
//! - Memory-mapped file storage for activations
//! - Ring buffer support for sliding window analysis
//! - Selective layer capture to reduce memory usage
//! - Lazy loading for efficient access patterns

use ndarray::{Array2, Array3, ArrayView3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Configuration for streaming capture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Directory for storing activation data
    pub storage_dir: PathBuf,
    /// Maximum memory usage in bytes (default 4GB)
    pub memory_limit_bytes: usize,
    /// Layers to capture (empty = all layers)
    pub capture_layers: Vec<usize>,
    /// Components to capture
    pub capture_components: Vec<String>,
    /// Whether to use memory-mapped files
    pub use_mmap: bool,
    /// Buffer size for ring buffer mode
    pub ring_buffer_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            storage_dir: PathBuf::from("/tmp/alignment_microscope"),
            memory_limit_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            capture_layers: Vec::new(),
            capture_components: vec![
                "residual".to_string(),
                "attn_out".to_string(),
                "mlp_out".to_string(),
            ],
            use_mmap: true,
            ring_buffer_size: 1000,
        }
    }
}

impl StreamingConfig {
    /// Create config for large models (70B+)
    pub fn for_large_model(storage_dir: PathBuf) -> Self {
        Self {
            storage_dir,
            memory_limit_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            capture_layers: Vec::new(), // Capture all
            capture_components: vec!["residual".to_string()], // Only residual
            use_mmap: true,
            ring_buffer_size: 100,
        }
    }

    /// Create config with selective layer capture
    pub fn selective(storage_dir: PathBuf, layers: Vec<usize>) -> Self {
        Self {
            storage_dir,
            capture_layers: layers,
            ..Default::default()
        }
    }
}

/// Metadata for a stored activation chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Layer index
    pub layer: usize,
    /// Component type
    pub component: String,
    /// Shape: [batch, seq_len, hidden_size]
    pub shape: Vec<usize>,
    /// Data type (f32, f16, etc.)
    pub dtype: String,
    /// Byte offset in the file
    pub offset: u64,
    /// Size in bytes
    pub size_bytes: usize,
    /// Token indices covered by this chunk
    pub token_range: (usize, usize),
}

/// Activation storage backend
pub struct ActivationStorage {
    /// Storage directory
    storage_dir: PathBuf,
    /// Chunk metadata index
    metadata: HashMap<String, Vec<ChunkMetadata>>,
    /// Current write offset per file
    offsets: HashMap<String, u64>,
    /// Open file handles for writing
    writers: HashMap<String, File>,
    /// Configuration
    config: StreamingConfig,
}

impl ActivationStorage {
    /// Create new storage backend
    pub fn new(config: StreamingConfig) -> io::Result<Self> {
        // Create storage directory if it doesn't exist
        fs::create_dir_all(&config.storage_dir)?;

        Ok(Self {
            storage_dir: config.storage_dir.clone(),
            metadata: HashMap::new(),
            offsets: HashMap::new(),
            writers: HashMap::new(),
            config,
        })
    }

    /// Get file path for a layer/component
    fn file_path(&self, layer: usize, component: &str) -> PathBuf {
        self.storage_dir.join(format!("layer_{layer}_{component}.bin"))
    }

    /// Get key for layer/component
    fn key(layer: usize, component: &str) -> String {
        format!("{layer}_{component}")
    }

    /// Store an activation chunk
    pub fn store(
        &mut self,
        layer: usize,
        component: &str,
        data: ArrayView3<f32>,
        token_range: (usize, usize),
    ) -> io::Result<()> {
        // Check if we should capture this layer/component
        if !self.config.capture_layers.is_empty()
            && !self.config.capture_layers.contains(&layer)
        {
            return Ok(());
        }

        if !self.config.capture_components.contains(&component.to_string()) {
            return Ok(());
        }

        let key = Self::key(layer, component);
        let path = self.file_path(layer, component);

        // Get or create writer (lazily create file on first access)
        if !self.writers.contains_key(&key) {
            let file = File::create(&path)?;
            self.writers.insert(key.clone(), file);
        }
        let writer = self.writers.get_mut(&key).ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "Failed to get writer")
        })?;

        // Get current offset
        let offset = *self.offsets.entry(key.clone()).or_insert(0);

        // Write data - handle non-contiguous arrays by iterating
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer.write_all(&bytes)?;

        // Create metadata
        let metadata = ChunkMetadata {
            layer,
            component: component.to_string(),
            shape: data.shape().to_vec(),
            dtype: "f32".to_string(),
            offset,
            size_bytes: bytes.len(),
            token_range,
        };

        // Update index
        self.metadata
            .entry(key.clone())
            .or_default()
            .push(metadata);

        // Update offset (key is guaranteed to exist from earlier insert)
        if let Some(off) = self.offsets.get_mut(&key) {
            *off += bytes.len() as u64;
        }

        Ok(())
    }

    /// Load an activation chunk
    pub fn load(
        &self,
        layer: usize,
        component: &str,
        chunk_idx: usize,
    ) -> io::Result<Array3<f32>> {
        let key = Self::key(layer, component);
        let path = self.file_path(layer, component);

        let chunks = self.metadata.get(&key).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "No data for layer/component")
        })?;

        let meta = chunks.get(chunk_idx).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "Chunk index out of range")
        })?;

        let mut file = File::open(&path)?;
        file.seek(SeekFrom::Start(meta.offset))?;

        let mut bytes = vec![0u8; meta.size_bytes];
        file.read_exact(&mut bytes)?;

        // Convert bytes to f32
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let shape = (meta.shape[0], meta.shape[1], meta.shape[2]);
        Array3::from_shape_vec(shape, floats)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Get all chunk metadata for a layer/component
    pub fn get_metadata(&self, layer: usize, component: &str) -> Option<&Vec<ChunkMetadata>> {
        let key = Self::key(layer, component);
        self.metadata.get(&key)
    }

    /// Get total stored size in bytes
    pub fn total_size_bytes(&self) -> usize {
        self.metadata
            .values()
            .flat_map(|chunks| chunks.iter())
            .map(|m| m.size_bytes)
            .sum()
    }

    /// List available layers
    pub fn available_layers(&self) -> Vec<usize> {
        let mut layers: Vec<usize> = self
            .metadata
            .keys()
            .filter_map(|key| key.split('_').next()?.parse().ok())
            .collect();
        layers.sort();
        layers.dedup();
        layers
    }

    /// Clean up storage
    pub fn cleanup(&mut self) -> io::Result<()> {
        // Close all writers
        self.writers.clear();

        // Remove all files
        for entry in fs::read_dir(&self.storage_dir)? {
            let entry = entry?;
            if entry.path().extension().map_or(false, |ext| ext == "bin") {
                fs::remove_file(entry.path())?;
            }
        }

        self.metadata.clear();
        self.offsets.clear();

        Ok(())
    }

    /// Flush all writers
    pub fn flush(&mut self) -> io::Result<()> {
        for writer in self.writers.values_mut() {
            writer.flush()?;
        }
        Ok(())
    }
}

/// Ring buffer for streaming activation analysis
pub struct ActivationRingBuffer {
    /// Buffer storage
    buffer: Vec<Option<Array3<f32>>>,
    /// Current write position
    write_pos: usize,
    /// Number of items in buffer
    count: usize,
    /// Layer this buffer is for
    layer: usize,
    /// Component this buffer is for
    component: String,
}

impl ActivationRingBuffer {
    /// Create new ring buffer
    pub fn new(capacity: usize, layer: usize, component: String) -> Self {
        Self {
            buffer: (0..capacity).map(|_| None).collect(),
            write_pos: 0,
            count: 0,
            layer,
            component,
        }
    }

    /// Push an activation to the buffer
    pub fn push(&mut self, activation: Array3<f32>) {
        self.buffer[self.write_pos] = Some(activation);
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        if self.count < self.buffer.len() {
            self.count += 1;
        }
    }

    /// Get the most recent n activations
    pub fn recent(&self, n: usize) -> Vec<&Array3<f32>> {
        let n = n.min(self.count);
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let idx = if self.write_pos >= i + 1 {
                self.write_pos - i - 1
            } else {
                self.buffer.len() - (i + 1 - self.write_pos)
            };

            if let Some(ref act) = self.buffer[idx] {
                result.push(act);
            }
        }

        result
    }

    /// Get all activations in order (oldest first)
    pub fn all(&self) -> Vec<&Array3<f32>> {
        let mut result = Vec::with_capacity(self.count);

        let start = if self.count < self.buffer.len() {
            0
        } else {
            self.write_pos
        };

        for i in 0..self.count {
            let idx = (start + i) % self.buffer.len();
            if let Some(ref act) = self.buffer[idx] {
                result.push(act);
            }
        }

        result
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Get current count
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        for slot in &mut self.buffer {
            *slot = None;
        }
        self.write_pos = 0;
        self.count = 0;
    }
}

/// Memory estimator for activation capture
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Estimate memory for capturing all activations
    pub fn estimate_full_capture(
        num_layers: usize,
        hidden_size: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> usize {
        // Each layer has: residual, attn_out, mlp_out
        // Each is: batch * seq * hidden * 4 bytes (f32)
        let per_component = batch_size * seq_len * hidden_size * 4;
        let per_layer = per_component * 3; // 3 components
        num_layers * per_layer
    }

    /// Suggest capture strategy based on model size
    pub fn suggest_strategy(
        num_layers: usize,
        hidden_size: usize,
        memory_limit_bytes: usize,
    ) -> CaptureStrategy {
        // Estimate memory for batch=1, seq=1024
        let full_mem = Self::estimate_full_capture(num_layers, hidden_size, 1, 1024);

        if full_mem < memory_limit_bytes {
            CaptureStrategy::InMemory
        } else if full_mem < memory_limit_bytes * 4 {
            CaptureStrategy::SelectiveLayers(Self::key_layers(num_layers))
        } else {
            CaptureStrategy::Streaming
        }
    }

    /// Get key layers to capture for a model
    fn key_layers(num_layers: usize) -> Vec<usize> {
        // Capture first, middle, and last few layers
        let mut layers = vec![0, 1];

        let mid = num_layers / 2;
        layers.extend([mid - 1, mid, mid + 1]);

        layers.extend([num_layers - 2, num_layers - 1]);

        layers.into_iter().filter(|&l| l < num_layers).collect()
    }
}

/// Capture strategy recommendation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureStrategy {
    /// All activations fit in memory
    InMemory,
    /// Only capture specific layers
    SelectiveLayers(Vec<usize>),
    /// Use streaming with disk storage
    Streaming,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use tempfile::tempdir;

    #[test]
    fn test_storage_creation() {
        let dir = tempdir().unwrap();
        let config = StreamingConfig {
            storage_dir: dir.path().to_path_buf(),
            ..Default::default()
        };

        let storage = ActivationStorage::new(config).unwrap();
        assert!(storage.available_layers().is_empty());
    }

    #[test]
    fn test_store_and_load() {
        let dir = tempdir().unwrap();
        let config = StreamingConfig {
            storage_dir: dir.path().to_path_buf(),
            capture_layers: vec![0],
            capture_components: vec!["residual".to_string()],
            ..Default::default()
        };

        let mut storage = ActivationStorage::new(config).unwrap();

        // Store activation
        let data = Array3::from_shape_fn((1, 10, 64), |(b, s, h)| {
            (b + s + h) as f32
        });
        storage.store(0, "residual", data.view(), (0, 10)).unwrap();
        storage.flush().unwrap();

        // Load activation
        let loaded = storage.load(0, "residual", 0).unwrap();
        assert_eq!(loaded.shape(), data.shape());
        assert_eq!(loaded[[0, 0, 0]], data[[0, 0, 0]]);
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = ActivationRingBuffer::new(3, 0, "residual".to_string());

        // Push 5 items into buffer of size 3
        for i in 0..5 {
            let data = Array3::from_elem((1, 4, 4), i as f32);
            buffer.push(data);
        }

        // Should have 3 items
        assert_eq!(buffer.len(), 3);

        // Most recent should be 4, 3, 2
        let recent = buffer.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0][[0, 0, 0]], 4.0);
        assert_eq!(recent[1][[0, 0, 0]], 3.0);
        assert_eq!(recent[2][[0, 0, 0]], 2.0);
    }

    #[test]
    fn test_memory_estimator() {
        // Small model should fit in memory
        let strategy = MemoryEstimator::suggest_strategy(
            12, // layers
            768, // hidden
            4 * 1024 * 1024 * 1024, // 4GB
        );
        assert_eq!(strategy, CaptureStrategy::InMemory);

        // Large model needs streaming
        let strategy = MemoryEstimator::suggest_strategy(
            80, // layers (70B model)
            8192, // hidden
            4 * 1024 * 1024 * 1024, // 4GB
        );
        assert_eq!(strategy, CaptureStrategy::Streaming);
    }

    #[test]
    fn test_selective_config() {
        let dir = tempdir().unwrap();
        let config = StreamingConfig::selective(
            dir.path().to_path_buf(),
            vec![0, 5, 10],
        );

        assert_eq!(config.capture_layers, vec![0, 5, 10]);
    }
}
