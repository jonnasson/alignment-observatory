//! Alignment Microscope - High-performance interpretability engine for transformer models
//!
//! This library provides tools for understanding what happens inside neural networks,
//! with a focus on AI alignment research. It enables:
//!
//! - **Activation Tracing**: Track how information flows through model layers
//! - **Attention Analysis**: Visualize and analyze attention patterns
//! - **Circuit Discovery**: Identify computational circuits automatically
//! - **Causal Intervention**: Understand which components cause specific outputs
//!
//! # Architecture
//!
//! The engine is designed for:
//! - Zero-copy analysis where possible
//! - Streaming support for real-time interpretation
//! - Memory efficiency for large models (70B+ parameters)
//! - Parallel computation across layers/heads

pub mod activation;
pub mod attention;
pub mod circuit;
pub mod hooks;
pub mod intervention;
pub mod python;
pub mod sae;
pub mod streaming;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use thiserror::Error;

/// Core error types for the interpretability engine
#[derive(Error, Debug)]
pub enum MicroscopeError {
    #[error("Invalid tensor shape: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    #[error("Invalid shape: expected {expected}, got {got}")]
    InvalidShape { expected: String, got: String },

    #[error("Layer {layer} not found in model with {total} layers")]
    LayerNotFound { layer: usize, total: usize },

    #[error("Hook '{name}' is not registered")]
    HookNotFound { name: String },

    #[error("Model architecture '{arch}' is not supported")]
    UnsupportedArchitecture { arch: String },

    #[error("Numerical error: {message}")]
    NumericalError { message: String },

    #[error("Lock poisoned: {context}")]
    LockPoisoned { context: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, MicroscopeError>;

/// Configuration for the interpretability engine
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MicroscopeConfig {
    /// Model architecture (llama, mistral, gpt2, etc.)
    pub architecture: String,
    /// Number of layers in the model
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Whether to use memory mapping for large tensors
    pub use_mmap: bool,
    /// Maximum batch size for parallel processing
    pub max_batch_size: usize,
}

impl Default for MicroscopeConfig {
    fn default() -> Self {
        Self {
            architecture: "llama".to_string(),
            num_layers: 32,
            num_heads: 32,
            hidden_size: 4096,
            use_mmap: true,
            max_batch_size: 32,
        }
    }
}

/// Main entry point for the interpretability engine
#[derive(Debug)]
pub struct Microscope {
    config: MicroscopeConfig,
    hooks: hooks::HookRegistry,
    tracer: activation::ActivationTracer,
}

impl Microscope {
    /// Create a new Microscope instance with the given configuration
    pub fn new(config: MicroscopeConfig) -> Self {
        Self {
            tracer: activation::ActivationTracer::new(config.num_layers),
            hooks: hooks::HookRegistry::new(),
            config,
        }
    }

    /// Create a Microscope for a Llama-style model
    pub fn for_llama(num_layers: usize, num_heads: usize, hidden_size: usize) -> Self {
        Self::new(MicroscopeConfig {
            architecture: "llama".to_string(),
            num_layers,
            num_heads,
            hidden_size,
            ..Default::default()
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &MicroscopeConfig {
        &self.config
    }

    /// Get mutable access to the hook registry
    pub fn hooks_mut(&mut self) -> &mut hooks::HookRegistry {
        &mut self.hooks
    }

    /// Get access to the activation tracer
    pub fn tracer(&self) -> &activation::ActivationTracer {
        &self.tracer
    }

    /// Get mutable access to the activation tracer
    pub fn tracer_mut(&mut self) -> &mut activation::ActivationTracer {
        &mut self.tracer
    }
}

/// Python module initialization
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyMicroscope>()?;
    m.add_class::<python::PyActivationTrace>()?;
    m.add_class::<python::PyAttentionPattern>()?;
    m.add_class::<python::PyCircuit>()?;
    m.add_class::<python::PyAttentionAnalyzer>()?;
    m.add_class::<python::PyCircuitDiscoverer>()?;
    m.add_class::<python::PyInterventionEngine>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microscope_creation() {
        let scope = Microscope::for_llama(32, 32, 4096);
        assert_eq!(scope.config().num_layers, 32);
        assert_eq!(scope.config().architecture, "llama");
    }

    #[test]
    fn test_default_config() {
        let config = MicroscopeConfig::default();
        assert_eq!(config.num_layers, 32);
        assert!(config.use_mmap);
    }
}
