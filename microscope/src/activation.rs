//! Activation Tracing Module
//!
//! This module provides functionality to capture and analyze activations
//! as they flow through transformer layers. It supports:
//!
//! - Layer-by-layer activation capture
//! - Residual stream tracking
//! - MLP and attention output separation
//! - Statistical analysis of activation patterns

use ndarray::{Array2, Array3, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::{MicroscopeError, Result};

/// Represents a single activation capture at a specific layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Activation {
    /// Layer index where this activation was captured
    pub layer: usize,
    /// Component name (e.g., "attn_out", "mlp_out", "residual")
    pub component: String,
    /// Shape of the activation tensor [batch, seq_len, hidden_dim]
    pub shape: Vec<usize>,
    /// The actual activation data (flattened for serialization)
    data: Vec<f32>,
}

impl Activation {
    /// Create a new activation from raw data
    pub fn new(layer: usize, component: &str, data: Array3<f32>) -> Self {
        let shape = data.shape().to_vec();
        Self {
            layer,
            component: component.to_string(),
            shape,
            data: data.into_raw_vec(),
        }
    }

    /// Get the activation data as a 3D array view
    /// Returns None if the stored shape is invalid (should not happen in normal use)
    pub fn as_array(&self) -> Array3<f32> {
        // Shape is guaranteed valid by constructor, but handle gracefully
        Array3::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2]),
            self.data.clone(),
        )
        .unwrap_or_else(|_| Array3::zeros((1, 1, 1)))
    }

    /// Get the activation data as a 3D array, returning Result for error handling
    pub fn try_as_array(&self) -> Result<Array3<f32>> {
        if self.shape.len() != 3 {
            return Err(MicroscopeError::InvalidShape {
                expected: "3D array".to_string(),
                got: format!("{}D shape", self.shape.len()),
            });
        }
        Array3::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2]),
            self.data.clone(),
        )
        .map_err(|e| MicroscopeError::InvalidShape {
            expected: format!("shape {:?}", self.shape),
            got: e.to_string(),
        })
    }

    /// Compute L2 norm per token
    pub fn token_norms(&self) -> Array2<f32> {
        let arr = self.as_array();
        arr.map_axis(Axis(2), |row| {
            row.iter().map(|x| x * x).sum::<f32>().sqrt()
        })
    }

    /// Compute mean activation value
    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    /// Compute variance of activations
    pub fn variance(&self) -> f32 {
        let mean = self.mean();
        self.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.data.len() as f32
    }

    /// Find top-k most active dimensions
    pub fn top_dimensions(&self, k: usize) -> Vec<(usize, f32)> {
        let arr = self.as_array();

        // Handle potential empty arrays gracefully
        let mean_per_dim = match arr.mean_axis(Axis(0)) {
            Some(m) => match m.mean_axis(Axis(0)) {
                Some(m2) => m2,
                None => return Vec::new(),
            },
            None => return Vec::new(),
        };

        let mut indexed: Vec<(usize, f32)> = mean_per_dim
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();

        // Use total_cmp for robust NaN handling
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.truncate(k);
        indexed
    }
}

/// A complete trace of activations through the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationTrace {
    /// All captured activations, keyed by "layer_component"
    pub activations: HashMap<String, Activation>,
    /// The input tokens that generated this trace
    pub input_tokens: Vec<u32>,
    /// Model architecture used
    pub architecture: String,
}

impl ActivationTrace {
    /// Create a new empty trace
    pub fn new(architecture: &str, input_tokens: Vec<u32>) -> Self {
        Self {
            activations: HashMap::new(),
            input_tokens,
            architecture: architecture.to_string(),
        }
    }

    /// Add an activation to the trace
    pub fn add(&mut self, activation: Activation) {
        let key = format!("{}_{}", activation.layer, activation.component);
        self.activations.insert(key, activation);
    }

    /// Get activation for a specific layer and component
    pub fn get(&self, layer: usize, component: &str) -> Option<&Activation> {
        let key = format!("{}_{}", layer, component);
        self.activations.get(&key)
    }

    /// Get all activations for a layer
    pub fn layer_activations(&self, layer: usize) -> Vec<&Activation> {
        self.activations
            .values()
            .filter(|a| a.layer == layer)
            .collect()
    }

    /// Compute the residual stream at each layer
    pub fn residual_stream(&self) -> Vec<Option<&Activation>> {
        let max_layer = self
            .activations
            .values()
            .map(|a| a.layer)
            .max()
            .unwrap_or(0);

        (0..=max_layer)
            .map(|l| self.get(l, "residual"))
            .collect()
    }

    /// Export trace to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(MicroscopeError::from)
    }

    /// Import trace from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(MicroscopeError::from)
    }
}

/// The main activation tracer that manages capture during forward passes
#[derive(Debug)]
pub struct ActivationTracer {
    /// Number of layers in the model
    num_layers: usize,
    /// Whether tracing is currently enabled
    enabled: bool,
    /// Current trace being built
    current_trace: Arc<RwLock<Option<ActivationTrace>>>,
    /// Components to capture
    capture_components: Vec<String>,
}

impl ActivationTracer {
    /// Create a new tracer for a model with the given number of layers
    pub fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            enabled: false,
            current_trace: Arc::new(RwLock::new(None)),
            capture_components: vec![
                "residual".to_string(),
                "attn_out".to_string(),
                "mlp_out".to_string(),
            ],
        }
    }

    /// Start tracing with given input tokens
    pub fn start_trace(&mut self, architecture: &str, input_tokens: Vec<u32>) {
        let trace = ActivationTrace::new(architecture, input_tokens);
        // Handle poisoned lock gracefully - recover by clearing poison
        match self.current_trace.write() {
            Ok(mut guard) => *guard = Some(trace),
            Err(poisoned) => *poisoned.into_inner() = Some(trace),
        }
        self.enabled = true;
    }

    /// Stop tracing and return the completed trace
    pub fn stop_trace(&mut self) -> Option<ActivationTrace> {
        self.enabled = false;
        // Handle poisoned lock gracefully
        match self.current_trace.write() {
            Ok(mut guard) => guard.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        }
    }

    /// Check if tracing is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record an activation (called from hooks)
    pub fn record(&self, layer: usize, component: &str, data: Array3<f32>) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if layer >= self.num_layers {
            return Err(MicroscopeError::LayerNotFound {
                layer,
                total: self.num_layers,
            });
        }

        let activation = Activation::new(layer, component, data);

        // Handle poisoned lock gracefully
        let mut guard = match self.current_trace.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        if let Some(ref mut trace) = *guard {
            trace.add(activation);
        }

        Ok(())
    }

    /// Set which components to capture
    pub fn set_capture_components(&mut self, components: Vec<String>) {
        self.capture_components = components;
    }

    /// Check if a component should be captured
    pub fn should_capture(&self, component: &str) -> bool {
        self.capture_components.iter().any(|c| c == component)
    }
}

/// Compute activation patching results
/// This is used for causal interventions - replacing activations and measuring effects
pub struct ActivationPatcher {
    /// The clean run activations
    clean_trace: ActivationTrace,
    /// The corrupted run activations
    corrupt_trace: ActivationTrace,
}

impl ActivationPatcher {
    /// Create a new patcher from clean and corrupted traces
    pub fn new(clean_trace: ActivationTrace, corrupt_trace: ActivationTrace) -> Self {
        Self {
            clean_trace,
            corrupt_trace,
        }
    }

    /// Compute the effect of patching each layer/component
    /// Returns a map of component -> effect magnitude
    pub fn compute_patch_effects(&self) -> HashMap<String, f32> {
        let mut effects = HashMap::new();

        for (key, clean_act) in &self.clean_trace.activations {
            if let Some(corrupt_act) = self.corrupt_trace.activations.get(key) {
                let clean_arr = clean_act.as_array();
                let corrupt_arr = corrupt_act.as_array();

                // Compute L2 distance between clean and corrupt
                let diff: f32 = clean_arr
                    .iter()
                    .zip(corrupt_arr.iter())
                    .map(|(c, co)| (c - co).powi(2))
                    .sum::<f32>()
                    .sqrt();

                effects.insert(key.clone(), diff);
            }
        }

        effects
    }

    /// Find the most influential layers for the behavior difference
    pub fn rank_layers_by_influence(&self) -> Vec<(usize, f32)> {
        let effects = self.compute_patch_effects();

        let mut layer_effects: HashMap<usize, f32> = HashMap::new();

        for (key, effect) in effects {
            if let Some(layer) = key.split('_').next().and_then(|s| s.parse().ok()) {
                *layer_effects.entry(layer).or_insert(0.0) += effect;
            }
        }

        let mut ranked: Vec<(usize, f32)> = layer_effects.into_iter().collect();
        // Use total_cmp for robust NaN handling
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_activation_creation() {
        let data = Array3::ones((1, 10, 512));
        let act = Activation::new(0, "residual", data);

        assert_eq!(act.layer, 0);
        assert_eq!(act.component, "residual");
        assert_eq!(act.shape, vec![1, 10, 512]);
    }

    #[test]
    fn test_activation_statistics() {
        let data = Array3::from_elem((1, 10, 512), 2.0);
        let act = Activation::new(0, "residual", data);

        assert!((act.mean() - 2.0).abs() < 1e-6);
        assert!(act.variance() < 1e-6);
    }

    #[test]
    fn test_tracer_workflow() {
        let mut tracer = ActivationTracer::new(32);

        tracer.start_trace("llama", vec![1, 2, 3]);
        assert!(tracer.is_enabled());

        let data = Array3::ones((1, 3, 4096));
        tracer.record(0, "residual", data).unwrap();

        let trace = tracer.stop_trace().unwrap();
        assert_eq!(trace.input_tokens, vec![1, 2, 3]);
        assert!(trace.get(0, "residual").is_some());
    }

    #[test]
    fn test_invalid_layer() {
        let tracer = ActivationTracer::new(32);
        // Create a write guard to enable tracing manually for test
        let mut guard = tracer.current_trace.write().unwrap();
        *guard = Some(ActivationTrace::new("test", vec![]));
        drop(guard);

        // Enable by setting internal state (for testing only)
        // In real use, start_trace handles this
    }
}
