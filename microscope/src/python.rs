//! Python Bindings Module
//!
//! This module provides Python bindings via PyO3 for the Alignment Microscope.
//! It enables seamless integration with PyTorch and other Python ML frameworks.

use numpy::{PyArray2, PyArray3, PyArray4, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::activation::ActivationTrace;
use crate::attention::{AttentionAnalyzer, AttentionPattern};
use crate::circuit::{Circuit, CircuitDiscoverer, CircuitEdge, CircuitNode, ComponentType};
use crate::intervention::InterventionEngine;
use crate::{Microscope, MicroscopeConfig};

/// Python wrapper for the Microscope
#[pyclass(name = "Microscope")]
pub struct PyMicroscope {
    inner: Microscope,
}

#[pymethods]
impl PyMicroscope {
    /// Create a new Microscope
    #[new]
    #[pyo3(signature = (architecture="llama", num_layers=32, num_heads=32, hidden_size=4096))]
    fn new(architecture: &str, num_layers: usize, num_heads: usize, hidden_size: usize) -> Self {
        let config = MicroscopeConfig {
            architecture: architecture.to_string(),
            num_layers,
            num_heads,
            hidden_size,
            ..Default::default()
        };
        Self {
            inner: Microscope::new(config),
        }
    }

    /// Create a Microscope configured for Llama models
    #[staticmethod]
    fn for_llama(num_layers: usize, num_heads: usize, hidden_size: usize) -> Self {
        Self {
            inner: Microscope::for_llama(num_layers, num_heads, hidden_size),
        }
    }

    /// Start tracing activations
    fn start_trace(&mut self, input_tokens: Vec<u32>) {
        let arch = self.inner.config().architecture.clone();
        self.inner.tracer_mut().start_trace(&arch, input_tokens);
    }

    /// Stop tracing and return the trace
    fn stop_trace(&mut self) -> Option<PyActivationTrace> {
        self.inner
            .tracer_mut()
            .stop_trace()
            .map(|t| PyActivationTrace { inner: t })
    }

    /// Record an activation (called from Python hooks)
    fn record_activation(
        &self,
        _py: Python<'_>,
        layer: usize,
        component: &str,
        data: PyReadonlyArray3<f32>,
    ) -> PyResult<()> {
        let arr = data.as_array();
        let owned = arr.to_owned();
        self.inner
            .tracer()
            .record(layer, component, owned)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get configuration as a dict
    fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        let config = self.inner.config();
        dict.set_item("architecture", &config.architecture)?;
        dict.set_item("num_layers", config.num_layers)?;
        dict.set_item("num_heads", config.num_heads)?;
        dict.set_item("hidden_size", config.hidden_size)?;
        Ok(dict.into())
    }
}

/// Python wrapper for ActivationTrace
#[pyclass(name = "ActivationTrace")]
#[derive(Clone)]
pub struct PyActivationTrace {
    inner: ActivationTrace,
}

#[pymethods]
impl PyActivationTrace {
    /// Get activation for a specific layer and component
    fn get<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
        component: &str,
    ) -> Option<Bound<'py, PyArray3<f32>>> {
        self.inner.get(layer, component).map(|act| {
            let arr = act.as_array();
            PyArray3::from_owned_array_bound(py, arr)
        })
    }

    /// Get all layer indices that have activations
    fn layers(&self) -> Vec<usize> {
        let mut layers: Vec<usize> = self
            .inner
            .activations
            .values()
            .map(|a| a.layer)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        layers.sort();
        layers
    }

    /// Get all components for a layer
    fn components(&self, layer: usize) -> Vec<String> {
        self.inner
            .activations
            .values()
            .filter(|a| a.layer == layer)
            .map(|a| a.component.clone())
            .collect()
    }

    /// Get input tokens
    fn input_tokens(&self) -> Vec<u32> {
        self.inner.input_tokens.clone()
    }

    /// Export to JSON
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Import from JSON
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        ActivationTrace::from_json(json)
            .map(|t| PyActivationTrace { inner: t })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Compute statistics for an activation
    fn stats(&self, py: Python<'_>, layer: usize, component: &str) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);

        if let Some(act) = self.inner.get(layer, component) {
            dict.set_item("mean", act.mean())?;
            dict.set_item("variance", act.variance())?;
            dict.set_item("shape", act.shape.clone())?;

            let top_dims = act.top_dimensions(10);
            dict.set_item(
                "top_dimensions",
                top_dims
                    .into_iter()
                    .map(|(i, v)| (i, v))
                    .collect::<Vec<_>>(),
            )?;
        }

        Ok(dict.into())
    }
}

/// Python wrapper for AttentionPattern
#[pyclass(name = "AttentionPattern")]
#[derive(Clone)]
pub struct PyAttentionPattern {
    inner: AttentionPattern,
}

#[pymethods]
impl PyAttentionPattern {
    /// Create from a numpy array
    #[new]
    fn new(layer: usize, data: PyReadonlyArray4<f32>) -> Self {
        let arr = data.as_array().to_owned();
        Self {
            inner: AttentionPattern::new(layer, arr),
        }
    }

    /// Get as numpy array
    fn as_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f32>> {
        PyArray4::from_owned_array_bound(py, self.inner.as_array())
    }

    /// Get pattern for a specific head
    fn head_pattern<'py>(
        &self,
        py: Python<'py>,
        batch: usize,
        head: usize,
    ) -> Bound<'py, PyArray2<f32>> {
        PyArray2::from_owned_array_bound(py, self.inner.head_pattern(batch, head))
    }

    /// Compute entropy
    fn entropy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        PyArray3::from_owned_array_bound(py, self.inner.entropy())
    }

    /// Classify a head's attention pattern
    fn classify_head(&self, head: usize) -> String {
        format!("{:?}", self.inner.classify_head_type(head))
    }

    /// Get layer index
    #[getter]
    fn layer(&self) -> usize {
        self.inner.layer
    }

    /// Get shape
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }
}

/// Python wrapper for Circuit
#[pyclass(name = "Circuit")]
#[derive(Clone)]
pub struct PyCircuit {
    inner: Circuit,
}

#[pymethods]
impl PyCircuit {
    /// Create a new circuit
    #[new]
    fn new(name: &str, description: &str, behavior: &str) -> Self {
        Self {
            inner: Circuit::new(name, description, behavior),
        }
    }

    /// Add a node
    fn add_node(&mut self, layer: usize, component: &str, head: Option<usize>) {
        let comp_type = match component {
            "attention" | "attn" => ComponentType::AttentionHead,
            "mlp" => ComponentType::MLP,
            "embed" | "embedding" => ComponentType::Embedding,
            "unembed" | "unembedding" => ComponentType::Unembedding,
            "ln" | "layernorm" => ComponentType::LayerNorm,
            _ => ComponentType::Residual,
        };

        self.inner.add_node(CircuitNode {
            layer,
            component: comp_type,
            head,
            position: None,
        });
    }

    /// Add an edge
    #[pyo3(signature = (from_layer, from_component, from_head, to_layer, to_component, to_head, importance))]
    fn add_edge(
        &mut self,
        from_layer: usize,
        from_component: &str,
        from_head: Option<usize>,
        to_layer: usize,
        to_component: &str,
        to_head: Option<usize>,
        importance: f32,
    ) {
        let parse_component = |s: &str| match s {
            "attention" | "attn" => ComponentType::AttentionHead,
            "mlp" => ComponentType::MLP,
            "embed" | "embedding" => ComponentType::Embedding,
            "unembed" | "unembedding" => ComponentType::Unembedding,
            _ => ComponentType::Residual,
        };

        self.inner.add_edge(CircuitEdge {
            from: CircuitNode {
                layer: from_layer,
                component: parse_component(from_component),
                head: from_head,
                position: None,
            },
            to: CircuitNode {
                layer: to_layer,
                component: parse_component(to_component),
                head: to_head,
                position: None,
            },
            importance,
            metadata: HashMap::new(),
        });
    }

    /// Get minimal circuit above threshold
    fn minimal(&self, threshold: f32) -> PyCircuit {
        PyCircuit {
            inner: self.inner.minimal(threshold),
        }
    }

    /// Export to DOT format
    fn to_dot(&self) -> String {
        self.inner.to_dot()
    }

    /// Get name
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Get description
    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }

    /// Get number of nodes
    fn num_nodes(&self) -> usize {
        self.inner.nodes.len()
    }

    /// Get number of edges
    fn num_edges(&self) -> usize {
        self.inner.edges.len()
    }

    /// Get total importance
    fn total_importance(&self) -> f32 {
        self.inner.total_importance()
    }

    /// Get edges as list of tuples
    fn edges(&self) -> Vec<(String, String, f32)> {
        self.inner
            .edges
            .iter()
            .map(|e| (e.from.to_string(), e.to.to_string(), e.importance))
            .collect()
    }
}

/// Attention analyzer for Python
#[pyclass(name = "AttentionAnalyzer")]
pub struct PyAttentionAnalyzer {
    inner: AttentionAnalyzer,
}

#[pymethods]
impl PyAttentionAnalyzer {
    #[new]
    fn new() -> Self {
        Self {
            inner: AttentionAnalyzer::new(),
        }
    }

    /// Add attention pattern
    fn add_pattern(&mut self, pattern: &PyAttentionPattern) {
        self.inner.add_pattern(pattern.inner.clone());
    }

    /// Analyze all heads
    fn analyze_all_heads(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let analyses = self.inner.analyze_all_heads();

        analyses
            .into_iter()
            .map(|a| {
                let dict = PyDict::new_bound(py);
                dict.set_item("layer", a.location.0)?;
                dict.set_item("head", a.location.1)?;
                dict.set_item("type", format!("{:?}", a.head_type))?;
                dict.set_item("importance", a.importance)?;
                dict.set_item("avg_entropy", a.avg_entropy)?;
                dict.set_item("sparsity", a.sparsity)?;
                Ok(dict.into())
            })
            .collect()
    }

    /// Find induction heads
    fn find_induction_heads(&self) -> Vec<(usize, usize)> {
        self.inner.find_induction_heads()
    }

    /// Compute attention flow matrix
    fn compute_attention_flow<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        PyArray2::from_owned_array_bound(py, self.inner.compute_attention_flow())
    }
}

/// Circuit discoverer for Python
#[pyclass(name = "CircuitDiscoverer")]
pub struct PyCircuitDiscoverer {
    inner: CircuitDiscoverer,
}

#[pymethods]
impl PyCircuitDiscoverer {
    #[new]
    fn new() -> Self {
        Self {
            inner: CircuitDiscoverer::new(),
        }
    }

    /// Add a trace
    fn add_trace(&mut self, trace: &PyActivationTrace) {
        self.inner.add_trace(trace.inner.clone());
    }

    /// Add attention pattern
    fn add_attention(&mut self, layer: usize, pattern: &PyAttentionPattern) {
        self.inner.add_attention(layer, pattern.inner.clone());
    }

    /// Discover known patterns
    fn discover_known_patterns(&self) -> Vec<PyCircuit> {
        self.inner
            .discover_known_patterns()
            .into_iter()
            .map(|c| PyCircuit { inner: c })
            .collect()
    }
}

/// Intervention engine for Python
#[pyclass(name = "InterventionEngine")]
pub struct PyInterventionEngine {
    inner: InterventionEngine,
}

#[pymethods]
impl PyInterventionEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: InterventionEngine::new(),
        }
    }

    /// Cache a trace
    fn cache_trace(&mut self, name: &str, trace: &PyActivationTrace) {
        self.inner.cache_trace(name, trace.inner.clone());
    }

    /// Get results summary
    fn results_summary(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .results()
            .iter()
            .map(|r| {
                let dict = PyDict::new_bound(py);
                dict.set_item("layer", r.intervention.target.layer)?;
                dict.set_item("component", format!("{:?}", r.intervention.target.component))?;
                dict.set_item("baseline", r.baseline_metric)?;
                dict.set_item("intervened", r.intervened_metric)?;
                dict.set_item("effect", r.effect)?;
                Ok(dict.into())
            })
            .collect()
    }

    /// Clear results
    fn clear_results(&mut self) {
        self.inner.clear_results();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_microscope_creation() {
        let scope = PyMicroscope::new("llama", 32, 32, 4096);
        // Basic creation test
    }
}
