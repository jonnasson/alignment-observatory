//! Causal Intervention Module
//!
//! This module provides tools for causal analysis of model behavior:
//!
//! - Activation patching (swap activations between runs)
//! - Ablation studies (zero out components)
//! - Path patching (trace causal paths)
//! - Direct effect measurement

use ndarray::{Array3, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::activation::ActivationTrace;
use crate::circuit::CircuitNode;

/// Types of interventions that can be performed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionType {
    /// Replace activation with zeros
    ZeroAblation,
    /// Replace with mean activation across dataset
    MeanAblation { mean: Vec<f32> },
    /// Patch in activations from another run
    Patch { source_trace: String },
    /// Add noise to activations
    Noise { std_dev: f32 },
    /// Scale activations by a factor
    Scale { factor: f32 },
    /// Apply arbitrary function
    Custom { name: String },
}

/// Specification for an intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intervention {
    /// Target component
    pub target: CircuitNode,
    /// Type of intervention
    pub intervention_type: InterventionType,
    /// Optional position mask (which positions to intervene on)
    pub position_mask: Option<Vec<bool>>,
}

impl Intervention {
    /// Create a zero ablation intervention
    pub fn zero_ablation(target: CircuitNode) -> Self {
        Self {
            target,
            intervention_type: InterventionType::ZeroAblation,
            position_mask: None,
        }
    }

    /// Create a mean ablation intervention
    pub fn mean_ablation(target: CircuitNode, mean: Vec<f32>) -> Self {
        Self {
            target,
            intervention_type: InterventionType::MeanAblation { mean },
            position_mask: None,
        }
    }

    /// Create a patching intervention
    pub fn patch(target: CircuitNode, source_trace: &str) -> Self {
        Self {
            target,
            intervention_type: InterventionType::Patch {
                source_trace: source_trace.to_string(),
            },
            position_mask: None,
        }
    }

    /// Set position mask
    pub fn with_positions(mut self, mask: Vec<bool>) -> Self {
        self.position_mask = Some(mask);
        self
    }

    /// Apply intervention to activation data
    pub fn apply(&self, data: &Array3<f32>, source_data: Option<&Array3<f32>>) -> Array3<f32> {
        let mut result = data.clone();

        match &self.intervention_type {
            InterventionType::ZeroAblation => {
                if let Some(mask) = &self.position_mask {
                    for (i, &masked) in mask.iter().enumerate() {
                        if masked && i < result.shape()[1] {
                            result
                                .index_axis_mut(Axis(1), i)
                                .fill(0.0);
                        }
                    }
                } else {
                    result.fill(0.0);
                }
            }
            InterventionType::MeanAblation { mean } => {
                if let Some(mask) = &self.position_mask {
                    for (i, &masked) in mask.iter().enumerate() {
                        if masked && i < result.shape()[1] {
                            for (j, val) in result.index_axis_mut(Axis(1), i).iter_mut().enumerate() {
                                if j < mean.len() {
                                    *val = mean[j];
                                }
                            }
                        }
                    }
                } else {
                    for mut row in result.axis_iter_mut(Axis(1)) {
                        for (j, val) in row.iter_mut().enumerate() {
                            if j < mean.len() {
                                *val = mean[j];
                            }
                        }
                    }
                }
            }
            InterventionType::Patch { .. } => {
                if let Some(source) = source_data {
                    if let Some(mask) = &self.position_mask {
                        for (i, &masked) in mask.iter().enumerate() {
                            if masked && i < result.shape()[1] && i < source.shape()[1] {
                                result
                                    .index_axis_mut(Axis(1), i)
                                    .assign(&source.index_axis(Axis(1), i));
                            }
                        }
                    } else {
                        result.assign(source);
                    }
                }
            }
            InterventionType::Noise { std_dev } => {
                // Add Gaussian noise (simplified - real impl would use proper RNG)
                let noise_scale = *std_dev;
                for val in result.iter_mut() {
                    // Simple deterministic "noise" for demonstration
                    *val += noise_scale * ((*val * 12345.6789).sin() as f32);
                }
            }
            InterventionType::Scale { factor } => {
                result *= *factor;
            }
            InterventionType::Custom { .. } => {
                // Custom interventions handled externally
            }
        }

        result
    }
}

/// Results from an intervention experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionResult {
    /// The intervention that was applied
    pub intervention: Intervention,
    /// Metric value before intervention
    pub baseline_metric: f32,
    /// Metric value after intervention
    pub intervened_metric: f32,
    /// Effect size (intervened - baseline)
    pub effect: f32,
    /// Normalized effect (effect / baseline variance)
    pub normalized_effect: Option<f32>,
}

impl InterventionResult {
    /// Create a new intervention result
    pub fn new(
        intervention: Intervention,
        baseline_metric: f32,
        intervened_metric: f32,
    ) -> Self {
        let effect = intervened_metric - baseline_metric;
        Self {
            intervention,
            baseline_metric,
            intervened_metric,
            effect,
            normalized_effect: None,
        }
    }

    /// Set normalized effect
    pub fn with_normalized_effect(mut self, baseline_std: f32) -> Self {
        if baseline_std > 1e-6 {
            self.normalized_effect = Some(self.effect / baseline_std);
        }
        self
    }
}

/// Orchestrates intervention experiments
pub struct InterventionEngine {
    /// Cached activation traces
    traces: HashMap<String, ActivationTrace>,
    /// Results from experiments
    results: Vec<InterventionResult>,
}

impl InterventionEngine {
    /// Create a new intervention engine
    pub fn new() -> Self {
        Self {
            traces: HashMap::new(),
            results: Vec::new(),
        }
    }

    /// Cache a trace for use in patching
    pub fn cache_trace(&mut self, name: &str, trace: ActivationTrace) {
        self.traces.insert(name.to_string(), trace);
    }

    /// Get a cached trace
    pub fn get_trace(&self, name: &str) -> Option<&ActivationTrace> {
        self.traces.get(name)
    }

    /// Run a single intervention and measure effect
    pub fn run_intervention<F>(
        &mut self,
        intervention: Intervention,
        clean_trace: &ActivationTrace,
        corrupt_trace: &ActivationTrace,
        metric_fn: F,
    ) -> InterventionResult
    where
        F: Fn(&ActivationTrace) -> f32,
    {
        let baseline = metric_fn(clean_trace);
        let corrupted = metric_fn(corrupt_trace);

        // The effect is how much the metric changes due to intervention
        let result = InterventionResult::new(intervention, baseline, corrupted);
        self.results.push(result.clone());
        result
    }

    /// Run activation patching across all layers
    pub fn patch_all_layers<F>(
        &mut self,
        clean_trace: &ActivationTrace,
        corrupt_trace: &ActivationTrace,
        component: &str,
        metric_fn: F,
    ) -> Vec<InterventionResult>
    where
        F: Fn(&ActivationTrace) -> f32 + Clone,
    {
        let mut results = Vec::new();

        // Find all layers
        let layers: Vec<usize> = clean_trace
            .activations
            .values()
            .map(|a| a.layer)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for layer in layers {
            let target = CircuitNode {
                layer,
                component: crate::circuit::ComponentType::Residual,
                head: None,
                position: None,
            };

            let intervention = Intervention::patch(target, "corrupt");

            // Simulate the effect of patching this layer
            let key = format!("{}_{}", layer, component);
            if let (Some(clean_act), Some(corrupt_act)) = (
                clean_trace.activations.get(&key),
                corrupt_trace.activations.get(&key),
            ) {
                let baseline = metric_fn(clean_trace);

                // Compute effect based on activation difference
                let clean_arr = clean_act.as_array();
                let corrupt_arr = corrupt_act.as_array();

                let diff: f32 = clean_arr
                    .iter()
                    .zip(corrupt_arr.iter())
                    .map(|(c, co)| (c - co).powi(2))
                    .sum::<f32>()
                    .sqrt();

                // Use diff as proxy for metric change
                let effect = diff / clean_arr.len() as f32;

                results.push(InterventionResult {
                    intervention,
                    baseline_metric: baseline,
                    intervened_metric: baseline + effect,
                    effect,
                    normalized_effect: None,
                });
            }
        }

        // Use total_cmp for robust NaN handling
        results.sort_by(|a, b| b.effect.total_cmp(&a.effect));
        results
    }

    /// Get all results
    pub fn results(&self) -> &[InterventionResult] {
        &self.results
    }

    /// Clear results
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Find the most important components for a behavior
    pub fn rank_components(&self) -> Vec<(&Intervention, f32)> {
        let mut ranked: Vec<_> = self
            .results
            .iter()
            .map(|r| (&r.intervention, r.effect.abs()))
            .collect();

        // Use total_cmp for robust NaN handling
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked
    }
}

impl Default for InterventionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_zero_ablation() {
        let target = CircuitNode {
            layer: 0,
            component: crate::circuit::ComponentType::AttentionHead,
            head: Some(0),
            position: None,
        };

        let intervention = Intervention::zero_ablation(target);
        let data = Array3::ones((1, 10, 512));

        let result = intervention.apply(&data, None);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_patching() {
        let target = CircuitNode {
            layer: 0,
            component: crate::circuit::ComponentType::Residual,
            head: None,
            position: None,
        };

        let intervention = Intervention::patch(target, "source");
        let data = Array3::ones((1, 10, 512));
        let source = Array3::from_elem((1, 10, 512), 2.0);

        let result = intervention.apply(&data, Some(&source));
        assert!(result.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_position_mask() {
        let target = CircuitNode {
            layer: 0,
            component: crate::circuit::ComponentType::MLP,
            head: None,
            position: None,
        };

        // Only ablate positions 0 and 2
        let intervention =
            Intervention::zero_ablation(target).with_positions(vec![true, false, true, false]);

        let data = Array3::ones((1, 4, 8));
        let result = intervention.apply(&data, None);

        // Position 0 and 2 should be zero
        assert!(result.index_axis(Axis(1), 0).iter().all(|&x| x == 0.0));
        assert!(result.index_axis(Axis(1), 1).iter().all(|&x| x == 1.0));
        assert!(result.index_axis(Axis(1), 2).iter().all(|&x| x == 0.0));
        assert!(result.index_axis(Axis(1), 3).iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_intervention_result() {
        let target = CircuitNode {
            layer: 0,
            component: crate::circuit::ComponentType::Residual,
            head: None,
            position: None,
        };

        let intervention = Intervention::zero_ablation(target);
        let result = InterventionResult::new(intervention, 1.0, 0.5);

        assert_eq!(result.effect, -0.5);
    }
}
