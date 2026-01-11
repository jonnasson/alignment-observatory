//! Sparse Autoencoder Module
//!
//! This module provides tools for working with Sparse Autoencoders (SAEs):
//!
//! - SAE encoding and decoding
//! - Feature activation analysis
//! - Sparsity computation
//! - Top-k feature selection

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for a Sparse Autoencoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAEConfig {
    /// Input dimension (hidden size of the model)
    pub d_in: usize,
    /// Feature dimension (number of SAE features)
    pub d_sae: usize,
    /// Activation function type
    pub activation: ActivationType,
    /// Whether the encoder has bias
    pub encoder_bias: bool,
    /// Whether the decoder has bias
    pub decoder_bias: bool,
}

/// Activation function type for SAE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// ReLU activation
    ReLU,
    /// TopK activation (only keep top k activations)
    TopK(usize),
    /// JumpReLU (ReLU with threshold)
    JumpReLU,
}

/// Represents the weights of a trained SAE
#[derive(Debug, Clone)]
pub struct SAEWeights {
    /// Encoder weights [d_in, d_sae]
    pub w_enc: Array2<f32>,
    /// Decoder weights [d_sae, d_in]
    pub w_dec: Array2<f32>,
    /// Encoder bias [d_sae] (optional)
    pub b_enc: Option<Array1<f32>>,
    /// Decoder bias [d_in] (optional)
    pub b_dec: Option<Array1<f32>>,
    /// Configuration
    pub config: SAEConfig,
}

impl SAEWeights {
    /// Create new SAE weights
    pub fn new(
        w_enc: Array2<f32>,
        w_dec: Array2<f32>,
        b_enc: Option<Array1<f32>>,
        b_dec: Option<Array1<f32>>,
        config: SAEConfig,
    ) -> Self {
        Self {
            w_enc,
            w_dec,
            b_enc,
            b_dec,
            config,
        }
    }

    /// Get input dimension
    pub fn d_in(&self) -> usize {
        self.config.d_in
    }

    /// Get SAE feature dimension
    pub fn d_sae(&self) -> usize {
        self.config.d_sae
    }
}

/// Result of SAE encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAEFeatures {
    /// Feature activations [batch, seq_len, d_sae] or [batch * seq_len, d_sae]
    shape: Vec<usize>,
    data: Vec<f32>,
    /// Sparsity ratio (fraction of zero activations)
    pub sparsity: f32,
    /// Number of active features per position (mean)
    pub mean_active_features: f32,
}

impl SAEFeatures {
    /// Create from raw data
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let total = data.len() as f32;
        let zeros = data.iter().filter(|&&x| x == 0.0).count() as f32;
        let sparsity = zeros / total;

        // Compute mean active features per position
        let d_sae = *shape.last().unwrap_or(&1);
        let num_positions = data.len() / d_sae;
        let active_count: usize = data.iter().filter(|&&x| x > 0.0).count();
        let mean_active_features = active_count as f32 / num_positions as f32;

        Self {
            shape,
            data,
            sparsity,
            mean_active_features,
        }
    }

    /// Get as 2D array [positions, d_sae]
    /// Returns a zero array if shape is invalid (should not happen in normal use)
    pub fn as_array(&self) -> Array2<f32> {
        let d_sae = *self.shape.last().unwrap_or(&1);
        let positions: usize = if d_sae > 0 { self.data.len() / d_sae } else { 0 };
        Array2::from_shape_vec((positions, d_sae), self.data.clone())
            .unwrap_or_else(|_| Array2::zeros((1, 1)))
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get raw data
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get indices of active features (above threshold)
    pub fn active_features(&self, threshold: f32) -> Vec<Vec<usize>> {
        let arr = self.as_array();
        arr.axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(_, &v)| v > threshold)
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect()
    }

    /// Get top-k feature indices and values per position
    pub fn top_k_features(&self, k: usize) -> Vec<Vec<(usize, f32)>> {
        let arr = self.as_array();
        arr.axis_iter(Axis(0))
            .map(|row| {
                let mut indexed: Vec<(usize, f32)> = row
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i, v))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(k);
                indexed
            })
            .collect()
    }

    /// Compute feature frequency across all positions
    pub fn feature_frequency(&self, threshold: f32) -> Array1<f32> {
        let arr = self.as_array();
        let num_positions = arr.shape()[0] as f32;
        arr.map_axis(Axis(0), |col| {
            col.iter().filter(|&&v| v > threshold).count() as f32 / num_positions
        })
    }
}

/// SAE Encoder/Decoder operations
pub struct SAEEncoder;

impl SAEEncoder {
    /// Encode activations to SAE features
    pub fn encode(
        activations: ArrayView2<f32>,
        weights: &SAEWeights,
    ) -> SAEFeatures {
        // Pre-encoder: subtract decoder bias if present
        let centered = if let Some(ref b_dec) = weights.b_dec {
            let b_dec_2d = b_dec.view().insert_axis(Axis(0));
            &activations - &b_dec_2d
        } else {
            activations.to_owned()
        };

        // Encode: x @ W_enc + b_enc
        let mut features = centered.dot(&weights.w_enc);

        if let Some(ref b_enc) = weights.b_enc {
            features = &features + &b_enc.view().insert_axis(Axis(0));
        }

        // Apply activation
        let activated = match weights.config.activation {
            ActivationType::ReLU => {
                features.mapv(|x| if x > 0.0 { x } else { 0.0 })
            }
            ActivationType::TopK(k) => {
                Self::apply_topk(features.view(), k)
            }
            ActivationType::JumpReLU => {
                // JumpReLU with default threshold of 0.0
                features.mapv(|x| if x > 0.0 { x } else { 0.0 })
            }
        };

        let shape = vec![activated.shape()[0], activated.shape()[1]];
        SAEFeatures::new(shape, activated.into_raw_vec())
    }

    /// Apply top-k activation
    fn apply_topk(features: ArrayView2<f32>, k: usize) -> Array2<f32> {
        let mut result = Array2::zeros(features.raw_dim());

        for (i, row) in features.axis_iter(Axis(0)).enumerate() {
            let mut indexed: Vec<(usize, f32)> = row
                .iter()
                .enumerate()
                .map(|(j, &v)| (j, v))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (idx, val) in indexed.iter().take(k) {
                if *val > 0.0 {
                    result[[i, *idx]] = *val;
                }
            }
        }

        result
    }

    /// Decode SAE features back to activations
    pub fn decode(
        features: &SAEFeatures,
        weights: &SAEWeights,
    ) -> Array2<f32> {
        let features_arr = features.as_array();

        // Decode: features @ W_dec + b_dec
        let mut reconstructed = features_arr.dot(&weights.w_dec);

        if let Some(ref b_dec) = weights.b_dec {
            reconstructed = &reconstructed + &b_dec.view().insert_axis(Axis(0));
        }

        reconstructed
    }

    /// Compute reconstruction error
    pub fn reconstruction_error(
        original: ArrayView2<f32>,
        features: &SAEFeatures,
        weights: &SAEWeights,
    ) -> f32 {
        let reconstructed = Self::decode(features, weights);
        let diff = &original.to_owned() - &reconstructed;
        let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
        mse
    }
}

/// Feature analysis utilities
pub struct FeatureAnalyzer;

impl FeatureAnalyzer {
    /// Find features that activate for specific token patterns
    pub fn find_pattern_features(
        features_by_position: &[SAEFeatures],
        pattern_mask: &[bool],
        threshold: f32,
    ) -> HashMap<usize, f32> {
        let mut feature_activations: HashMap<usize, Vec<f32>> = HashMap::new();

        for (i, (features, &in_pattern)) in features_by_position.iter().zip(pattern_mask).enumerate() {
            if !in_pattern {
                continue;
            }

            let arr = features.as_array();
            // Assuming single position per SAEFeatures for this function
            if arr.shape()[0] > 0 {
                for (feat_idx, &val) in arr.row(0).iter().enumerate() {
                    if val > threshold {
                        feature_activations.entry(feat_idx).or_default().push(val);
                    }
                }
            }
        }

        // Compute mean activation per feature
        feature_activations
            .into_iter()
            .map(|(idx, vals)| {
                let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                (idx, mean)
            })
            .collect()
    }

    /// Compute feature co-activation matrix
    pub fn feature_coactivation(
        features: &SAEFeatures,
        top_k: usize,
    ) -> Array2<f32> {
        let arr = features.as_array();
        let d_sae = arr.shape()[1];

        // Get top-k features per position
        let active_sets = features.top_k_features(top_k);

        // Build co-activation counts
        let mut coact = Array2::zeros((d_sae, d_sae));

        for active_features in &active_sets {
            for (i, _) in active_features {
                for (j, _) in active_features {
                    coact[[*i, *j]] += 1.0;
                }
            }
        }

        // Normalize by number of positions
        let num_positions = arr.shape()[0] as f32;
        coact /= num_positions;

        coact
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_weights() -> SAEWeights {
        let d_in = 64;
        let d_sae = 256;

        let config = SAEConfig {
            d_in,
            d_sae,
            activation: ActivationType::ReLU,
            encoder_bias: true,
            decoder_bias: true,
        };

        // Initialize random-ish weights
        let w_enc = Array2::from_shape_fn((d_in, d_sae), |(i, j)| {
            ((i + j) as f32 * 0.01).sin()
        });
        let w_dec = Array2::from_shape_fn((d_sae, d_in), |(i, j)| {
            ((i + j) as f32 * 0.01).cos()
        });
        let b_enc = Some(Array1::zeros(d_sae));
        let b_dec = Some(Array1::zeros(d_in));

        SAEWeights::new(w_enc, w_dec, b_enc, b_dec, config)
    }

    #[test]
    fn test_sae_encode_decode() {
        let weights = create_test_weights();
        let activations = Array2::from_shape_fn((4, 64), |(i, j)| {
            ((i * j) as f32 * 0.1).tanh()
        });

        let features = SAEEncoder::encode(activations.view(), &weights);

        assert_eq!(features.shape(), &[4, 256]);
        assert!(features.sparsity >= 0.0 && features.sparsity <= 1.0);
    }

    #[test]
    fn test_sae_features_active() {
        let weights = create_test_weights();
        let activations = Array2::ones((2, 64));

        let features = SAEEncoder::encode(activations.view(), &weights);
        let active = features.active_features(0.0);

        assert_eq!(active.len(), 2);  // 2 positions
    }

    #[test]
    fn test_sae_topk_features() {
        let weights = create_test_weights();
        let activations = Array2::ones((2, 64));

        let features = SAEEncoder::encode(activations.view(), &weights);
        let top5 = features.top_k_features(5);

        assert_eq!(top5.len(), 2);
        assert!(top5[0].len() <= 5);
    }

    #[test]
    fn test_reconstruction_error() {
        let weights = create_test_weights();
        let activations = Array2::ones((2, 64));

        let features = SAEEncoder::encode(activations.view(), &weights);
        let error = SAEEncoder::reconstruction_error(activations.view(), &features, &weights);

        // Error should be non-negative
        assert!(error >= 0.0);
    }

    #[test]
    fn test_feature_frequency() {
        let features = SAEFeatures::new(
            vec![4, 10],
            vec![
                1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );

        let freq = features.feature_frequency(0.0);

        // Feature 0 is active in 3/4 positions
        assert!((freq[0] - 0.75).abs() < 0.01);
        // Feature 1 is active in 1/4 positions
        assert!((freq[1] - 0.25).abs() < 0.01);
    }
}
