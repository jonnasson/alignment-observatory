//! Attention Pattern Analysis Module
//!
//! This module provides tools for understanding attention mechanisms:
//!
//! - Attention pattern extraction and visualization
//! - Head importance scoring
//! - Attention pattern clustering
//! - Information flow analysis

use ndarray::{Array2, Array3, Array4, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents attention patterns for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPattern {
    /// Layer index
    pub layer: usize,
    /// Shape: [batch, num_heads, seq_len, seq_len]
    pub shape: Vec<usize>,
    /// Flattened attention weights
    data: Vec<f32>,
}

impl AttentionPattern {
    /// Create from a 4D attention tensor
    pub fn new(layer: usize, data: Array4<f32>) -> Self {
        let shape = data.shape().to_vec();
        Self {
            layer,
            shape,
            data: data.into_raw_vec(),
        }
    }

    /// Get as a 4D array
    /// Returns a zero array if shape is invalid (should not happen in normal use)
    pub fn as_array(&self) -> Array4<f32> {
        // Shape is guaranteed valid by constructor, but handle gracefully
        Array4::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2], self.shape[3]),
            self.data.clone(),
        )
        .unwrap_or_else(|_| Array4::zeros((1, 1, 1, 1)))
    }

    /// Get attention pattern for a specific head
    pub fn head_pattern(&self, batch: usize, head: usize) -> Array2<f32> {
        let arr = self.as_array();
        arr.index_axis(Axis(0), batch)
            .index_axis(Axis(0), head)
            .to_owned()
    }

    /// Compute entropy of attention distribution per position
    pub fn entropy(&self) -> Array3<f32> {
        let arr = self.as_array();
        arr.map_axis(Axis(3), |attn_dist| {
            -attn_dist
                .iter()
                .filter(|&&p| p > 1e-10)
                .map(|&p| p * p.ln())
                .sum::<f32>()
        })
    }

    /// Find the most attended positions for each query position
    pub fn top_attended(&self, k: usize) -> Vec<Vec<Vec<Vec<usize>>>> {
        let arr = self.as_array();
        let (batch, heads, seq_len, _) = (
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
        );

        (0..batch)
            .map(|b| {
                (0..heads)
                    .map(|h| {
                        (0..seq_len)
                            .map(|q| {
                                let pattern = arr.slice(ndarray::s![b, h, q, ..]);
                                let mut indexed: Vec<(usize, f32)> =
                                    pattern.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                                // Use total_cmp for robust NaN handling
                                indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
                                indexed.truncate(k);
                                indexed.into_iter().map(|(i, _)| i).collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// Detect attention heads that strongly attend to specific patterns
    pub fn classify_head_type(&self, head: usize) -> HeadType {
        let pattern = self.head_pattern(0, head);
        let seq_len = pattern.shape()[0];

        // Check for different attention patterns
        let mut prev_token_score = 0.0;
        let mut _induction_score = 0.0;
        let mut bos_score = 0.0;
        let mut uniform_score = 0.0;

        for i in 0..seq_len {
            // Previous token attention
            if i > 0 {
                prev_token_score += pattern[[i, i - 1]];
            }

            // BOS/first token attention
            bos_score += pattern[[i, 0]];

            // Check uniformity
            let row = pattern.row(i);
            let mean = row.mean().unwrap_or(0.0);
            let variance: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / seq_len as f32;
            if variance < 0.01 {
                uniform_score += 1.0;
            }
        }

        prev_token_score /= seq_len as f32;
        bos_score /= seq_len as f32;
        uniform_score /= seq_len as f32;

        // Classify based on scores
        if prev_token_score > 0.5 {
            HeadType::PreviousToken
        } else if bos_score > 0.5 {
            HeadType::BeginningOfSequence
        } else if uniform_score > 0.5 {
            HeadType::Uniform
        } else {
            HeadType::Other
        }
    }
}

/// Types of attention head behaviors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HeadType {
    /// Attends primarily to the previous token
    PreviousToken,
    /// Attends to beginning of sequence (often BOS token)
    BeginningOfSequence,
    /// Attends uniformly across positions
    Uniform,
    /// Induction head (copies patterns seen before)
    Induction,
    /// Duplicate token detection
    DuplicateToken,
    /// IOI: Name Mover head (moves IO name to final position)
    NameMover,
    /// IOI: S-Inhibition head (inhibits subject from being copied)
    SInhibition,
    /// IOI: Backup Name Mover (redundant name mover)
    BackupNameMover,
    /// Other/unclassified pattern
    Other,
}

/// Analysis results for attention heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadAnalysis {
    /// Head identifier (layer, head)
    pub location: (usize, usize),
    /// Classified type
    pub head_type: HeadType,
    /// Importance score (based on attention entropy)
    pub importance: f32,
    /// Average attention entropy
    pub avg_entropy: f32,
    /// Sparsity (fraction of near-zero attention weights)
    pub sparsity: f32,
}

/// Analyzer for attention patterns across the model
pub struct AttentionAnalyzer {
    /// All attention patterns by layer
    patterns: HashMap<usize, AttentionPattern>,
}

impl AttentionAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    /// Add an attention pattern
    pub fn add_pattern(&mut self, pattern: AttentionPattern) {
        self.patterns.insert(pattern.layer, pattern);
    }

    /// Analyze all heads
    pub fn analyze_all_heads(&self) -> Vec<HeadAnalysis> {
        let mut analyses = Vec::new();

        for (layer, pattern) in &self.patterns {
            let num_heads = pattern.shape[1];
            let entropy = pattern.entropy();

            for head in 0..num_heads {
                let head_type = pattern.classify_head_type(head);

                // Compute average entropy for this head
                let head_entropy = entropy.index_axis(Axis(1), head);
                let avg_entropy = head_entropy.mean().unwrap_or(0.0);

                // Compute sparsity
                let head_pattern = pattern.head_pattern(0, head);
                let sparsity = head_pattern.iter().filter(|&&x| x < 0.01).count() as f32
                    / head_pattern.len() as f32;

                // Importance based on entropy (lower entropy = more focused = often more important)
                let importance = 1.0 / (1.0 + avg_entropy);

                analyses.push(HeadAnalysis {
                    location: (*layer, head),
                    head_type,
                    importance,
                    avg_entropy,
                    sparsity,
                });
            }
        }

        // Sort by importance (use total_cmp for robust NaN handling)
        analyses.sort_by(|a, b| b.importance.total_cmp(&a.importance));
        analyses
    }

    /// Find induction heads (heads that copy patterns)
    pub fn find_induction_heads(&self) -> Vec<(usize, usize)> {
        let mut induction_heads = Vec::new();

        for (layer, pattern) in &self.patterns {
            let num_heads = pattern.shape[1];

            for head in 0..num_heads {
                if self.is_induction_head(pattern, head) {
                    induction_heads.push((*layer, head));
                }
            }
        }

        induction_heads
    }

    /// Check if a head exhibits induction behavior
    fn is_induction_head(&self, pattern: &AttentionPattern, head: usize) -> bool {
        let head_pattern = pattern.head_pattern(0, head);
        let seq_len = head_pattern.shape()[0];

        if seq_len < 4 {
            return false;
        }

        // Induction heads attend to positions where the previous token matches
        // the current previous token. This creates a diagonal stripe pattern
        // offset by the sequence repeat.
        //
        // For now, use a simplified heuristic: check for strong off-diagonal attention

        let mut off_diag_score = 0.0;
        let mut count = 0;

        for i in 2..seq_len {
            for j in 1..i {
                if j < i - 1 {
                    off_diag_score += head_pattern[[i, j]];
                    count += 1;
                }
            }
        }

        if count > 0 {
            off_diag_score /= count as f32;
        }

        // Induction heads typically have moderate off-diagonal attention
        off_diag_score > 0.1 && off_diag_score < 0.5
    }

    /// Compute attention flow between token positions
    pub fn compute_attention_flow(&self) -> Array2<f32> {
        // Aggregate attention across all layers
        let first_pattern = match self.patterns.values().next() {
            Some(p) => p,
            None => return Array2::zeros((0, 0)),
        };

        let seq_len = first_pattern.shape[2];
        let mut flow = Array2::zeros((seq_len, seq_len));

        for pattern in self.patterns.values() {
            let arr = pattern.as_array();
            // Average across batch and heads, handling potential empty arrays
            if let Some(mean_batch) = arr.mean_axis(Axis(0)) {
                if let Some(layer_flow) = mean_batch.mean_axis(Axis(0)) {
                    flow = flow + &layer_flow;
                }
            }
        }

        // Normalize (guard against division by zero)
        let num_layers = self.patterns.len() as f32;
        if num_layers > 0.0 {
            flow /= num_layers;
        }
        flow
    }
}

impl Default for AttentionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_attention_pattern_creation() {
        let data = Array4::ones((1, 8, 10, 10)) / 10.0; // Uniform attention
        let pattern = AttentionPattern::new(0, data);

        assert_eq!(pattern.layer, 0);
        assert_eq!(pattern.shape, vec![1, 8, 10, 10]);
    }

    #[test]
    fn test_head_classification() {
        // Create a "previous token" attention pattern
        let mut data = Array4::zeros((1, 1, 10, 10));
        for i in 1..10 {
            data[[0, 0, i, i - 1]] = 1.0;
        }
        data[[0, 0, 0, 0]] = 1.0; // First token attends to itself

        let pattern = AttentionPattern::new(0, data);
        assert_eq!(pattern.classify_head_type(0), HeadType::PreviousToken);
    }

    #[test]
    fn test_entropy_computation() {
        // Uniform attention should have high entropy
        let data = Array4::ones((1, 1, 4, 4)) / 4.0;
        let pattern = AttentionPattern::new(0, data);
        let entropy = pattern.entropy();

        // Entropy of uniform distribution over 4 items is ln(4) ≈ 1.386
        assert!(entropy[[0, 0, 0]] > 1.0);
    }
}
