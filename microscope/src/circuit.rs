//! Circuit Discovery Module
//!
//! This module provides tools for automatically discovering computational
//! circuits in transformer models. Based on research from:
//! - Anthropic's "A Mathematical Framework for Transformer Circuits"
//! - "In-context Learning and Induction Heads"
//!
//! Key capabilities:
//! - Automated circuit identification
//! - Edge importance scoring
//! - Minimal circuit extraction

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::activation::ActivationTrace;
use crate::attention::AttentionPattern;

/// Represents a node in a computational circuit
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CircuitNode {
    /// Layer index
    pub layer: usize,
    /// Component type
    pub component: ComponentType,
    /// Head index (for attention components)
    pub head: Option<usize>,
    /// Token position (if position-specific)
    pub position: Option<usize>,
}

/// Types of components in a transformer circuit
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    Embedding,
    AttentionHead,
    MLP,
    LayerNorm,
    Residual,
    Unembedding,
}

impl std::fmt::Display for CircuitNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.component {
            ComponentType::AttentionHead => {
                write!(
                    f,
                    "L{}H{}",
                    self.layer,
                    self.head.unwrap_or(0)
                )
            }
            ComponentType::MLP => write!(f, "L{}MLP", self.layer),
            ComponentType::Embedding => write!(f, "Embed"),
            ComponentType::Unembedding => write!(f, "Unembed"),
            ComponentType::LayerNorm => write!(f, "L{}LN", self.layer),
            ComponentType::Residual => write!(f, "L{}Res", self.layer),
        }
    }
}

/// Represents an edge in the circuit graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitEdge {
    /// Source node
    pub from: CircuitNode,
    /// Target node
    pub to: CircuitNode,
    /// Edge importance score (0 to 1)
    pub importance: f32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// A complete computational circuit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circuit {
    /// Human-readable name for this circuit
    pub name: String,
    /// Description of what this circuit computes
    pub description: String,
    /// All nodes in the circuit
    pub nodes: HashSet<CircuitNode>,
    /// All edges with importance scores
    pub edges: Vec<CircuitEdge>,
    /// The behavior this circuit implements
    pub behavior: String,
}

impl Circuit {
    /// Create a new empty circuit
    pub fn new(name: &str, description: &str, behavior: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            nodes: HashSet::new(),
            edges: Vec::new(),
            behavior: behavior.to_string(),
        }
    }

    /// Add a node to the circuit
    pub fn add_node(&mut self, node: CircuitNode) {
        self.nodes.insert(node);
    }

    /// Add an edge to the circuit
    pub fn add_edge(&mut self, edge: CircuitEdge) {
        self.nodes.insert(edge.from.clone());
        self.nodes.insert(edge.to.clone());
        self.edges.push(edge);
    }

    /// Get the minimal circuit (remove edges below importance threshold)
    pub fn minimal(&self, threshold: f32) -> Circuit {
        let mut minimal = Circuit::new(&self.name, &self.description, &self.behavior);

        for edge in &self.edges {
            if edge.importance >= threshold {
                minimal.add_edge(edge.clone());
            }
        }

        minimal
    }

    /// Export circuit to DOT format for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph Circuit {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add nodes
        for node in &self.nodes {
            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\"];\n",
                node.to_string(),
                node.to_string()
            ));
        }

        dot.push_str("\n");

        // Add edges
        for edge in &self.edges {
            let width = 1.0 + edge.importance * 3.0;
            dot.push_str(&format!(
                "  \"{}\" -> \"{}\" [penwidth={:.1}, label=\"{:.2}\"];\n",
                edge.from.to_string(),
                edge.to.to_string(),
                width,
                edge.importance
            ));
        }

        dot.push_str("}\n");
        dot
    }

    /// Get total circuit importance (sum of edge importances)
    pub fn total_importance(&self) -> f32 {
        self.edges.iter().map(|e| e.importance).sum()
    }

    /// Get average edge importance
    pub fn average_importance(&self) -> f32 {
        if self.edges.is_empty() {
            return 0.0;
        }
        self.total_importance() / self.edges.len() as f32
    }
}

/// Circuit discovery engine
pub struct CircuitDiscoverer {
    /// Activation traces for analysis
    traces: Vec<ActivationTrace>,
    /// Attention patterns by layer
    attention: HashMap<usize, AttentionPattern>,
    /// Discovered circuits
    circuits: Vec<Circuit>,
}

impl CircuitDiscoverer {
    /// Create a new circuit discoverer
    pub fn new() -> Self {
        Self {
            traces: Vec::new(),
            attention: HashMap::new(),
            circuits: Vec::new(),
        }
    }

    /// Add an activation trace for analysis
    pub fn add_trace(&mut self, trace: ActivationTrace) {
        self.traces.push(trace);
    }

    /// Add attention patterns
    pub fn add_attention(&mut self, layer: usize, pattern: AttentionPattern) {
        self.attention.insert(layer, pattern);
    }

    /// Discover the circuit for a specific behavior using activation patching
    pub fn discover_circuit(
        &mut self,
        behavior_name: &str,
        clean_trace: &ActivationTrace,
        corrupt_trace: &ActivationTrace,
        metric_fn: impl Fn(&ActivationTrace) -> f32,
    ) -> Circuit {
        let mut circuit = Circuit::new(
            behavior_name,
            &format!("Auto-discovered circuit for {}", behavior_name),
            behavior_name,
        );

        let clean_metric = metric_fn(clean_trace);
        let corrupt_metric = metric_fn(corrupt_trace);
        let metric_diff = (clean_metric - corrupt_metric).abs();

        if metric_diff < 1e-6 {
            return circuit; // No difference, empty circuit
        }

        // Find layers where activation differences matter most
        let mut layer_importance: HashMap<usize, f32> = HashMap::new();

        for (key, clean_act) in &clean_trace.activations {
            if let Some(corrupt_act) = corrupt_trace.activations.get(key) {
                let clean_arr = clean_act.as_array();
                let corrupt_arr = corrupt_act.as_array();

                // Compute L2 difference
                let diff: f32 = clean_arr
                    .iter()
                    .zip(corrupt_arr.iter())
                    .map(|(c, co)| (c - co).powi(2))
                    .sum::<f32>()
                    .sqrt();

                // Normalize by size
                let normalized_diff = diff / clean_arr.len() as f32;

                *layer_importance.entry(clean_act.layer).or_insert(0.0) += normalized_diff;
            }
        }

        // Create nodes and edges based on importance
        let max_importance = layer_importance.values().cloned().fold(0.0f32, f32::max);

        if max_importance > 0.0 {
            let mut prev_node: Option<CircuitNode> = None;

            // Sort by layer
            let mut layers: Vec<_> = layer_importance.into_iter().collect();
            layers.sort_by_key(|(l, _)| *l);

            for (layer, importance) in layers {
                let normalized_importance = importance / max_importance;

                if normalized_importance > 0.1 {
                    // Threshold for inclusion
                    let node = CircuitNode {
                        layer,
                        component: ComponentType::Residual,
                        head: None,
                        position: None,
                    };

                    circuit.add_node(node.clone());

                    if let Some(prev) = prev_node {
                        circuit.add_edge(CircuitEdge {
                            from: prev,
                            to: node.clone(),
                            importance: normalized_importance,
                            metadata: HashMap::new(),
                        });
                    }

                    prev_node = Some(node);
                }
            }
        }

        self.circuits.push(circuit.clone());
        circuit
    }

    /// Discover known circuit patterns (induction, copying, etc.)
    pub fn discover_known_patterns(&self) -> Vec<Circuit> {
        let mut patterns = Vec::new();

        // Look for induction circuits
        if let Some(induction) = self.find_induction_circuit() {
            patterns.push(induction);
        }

        patterns
    }

    /// Search for induction head circuits
    fn find_induction_circuit(&self) -> Option<Circuit> {
        // Induction circuits typically involve:
        // 1. A "previous token" head in an early layer
        // 2. An "induction" head in a later layer that copies
        //
        // This is a simplified implementation

        let mut circuit = Circuit::new(
            "Induction",
            "Circuit for in-context learning via pattern copying",
            "copy_previous_pattern",
        );

        // Find potential previous token heads
        let mut prev_token_heads = Vec::new();
        let mut induction_heads = Vec::new();

        for (&layer, pattern) in &self.attention {
            let num_heads = pattern.shape[1];

            for head in 0..num_heads {
                let head_type = pattern.classify_head_type(head);

                match head_type {
                    crate::attention::HeadType::PreviousToken => {
                        prev_token_heads.push((layer, head));
                    }
                    crate::attention::HeadType::Induction => {
                        induction_heads.push((layer, head));
                    }
                    _ => {}
                }
            }
        }

        // Build circuit if we found both components
        if !prev_token_heads.is_empty() && !induction_heads.is_empty() {
            // Add previous token heads
            for (layer, head) in &prev_token_heads {
                circuit.add_node(CircuitNode {
                    layer: *layer,
                    component: ComponentType::AttentionHead,
                    head: Some(*head),
                    position: None,
                });
            }

            // Add induction heads
            for (layer, head) in &induction_heads {
                circuit.add_node(CircuitNode {
                    layer: *layer,
                    component: ComponentType::AttentionHead,
                    head: Some(*head),
                    position: None,
                });
            }

            // Connect them
            for (pt_layer, pt_head) in &prev_token_heads {
                for (ind_layer, ind_head) in &induction_heads {
                    if ind_layer > pt_layer {
                        circuit.add_edge(CircuitEdge {
                            from: CircuitNode {
                                layer: *pt_layer,
                                component: ComponentType::AttentionHead,
                                head: Some(*pt_head),
                                position: None,
                            },
                            to: CircuitNode {
                                layer: *ind_layer,
                                component: ComponentType::AttentionHead,
                                head: Some(*ind_head),
                                position: None,
                            },
                            importance: 0.8,
                            metadata: HashMap::new(),
                        });
                    }
                }
            }

            return Some(circuit);
        }

        None
    }

    /// Get all discovered circuits
    pub fn circuits(&self) -> &[Circuit] {
        &self.circuits
    }
}

impl Default for CircuitDiscoverer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// IOI (Indirect Object Identification) Circuit Detection
// Based on Wang et al. 2022: "Interpretability in the Wild"
// ============================================================================

/// Token roles in an IOI sentence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IOITokenRole {
    /// Subject name (e.g., "Mary" in "Mary gave to John")
    Subject,
    /// Indirect object name (e.g., "John" in "Mary gave to John")
    IndirectObject,
    /// Second occurrence of subject name
    SubjectRepeat,
    /// The token at the end position (where prediction happens)
    EndPosition,
    /// Other tokens
    Other,
}

/// Parsed IOI sentence with token role annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOISentence {
    /// Token IDs
    pub tokens: Vec<u32>,
    /// Token strings
    pub token_strings: Vec<String>,
    /// Position of each token role
    pub token_positions: HashMap<IOITokenRole, Vec<usize>>,
    /// The correct answer (IO name)
    pub correct_answer: String,
    /// The distractor (subject name)
    pub distractor: String,
}

impl IOISentence {
    /// Create a new IOI sentence with manual position annotations
    pub fn new(
        tokens: Vec<u32>,
        token_strings: Vec<String>,
        subject_positions: Vec<usize>,
        io_position: usize,
        subject2_position: usize,
        end_position: usize,
        correct_answer: String,
        distractor: String,
    ) -> Self {
        let mut token_positions = HashMap::new();
        token_positions.insert(IOITokenRole::Subject, subject_positions);
        token_positions.insert(IOITokenRole::IndirectObject, vec![io_position]);
        token_positions.insert(IOITokenRole::SubjectRepeat, vec![subject2_position]);
        token_positions.insert(IOITokenRole::EndPosition, vec![end_position]);

        Self {
            tokens,
            token_strings,
            token_positions,
            correct_answer,
            distractor,
        }
    }

    /// Get the IO position
    pub fn io_position(&self) -> Option<usize> {
        self.token_positions
            .get(&IOITokenRole::IndirectObject)
            .and_then(|v| v.first().copied())
    }

    /// Get the subject repeat (S2) position
    pub fn s2_position(&self) -> Option<usize> {
        self.token_positions
            .get(&IOITokenRole::SubjectRepeat)
            .and_then(|v| v.first().copied())
    }

    /// Get the end position
    pub fn end_position(&self) -> Option<usize> {
        self.token_positions
            .get(&IOITokenRole::EndPosition)
            .and_then(|v| v.first().copied())
    }
}

/// Configuration for IOI circuit detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOIDetectionConfig {
    /// Minimum attention score to consider a head as a Name Mover
    pub name_mover_threshold: f32,
    /// Minimum inhibition score for S-Inhibition heads
    pub s_inhibition_threshold: f32,
    /// Number of top heads to return per component type
    pub top_k_heads: usize,
    /// Layer ranges to search for each component type
    pub layer_ranges: HashMap<String, (usize, usize)>,
}

impl Default for IOIDetectionConfig {
    fn default() -> Self {
        let mut layer_ranges = HashMap::new();
        // Based on GPT-2 small findings from the paper
        layer_ranges.insert("duplicate_token".to_string(), (0, 3));
        layer_ranges.insert("previous_token".to_string(), (0, 3));
        layer_ranges.insert("induction".to_string(), (4, 7));
        layer_ranges.insert("s_inhibition".to_string(), (6, 9));
        layer_ranges.insert("name_mover".to_string(), (9, 12));
        layer_ranges.insert("backup_name_mover".to_string(), (9, 12));

        Self {
            name_mover_threshold: 0.3,
            s_inhibition_threshold: 0.2,
            top_k_heads: 5,
            layer_ranges,
        }
    }
}

/// Detected IOI head with scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOIHead {
    /// Layer index
    pub layer: usize,
    /// Head index
    pub head: usize,
    /// Component type (name_mover, s_inhibition, etc.)
    pub component_type: String,
    /// Detection confidence score (0-1)
    pub score: f32,
    /// Additional diagnostic metrics
    pub metrics: HashMap<String, f32>,
}

impl IOIHead {
    /// Create a new IOI head
    pub fn new(layer: usize, head: usize, component_type: &str, score: f32) -> Self {
        Self {
            layer,
            head,
            component_type: component_type.to_string(),
            score,
            metrics: HashMap::new(),
        }
    }

    /// Add a metric
    pub fn with_metric(mut self, key: &str, value: f32) -> Self {
        self.metrics.insert(key.to_string(), value);
        self
    }
}

/// Complete IOI circuit detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOICircuitResult {
    /// The constructed circuit
    pub circuit: Circuit,
    /// All detected Name Mover heads
    pub name_mover_heads: Vec<IOIHead>,
    /// All detected S-Inhibition heads
    pub s_inhibition_heads: Vec<IOIHead>,
    /// All detected Duplicate Token heads
    pub duplicate_token_heads: Vec<IOIHead>,
    /// All detected Previous Token heads
    pub previous_token_heads: Vec<IOIHead>,
    /// Backup Name Mover heads
    pub backup_name_mover_heads: Vec<IOIHead>,
    /// Overall circuit validity score
    pub validity_score: f32,
    /// The IOI sentence used for detection
    pub sentence: IOISentence,
}

/// Known IOI heads from the original paper (GPT-2 small)
pub struct KnownIOIHeads;

impl KnownIOIHeads {
    /// Get known Name Mover heads for GPT-2 small
    pub fn name_movers_gpt2() -> Vec<(usize, usize)> {
        vec![(9, 9), (10, 0), (9, 6)]
    }

    /// Get known Backup Name Mover heads for GPT-2 small
    pub fn backup_name_movers_gpt2() -> Vec<(usize, usize)> {
        vec![(10, 10), (10, 6), (10, 2), (11, 2), (9, 7), (10, 1)]
    }

    /// Get known S-Inhibition heads for GPT-2 small
    pub fn s_inhibition_gpt2() -> Vec<(usize, usize)> {
        vec![(7, 3), (7, 9), (8, 6), (8, 10)]
    }

    /// Get known Duplicate Token heads for GPT-2 small
    pub fn duplicate_token_gpt2() -> Vec<(usize, usize)> {
        vec![(0, 1), (0, 10), (3, 0)]
    }

    /// Get all known heads as a map
    pub fn all_gpt2() -> HashMap<String, Vec<(usize, usize)>> {
        let mut known = HashMap::new();
        known.insert("name_mover".to_string(), Self::name_movers_gpt2());
        known.insert("backup_name_mover".to_string(), Self::backup_name_movers_gpt2());
        known.insert("s_inhibition".to_string(), Self::s_inhibition_gpt2());
        known.insert("duplicate_token".to_string(), Self::duplicate_token_gpt2());
        known
    }
}

impl CircuitDiscoverer {
    /// Discover the IOI circuit for a given sentence
    pub fn find_ioi_circuit(
        &self,
        sentence: &IOISentence,
        config: &IOIDetectionConfig,
    ) -> IOICircuitResult {
        let mut circuit = Circuit::new(
            "IOI",
            "Indirect Object Identification circuit (Wang et al. 2022)",
            "predict_indirect_object",
        );

        // Detect each component type
        let name_mover_heads = self.find_name_mover_heads(sentence, config);
        let s_inhibition_heads = self.find_s_inhibition_heads(sentence, config);
        let duplicate_token_heads = self.find_duplicate_token_heads(sentence, config);
        let previous_token_heads = self.find_previous_token_heads(config);
        let backup_name_mover_heads = self.find_backup_name_mover_heads(sentence, config);

        // Build circuit nodes
        for head in &name_mover_heads {
            circuit.add_node(CircuitNode {
                layer: head.layer,
                component: ComponentType::AttentionHead,
                head: Some(head.head),
                position: None,
            });
        }

        for head in &s_inhibition_heads {
            circuit.add_node(CircuitNode {
                layer: head.layer,
                component: ComponentType::AttentionHead,
                head: Some(head.head),
                position: None,
            });
        }

        for head in &duplicate_token_heads {
            circuit.add_node(CircuitNode {
                layer: head.layer,
                component: ComponentType::AttentionHead,
                head: Some(head.head),
                position: None,
            });
        }

        // Connect duplicate token → s_inhibition → name_mover
        for dt_head in &duplicate_token_heads {
            for si_head in &s_inhibition_heads {
                if si_head.layer > dt_head.layer {
                    let importance = (dt_head.score + si_head.score) / 2.0;
                    circuit.add_edge(CircuitEdge {
                        from: CircuitNode {
                            layer: dt_head.layer,
                            component: ComponentType::AttentionHead,
                            head: Some(dt_head.head),
                            position: None,
                        },
                        to: CircuitNode {
                            layer: si_head.layer,
                            component: ComponentType::AttentionHead,
                            head: Some(si_head.head),
                            position: None,
                        },
                        importance,
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        for si_head in &s_inhibition_heads {
            for nm_head in &name_mover_heads {
                if nm_head.layer > si_head.layer {
                    let importance = (si_head.score + nm_head.score) / 2.0;
                    circuit.add_edge(CircuitEdge {
                        from: CircuitNode {
                            layer: si_head.layer,
                            component: ComponentType::AttentionHead,
                            head: Some(si_head.head),
                            position: None,
                        },
                        to: CircuitNode {
                            layer: nm_head.layer,
                            component: ComponentType::AttentionHead,
                            head: Some(nm_head.head),
                            position: None,
                        },
                        importance,
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        // Compute validity score
        let validity_score = self.compute_ioi_validity_score(
            &name_mover_heads,
            &s_inhibition_heads,
            &duplicate_token_heads,
        );

        IOICircuitResult {
            circuit,
            name_mover_heads,
            s_inhibition_heads,
            duplicate_token_heads,
            previous_token_heads,
            backup_name_mover_heads,
            validity_score,
            sentence: sentence.clone(),
        }
    }

    /// Find Name Mover heads by attention pattern analysis
    fn find_name_mover_heads(
        &self,
        sentence: &IOISentence,
        config: &IOIDetectionConfig,
    ) -> Vec<IOIHead> {
        let mut heads = Vec::new();

        let io_pos = match sentence.io_position() {
            Some(p) => p,
            None => return heads,
        };
        let end_pos = match sentence.end_position() {
            Some(p) => p,
            None => return heads,
        };

        let (min_layer, max_layer) = config
            .layer_ranges
            .get("name_mover")
            .copied()
            .unwrap_or((9, 12));

        for (&layer, pattern) in &self.attention {
            if layer < min_layer || layer >= max_layer {
                continue;
            }

            let num_heads = pattern.shape[1];
            for head in 0..num_heads {
                let head_pattern = pattern.head_pattern(0, head);

                // Name Mover: high attention from END to IO position
                let end_to_io_attention = if end_pos < head_pattern.shape()[0]
                    && io_pos < head_pattern.shape()[1]
                {
                    head_pattern[[end_pos, io_pos]]
                } else {
                    0.0
                };

                if end_to_io_attention > config.name_mover_threshold {
                    heads.push(
                        IOIHead::new(layer, head, "name_mover", end_to_io_attention)
                            .with_metric("end_to_io_attention", end_to_io_attention),
                    );
                }
            }
        }

        // Sort by score and take top-k
        heads.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        heads.truncate(config.top_k_heads);
        heads
    }

    /// Find S-Inhibition heads
    fn find_s_inhibition_heads(
        &self,
        sentence: &IOISentence,
        config: &IOIDetectionConfig,
    ) -> Vec<IOIHead> {
        let mut heads = Vec::new();

        let s2_pos = match sentence.s2_position() {
            Some(p) => p,
            None => return heads,
        };
        let end_pos = match sentence.end_position() {
            Some(p) => p,
            None => return heads,
        };

        let (min_layer, max_layer) = config
            .layer_ranges
            .get("s_inhibition")
            .copied()
            .unwrap_or((6, 9));

        for (&layer, pattern) in &self.attention {
            if layer < min_layer || layer >= max_layer {
                continue;
            }

            let num_heads = pattern.shape[1];
            for head in 0..num_heads {
                let head_pattern = pattern.head_pattern(0, head);

                // S-Inhibition: attend from END to S2 position
                let end_to_s2_attention = if end_pos < head_pattern.shape()[0]
                    && s2_pos < head_pattern.shape()[1]
                {
                    head_pattern[[end_pos, s2_pos]]
                } else {
                    0.0
                };

                if end_to_s2_attention > config.s_inhibition_threshold {
                    heads.push(
                        IOIHead::new(layer, head, "s_inhibition", end_to_s2_attention)
                            .with_metric("end_to_s2_attention", end_to_s2_attention),
                    );
                }
            }
        }

        heads.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        heads.truncate(config.top_k_heads);
        heads
    }

    /// Find Duplicate Token heads
    fn find_duplicate_token_heads(
        &self,
        sentence: &IOISentence,
        config: &IOIDetectionConfig,
    ) -> Vec<IOIHead> {
        let mut heads = Vec::new();

        let subject_positions = sentence
            .token_positions
            .get(&IOITokenRole::Subject)
            .cloned()
            .unwrap_or_default();

        if subject_positions.is_empty() {
            return heads;
        }

        let s1_pos = subject_positions[0];
        let s2_pos = match sentence.s2_position() {
            Some(p) => p,
            None => return heads,
        };

        let (min_layer, max_layer) = config
            .layer_ranges
            .get("duplicate_token")
            .copied()
            .unwrap_or((0, 3));

        for (&layer, pattern) in &self.attention {
            if layer < min_layer || layer >= max_layer {
                continue;
            }

            let num_heads = pattern.shape[1];
            for head in 0..num_heads {
                let head_pattern = pattern.head_pattern(0, head);

                // Duplicate Token: high attention from S2 to S1
                let s2_to_s1_attention = if s2_pos < head_pattern.shape()[0]
                    && s1_pos < head_pattern.shape()[1]
                {
                    head_pattern[[s2_pos, s1_pos]]
                } else {
                    0.0
                };

                if s2_to_s1_attention > 0.2 {
                    heads.push(
                        IOIHead::new(layer, head, "duplicate_token", s2_to_s1_attention)
                            .with_metric("s2_to_s1_attention", s2_to_s1_attention),
                    );
                }
            }
        }

        heads.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        heads.truncate(config.top_k_heads);
        heads
    }

    /// Find Previous Token heads (using existing classification)
    fn find_previous_token_heads(&self, config: &IOIDetectionConfig) -> Vec<IOIHead> {
        let mut heads = Vec::new();

        let (min_layer, max_layer) = config
            .layer_ranges
            .get("previous_token")
            .copied()
            .unwrap_or((0, 3));

        for (&layer, pattern) in &self.attention {
            if layer < min_layer || layer >= max_layer {
                continue;
            }

            let num_heads = pattern.shape[1];
            for head in 0..num_heads {
                let head_type = pattern.classify_head_type(head);
                if head_type == crate::attention::HeadType::PreviousToken {
                    heads.push(IOIHead::new(layer, head, "previous_token", 0.8));
                }
            }
        }

        heads.truncate(config.top_k_heads);
        heads
    }

    /// Find Backup Name Mover heads
    fn find_backup_name_mover_heads(
        &self,
        sentence: &IOISentence,
        config: &IOIDetectionConfig,
    ) -> Vec<IOIHead> {
        let mut heads = Vec::new();

        let io_pos = match sentence.io_position() {
            Some(p) => p,
            None => return heads,
        };
        let end_pos = match sentence.end_position() {
            Some(p) => p,
            None => return heads,
        };

        let (min_layer, max_layer) = config
            .layer_ranges
            .get("backup_name_mover")
            .copied()
            .unwrap_or((9, 12));

        // Same detection as name_mover but with lower threshold
        let backup_threshold = config.name_mover_threshold * 0.7;

        for (&layer, pattern) in &self.attention {
            if layer < min_layer || layer >= max_layer {
                continue;
            }

            let num_heads = pattern.shape[1];
            for head in 0..num_heads {
                let head_pattern = pattern.head_pattern(0, head);

                let end_to_io_attention = if end_pos < head_pattern.shape()[0]
                    && io_pos < head_pattern.shape()[1]
                {
                    head_pattern[[end_pos, io_pos]]
                } else {
                    0.0
                };

                // Backup name movers have moderate attention
                if end_to_io_attention > backup_threshold
                    && end_to_io_attention < config.name_mover_threshold
                {
                    heads.push(
                        IOIHead::new(layer, head, "backup_name_mover", end_to_io_attention)
                            .with_metric("end_to_io_attention", end_to_io_attention),
                    );
                }
            }
        }

        heads.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        heads.truncate(config.top_k_heads);
        heads
    }

    /// Compute validity score for the detected IOI circuit
    fn compute_ioi_validity_score(
        &self,
        name_mover_heads: &[IOIHead],
        s_inhibition_heads: &[IOIHead],
        duplicate_token_heads: &[IOIHead],
    ) -> f32 {
        // Circuit is valid if we found at least one of each key component
        let has_name_mover = !name_mover_heads.is_empty();
        let has_s_inhibition = !s_inhibition_heads.is_empty();
        let has_duplicate_token = !duplicate_token_heads.is_empty();

        let mut score = 0.0;
        if has_name_mover {
            score += 0.4;
        }
        if has_s_inhibition {
            score += 0.3;
        }
        if has_duplicate_token {
            score += 0.3;
        }

        // Bonus for strong scores
        if let Some(nm) = name_mover_heads.first() {
            score += nm.score * 0.1;
        }
        if let Some(si) = s_inhibition_heads.first() {
            score += si.score * 0.1;
        }

        score.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_creation() {
        let mut circuit = Circuit::new("Test", "Test circuit", "test_behavior");

        let node1 = CircuitNode {
            layer: 0,
            component: ComponentType::AttentionHead,
            head: Some(5),
            position: None,
        };

        let node2 = CircuitNode {
            layer: 1,
            component: ComponentType::MLP,
            head: None,
            position: None,
        };

        circuit.add_edge(CircuitEdge {
            from: node1,
            to: node2,
            importance: 0.9,
            metadata: HashMap::new(),
        });

        assert_eq!(circuit.nodes.len(), 2);
        assert_eq!(circuit.edges.len(), 1);
    }

    #[test]
    fn test_minimal_circuit() {
        let mut circuit = Circuit::new("Test", "Test", "test");

        let node1 = CircuitNode {
            layer: 0,
            component: ComponentType::Embedding,
            head: None,
            position: None,
        };

        let node2 = CircuitNode {
            layer: 1,
            component: ComponentType::MLP,
            head: None,
            position: None,
        };

        let node3 = CircuitNode {
            layer: 2,
            component: ComponentType::Unembedding,
            head: None,
            position: None,
        };

        // High importance edge
        circuit.add_edge(CircuitEdge {
            from: node1.clone(),
            to: node2.clone(),
            importance: 0.9,
            metadata: HashMap::new(),
        });

        // Low importance edge
        circuit.add_edge(CircuitEdge {
            from: node2,
            to: node3,
            importance: 0.1,
            metadata: HashMap::new(),
        });

        let minimal = circuit.minimal(0.5);
        assert_eq!(minimal.edges.len(), 1);
        assert_eq!(minimal.edges[0].importance, 0.9);
    }

    #[test]
    fn test_to_dot() {
        let mut circuit = Circuit::new("Test", "Test", "test");

        circuit.add_node(CircuitNode {
            layer: 0,
            component: ComponentType::AttentionHead,
            head: Some(0),
            position: None,
        });

        let dot = circuit.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("L0H0"));
    }
}
