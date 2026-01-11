/**
 * Attention pattern types for transformer interpretability
 */

import type { TensorData, TensorStats } from './tensor.types'

/** Classification of attention head behavior */
export type HeadClassification =
  | 'previous_token'
  | 'bos'
  | 'uniform'
  | 'induction'
  | 'name_mover'
  | 's_inhibition'
  | 'backup_name_mover'
  | 'duplicate_token'
  | 'mixed'
  | 'other'

/** Classification object returned by API (extended format) */
export interface HeadClassificationInfo {
  category: HeadClassification
  confidence: number
  description?: string | null
}

/** Attention pattern for a single layer */
export interface AttentionPattern {
  /** Layer index (0-indexed) */
  layer: number
  /** Head index (null means all heads) */
  head: number | null
  /** Attention weights: shape [batch, heads, seq_q, seq_k] */
  pattern: TensorData<number[][][][]>
  /** Input tokens */
  tokens: string[]
  /** Number of attention heads */
  numHeads?: number
  /** Sequence length */
  seqLen?: number
  /** Statistics about the attention pattern */
  stats?: TensorStats
}

/** Analysis results for a single attention head */
export interface HeadAnalysis {
  /** Layer index */
  layer: number
  /** Head index within the layer */
  head: number
  /** Classified behavior type - can be string or object from API */
  classification: HeadClassification | HeadClassificationInfo
  /** Entropy of attention distribution (higher = more uniform) */
  entropy: number
  /** Average sparsity (fraction of near-zero weights) */
  sparsity: number
  /** Top-k attended positions per query position */
  topAttended: number[][]
  /** Additional metrics */
  metrics?: Record<string, number>
}

/** Attention patterns for all heads in a layer */
export interface LayerAttention {
  layer: number
  /** Per-head patterns: shape [heads, seq_q, seq_k] */
  patterns: number[][][]
  /** Per-head entropy: shape [heads, seq_q] */
  entropy: number[][]
  /** Head classifications */
  classifications: HeadClassification[]
}

/** Summary of attention across all layers */
export interface AttentionSummary {
  /** Total layers in the model */
  numLayers: number
  /** Total heads per layer */
  numHeads: number
  /** Average entropy per layer */
  avgEntropyPerLayer: number[]
  /** Head type distribution */
  headTypeDistribution: Record<HeadClassification, number>
  /** Most important heads by some metric */
  topHeads: HeadAnalysis[]
}

/** Request to analyze attention patterns */
export interface AttentionAnalysisRequest {
  traceId: string
  layers?: number[]
  computeEntropy?: boolean
  classifyHeads?: boolean
}

/** Response from attention analysis (batch mode - multiple layers) */
export interface AttentionAnalysisResponse {
  traceId: string
  patterns: Record<number, AttentionPattern>
  analyses?: Record<number, HeadAnalysis[]>
  stats?: TensorStats
}

/** Response from single layer attention query */
export interface AttentionResponse {
  traceId: string
  pattern: AttentionPattern
  analysis?: HeadAnalysis[]
}
