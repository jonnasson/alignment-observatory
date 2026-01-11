/**
 * Activation trace types for transformer interpretability
 */

import type { TensorData, TensorStats, TensorMetadata } from './tensor.types'

/** Component types in transformer architecture */
export type ComponentType = 'residual' | 'attn_out' | 'mlp_out' | 'embed' | 'unembed'

/** A complete activation trace from a forward pass */
export interface ActivationTrace {
  /** Unique identifier for this trace */
  traceId: string
  /** Model architecture name */
  architecture: string
  /** Input token IDs */
  inputTokens: number[]
  /** Token strings (decoded) */
  tokenStrings?: string[]
  /** Layers that have captured activations */
  layers: number[]
  /** Components captured per layer */
  components: ComponentType[]
  /** Creation timestamp */
  createdAt: string
  /** Model configuration */
  modelConfig?: ModelConfig
}

/** Model configuration metadata */
export interface ModelConfig {
  numLayers: number
  numHeads: number
  hiddenSize: number
  vocabSize: number
  maxSeqLen: number
  architecture: string
}

/** Activation data for a specific layer and component */
export interface LayerActivation {
  /** Layer index */
  layer: number
  /** Component type */
  component: ComponentType
  /** Shape: [batch, seq, hidden] */
  shape: [number, number, number]
  /** Tensor data (may be lazy-loaded) */
  data?: TensorData
  /** Metadata for lazy loading */
  metadata?: TensorMetadata
  /** Computed statistics */
  stats?: TensorStats
}

/** Token-level activation norms across layers */
export interface TokenNorms {
  /** Token index in sequence */
  tokenIndex: number
  /** Token string */
  tokenString: string
  /** Norm values per layer and component */
  norms: Record<string, number[]>
}

/** Request to fetch activation data */
export interface ActivationRequest {
  traceId: string
  layer: number
  component: ComponentType
  /** Optional: downsample factor for visualization */
  downsample?: number
  /** Optional: format preference */
  format?: 'json' | 'binary'
}

/** Response containing activation data */
export interface ActivationResponse {
  layer: number
  component: ComponentType
  shape: number[]
  dtype: string
  /** Inline data for small tensors */
  data?: number[]
  /** URL to fetch binary data for large tensors */
  dataUrl?: string
  stats: TensorStats
}

/** Residual stream analysis at a position */
export interface ResidualAnalysis {
  layer: number
  position: number
  /** Contribution from attention */
  attentionContribution: number
  /** Contribution from MLP */
  mlpContribution: number
  /** Cumulative residual norm */
  residualNorm: number
  /** Top contributing dimensions */
  topDimensions: [number, number][]
}

/** Request to create a new trace via inference */
export interface CreateTraceRequest {
  /** Model ID (HuggingFace format) */
  modelId: string
  /** Input text to trace */
  inputText: string
  /** Components to capture */
  captureComponents?: ComponentType[]
  /** Layers to capture (empty = all) */
  captureLayers?: number[]
  /** Use streaming storage for large models */
  streaming?: boolean
}

/** Request to load a pre-computed trace from disk */
export interface LoadTraceRequest {
  /** Path to trace file */
  path: string
}
