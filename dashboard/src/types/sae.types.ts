/**
 * Sparse Autoencoder (SAE) types for feature analysis
 */

import type { TensorData, TensorStats } from './tensor.types'

/** SAE activation function types */
export type SAEActivation = 'relu' | 'topk' | 'jumprelu'

/** Configuration for a Sparse Autoencoder */
export interface SAEConfig {
  /** Input dimension (model hidden size) */
  dIn: number
  /** SAE feature dimension (typically 4K-131K) */
  dSae: number
  /** Activation function */
  activation: SAEActivation
  /** k value for top-k activation */
  k?: number
  /** Hook point in the model (e.g., "blocks.5.hook_resid_post") */
  hookPoint: string
  /** Layer index */
  layer: number
}

/** Encoded SAE features for a set of activations */
export interface SAEFeatures {
  /** Feature activations: shape [positions, d_sae] or [batch, seq, d_sae] */
  activations: TensorData
  /** Sparsity (fraction of zero activations) */
  sparsity: number
  /** Mean number of active features per position */
  meanActiveFeatures: number
  /** SAE configuration used */
  config?: SAEConfig
}

/** A single feature with its activation value */
export interface FeatureActivation {
  /** Feature index in the SAE */
  featureIdx: number
  /** Activation value */
  activation: number
}

/** Top-k features at a specific position */
export interface PositionFeatures {
  /** Position in sequence */
  position: number
  /** Token at this position */
  token?: string
  /** Top-k features sorted by activation */
  topK: FeatureActivation[]
}

/** Feature frequency across a trace */
export interface FeatureFrequency {
  /** Feature index */
  featureIdx: number
  /** Fraction of positions where feature is active */
  frequency: number
  /** Mean activation when active */
  meanActivation: number
  /** Max activation observed */
  maxActivation: number
}

/** Co-activation between two features */
export interface FeatureCoactivation {
  /** First feature index */
  featureA: number
  /** Second feature index */
  featureB: number
  /** Co-activation score (correlation or joint frequency) */
  score: number
}

/** Results from behavior feature analysis */
export interface BehaviorFeatures {
  /** Features that activate more in clean vs corrupt */
  activated: number[]
  /** Features that deactivate in clean vs corrupt */
  deactivated: number[]
  /** Threshold used for classification */
  threshold: number
}

/** Request to load an SAE */
export interface LoadSAERequest {
  /** SAELens model name (e.g., "gpt2-small") */
  modelName: string
  /** Hook point (e.g., "blocks.0.hook_resid_post") */
  hookPoint: string
  /** Layer index */
  layer?: number
}

/** Request to encode activations with SAE */
export interface EncodeRequest {
  /** SAE identifier */
  saeId: string
  /** Trace ID containing activations */
  traceId: string
  /** Layer to encode */
  layer: number
  /** Component to encode */
  component?: string
}

/** Response from SAE encoding */
export interface EncodeResponse {
  features: SAEFeatures
  /** Top-k features per position */
  topKPerPosition?: PositionFeatures[]
  stats?: TensorStats
}

/** SAE analysis session */
export interface SAEAnalysisSession {
  /** Session ID */
  id: string
  /** Loaded SAE configurations */
  loadedSAEs: Record<string, SAEConfig>
  /** Cached feature encodings */
  encodings: Record<string, SAEFeatures>
}

/** Options for SAE visualization */
export interface SAEViewOptions {
  /** Number of top features to show per position */
  topK: number
  /** Minimum activation to display */
  activationThreshold: number
  /** Show feature co-activation network */
  showCoactivation: boolean
  /** Feature indices to highlight */
  highlightedFeatures: number[]
}
