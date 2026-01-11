/**
 * SAE Analysis Composable
 *
 * Provides utilities for Sparse Autoencoder feature visualization and analysis.
 */

import { ref, computed, type Ref } from 'vue'
import type {
  SAEFeatures,
  PositionFeatures,
  FeatureActivation,
  FeatureFrequency,
  FeatureCoactivation,
  SAEViewOptions,
} from '@/types'

/** Color scale for feature activations */
export function getFeatureColor(activation: number, maxActivation: number): string {
  const normalized = Math.min(1, activation / maxActivation)
  // Use a viridis-like color scale
  if (normalized < 0.25) {
    const t = normalized / 0.25
    return `rgb(${68 + t * 30}, ${1 + t * 60}, ${84 + t * 60})`
  } else if (normalized < 0.5) {
    const t = (normalized - 0.25) / 0.25
    return `rgb(${98 - t * 40}, ${61 + t * 50}, ${144 - t * 30})`
  } else if (normalized < 0.75) {
    const t = (normalized - 0.5) / 0.25
    return `rgb(${58 + t * 120}, ${111 + t * 60}, ${114 - t * 30})`
  } else {
    const t = (normalized - 0.75) / 0.25
    return `rgb(${178 + t * 75}, ${171 - t * 40}, ${84 - t * 50})`
  }
}

/** Options for useSAE composable */
export interface UseSAEOptions {
  features: Ref<SAEFeatures | null>
  topKPerPosition?: Ref<PositionFeatures[] | null>
  viewOptions?: Ref<SAEViewOptions>
}

/** Statistics computed from SAE features */
export interface SAEStats {
  totalFeatures: number
  activeFeatures: number
  sparsity: number
  meanActivePerPosition: number
  maxActivation: number
  minActivation: number
}

/**
 * Composable for SAE feature analysis and visualization
 */
export function useSAE(options: UseSAEOptions) {
  const { features, topKPerPosition, viewOptions } = options

  // Default view options
  const defaultViewOptions: SAEViewOptions = {
    topK: 10,
    activationThreshold: 0.01,
    showCoactivation: false,
    highlightedFeatures: [],
  }

  const currentViewOptions = computed(() => viewOptions?.value ?? defaultViewOptions)

  // Selected feature for detail view
  const selectedFeature = ref<number | null>(null)
  const hoveredFeature = ref<number | null>(null)

  // Computed statistics
  const stats = computed((): SAEStats | null => {
    if (!features.value) return null

    const data = features.value.activations.data
    let activeCount = 0
    let maxVal = -Infinity
    let minVal = Infinity

    for (const val of data) {
      if (val > 0) {
        activeCount++
        maxVal = Math.max(maxVal, val)
        minVal = Math.min(minVal, val)
      }
    }

    return {
      totalFeatures: features.value.config?.dSae ?? data.length,
      activeFeatures: activeCount,
      sparsity: features.value.sparsity,
      meanActivePerPosition: features.value.meanActiveFeatures,
      maxActivation: maxVal === -Infinity ? 0 : maxVal,
      minActivation: minVal === Infinity ? 0 : minVal,
    }
  })

  // Compute top-K features globally (most frequently active)
  const topKGlobal = computed((): FeatureFrequency[] => {
    if (!topKPerPosition?.value) return []

    const featureCounts: Map<number, { count: number; sumAct: number; maxAct: number }> =
      new Map()

    for (const pos of topKPerPosition.value) {
      for (const feat of pos.topK) {
        const existing = featureCounts.get(feat.featureIdx) ?? {
          count: 0,
          sumAct: 0,
          maxAct: 0,
        }
        featureCounts.set(feat.featureIdx, {
          count: existing.count + 1,
          sumAct: existing.sumAct + feat.activation,
          maxAct: Math.max(existing.maxAct, feat.activation),
        })
      }
    }

    const numPositions = topKPerPosition.value.length
    const frequencies: FeatureFrequency[] = []

    for (const [idx, counts] of featureCounts.entries()) {
      frequencies.push({
        featureIdx: idx,
        frequency: counts.count / numPositions,
        meanActivation: counts.sumAct / counts.count,
        maxActivation: counts.maxAct,
      })
    }

    return frequencies
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, currentViewOptions.value.topK)
  })

  // Compute feature co-activations
  const coactivations = computed((): FeatureCoactivation[] => {
    if (!topKPerPosition?.value || !currentViewOptions.value.showCoactivation) {
      return []
    }

    const pairCounts: Map<string, number> = new Map()

    for (const pos of topKPerPosition.value) {
      const features = pos.topK.map((f) => f.featureIdx)

      // Count co-occurring pairs
      for (let i = 0; i < features.length; i++) {
        for (let j = i + 1; j < features.length; j++) {
          const fi = features[i]
          const fj = features[j]
          if (fi !== undefined && fj !== undefined) {
            const key = `${Math.min(fi, fj)}-${Math.max(fi, fj)}`
            pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1)
          }
        }
      }
    }

    const numPositions = topKPerPosition.value.length
    const result: FeatureCoactivation[] = []

    for (const [key, count] of pairCounts.entries()) {
      const parts = key.split('-').map(Number)
      result.push({
        featureA: parts[0] ?? 0,
        featureB: parts[1] ?? 0,
        score: count / numPositions,
      })
    }

    return result.sort((a, b) => b.score - a.score).slice(0, 50)
  })

  // Get features for a specific position
  function getFeaturesAtPosition(position: number): FeatureActivation[] {
    if (!topKPerPosition?.value) return []

    const posData = topKPerPosition.value.find((p) => p.position === position)
    if (!posData) return []

    return posData.topK.filter(
      (f) => f.activation >= currentViewOptions.value.activationThreshold
    )
  }

  // Get positions where a feature is active
  function getFeaturePositions(featureIdx: number): number[] {
    if (!topKPerPosition?.value) return []

    return topKPerPosition.value
      .filter((pos) => pos.topK.some((f) => f.featureIdx === featureIdx))
      .map((pos) => pos.position)
  }

  // Check if feature is highlighted
  function isFeatureHighlighted(featureIdx: number): boolean {
    return currentViewOptions.value.highlightedFeatures.includes(featureIdx)
  }

  // Get color for a feature activation
  function getColorForActivation(activation: number): string {
    const maxAct = stats.value?.maxActivation ?? 1
    return getFeatureColor(activation, maxAct)
  }

  return {
    // State
    selectedFeature,
    hoveredFeature,
    // Computed
    stats,
    topKGlobal,
    coactivations,
    currentViewOptions,
    // Methods
    getFeaturesAtPosition,
    getFeaturePositions,
    isFeatureHighlighted,
    getColorForActivation,
  }
}

/**
 * Generate a matrix for feature co-activation heatmap
 */
export function generateCoactivationMatrix(
  coactivations: FeatureCoactivation[],
  featureIndices: number[]
): number[][] {
  const n = featureIndices.length
  const indexMap = new Map(featureIndices.map((f, i) => [f, i]))
  const matrix: number[][] = Array(n)
    .fill(null)
    .map(() => Array(n).fill(0))

  for (const coact of coactivations) {
    const i = indexMap.get(coact.featureA)
    const j = indexMap.get(coact.featureB)
    if (i !== undefined && j !== undefined && matrix[i] && matrix[j]) {
      matrix[i][j] = coact.score
      matrix[j][i] = coact.score
    }
  }

  return matrix
}

/**
 * Format feature index for display
 */
export function formatFeatureIdx(idx: number): string {
  return `#${idx.toLocaleString()}`
}

/**
 * Format activation value for display
 */
export function formatActivation(value: number): string {
  if (value >= 100) return value.toFixed(0)
  if (value >= 10) return value.toFixed(1)
  if (value >= 1) return value.toFixed(2)
  return value.toFixed(3)
}

