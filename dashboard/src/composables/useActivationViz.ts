/**
 * Activation Visualization Composable
 *
 * Provides utilities for activation pattern visualization.
 */

import { computed, ref, type Ref, type ComputedRef } from 'vue'
import type { TensorData } from '@/types'

export interface ActivationStats {
  mean: number
  variance: number
  min: number
  max: number
  l2Norm: number
  sparsity: number
}

export interface TokenStats {
  tokenIdx: number
  token: string
  l2Norm: number
  mean: number
  maxAbs: number
}

export interface DimensionStats {
  dimIdx: number
  mean: number
  variance: number
  maxAbs: number
}

export interface ActivationMatrixData {
  /** Shape: [seqLen, hiddenDim] */
  data: number[][]
  seqLen: number
  hiddenDim: number
  tokens: string[]
  stats: ActivationStats
}

/**
 * Compute statistics for an array of values
 */
export function computeArrayStats(values: number[]): ActivationStats {
  if (values.length === 0) {
    return { mean: 0, variance: 0, min: 0, max: 0, l2Norm: 0, sparsity: 0 }
  }

  let sum = 0
  let sumSq = 0
  let min = Infinity
  let max = -Infinity
  let sparseCount = 0

  for (const val of values) {
    sum += val
    sumSq += val * val
    min = Math.min(min, val)
    max = Math.max(max, val)
    if (Math.abs(val) < 0.01) sparseCount++
  }

  const mean = sum / values.length
  const variance = sumSq / values.length - mean * mean
  const l2Norm = Math.sqrt(sumSq)
  const sparsity = sparseCount / values.length

  return { mean, variance, min, max, l2Norm, sparsity }
}

/**
 * Compute L2 norm for each row (token)
 */
export function computeTokenNorms(data: number[][]): number[] {
  return data.map((row) => {
    const sumSq = row.reduce((acc, val) => acc + val * val, 0)
    return Math.sqrt(sumSq)
  })
}

/**
 * Compute statistics per token
 */
export function computeTokenStats(data: number[][], tokens: string[]): TokenStats[] {
  return data.map((row, idx) => {
    const sumSq = row.reduce((acc, val) => acc + val * val, 0)
    const sum = row.reduce((acc, val) => acc + val, 0)
    const maxAbs = row.reduce((acc, val) => Math.max(acc, Math.abs(val)), 0)

    return {
      tokenIdx: idx,
      token: tokens[idx] ?? `[${idx}]`,
      l2Norm: Math.sqrt(sumSq),
      mean: sum / row.length,
      maxAbs,
    }
  })
}

/**
 * Compute statistics per dimension
 */
export function computeDimensionStats(data: number[][]): DimensionStats[] {
  if (data.length === 0 || !data[0]) return []

  const hiddenDim = data[0].length
  const result: DimensionStats[] = []

  for (let d = 0; d < hiddenDim; d++) {
    let sum = 0
    let sumSq = 0
    let maxAbs = 0

    for (const row of data) {
      const val = row[d] ?? 0
      sum += val
      sumSq += val * val
      maxAbs = Math.max(maxAbs, Math.abs(val))
    }

    const mean = sum / data.length
    const variance = sumSq / data.length - mean * mean

    result.push({ dimIdx: d, mean, variance, maxAbs })
  }

  return result
}

/**
 * Get top-k dimensions by variance
 */
export function getTopDimensionsByVariance(dimStats: DimensionStats[], k = 10): DimensionStats[] {
  return [...dimStats].sort((a, b) => b.variance - a.variance).slice(0, k)
}

/**
 * Get top-k dimensions by max absolute value
 */
export function getTopDimensionsByMaxAbs(dimStats: DimensionStats[], k = 10): DimensionStats[] {
  return [...dimStats].sort((a, b) => b.maxAbs - a.maxAbs).slice(0, k)
}

/**
 * Extract 2D matrix from TensorData
 * Handles shapes like [batch, seq, hidden] or [seq, hidden]
 */
export function extractActivationMatrix(
  tensorData: TensorData,
  batchIdx = 0
): number[][] {
  // Defensive check for missing or invalid tensorData/shape
  if (!tensorData?.shape || !Array.isArray(tensorData.shape)) {
    return []
  }

  const { data, shape } = tensorData
  const flatData = data as number[]

  if (shape.length === 2) {
    // [seq, hidden]
    const [seqLen, hiddenDim] = shape
    if (seqLen === undefined || hiddenDim === undefined) return []

    const result: number[][] = []
    for (let s = 0; s < seqLen; s++) {
      const row: number[] = []
      for (let h = 0; h < hiddenDim; h++) {
        row.push(flatData[s * hiddenDim + h] ?? 0)
      }
      result.push(row)
    }
    return result
  }

  if (shape.length === 3) {
    // [batch, seq, hidden]
    const [, seqLen, hiddenDim] = shape
    if (seqLen === undefined || hiddenDim === undefined) return []

    const batchOffset = batchIdx * seqLen * hiddenDim
    const result: number[][] = []
    for (let s = 0; s < seqLen; s++) {
      const row: number[] = []
      for (let h = 0; h < hiddenDim; h++) {
        row.push(flatData[batchOffset + s * hiddenDim + h] ?? 0)
      }
      result.push(row)
    }
    return result
  }

  return []
}

/**
 * Normalize matrix values to [0, 1] range
 */
export function normalizeMatrix(data: number[][]): number[][] {
  let min = Infinity
  let max = -Infinity

  for (const row of data) {
    for (const val of row) {
      min = Math.min(min, val)
      max = Math.max(max, val)
    }
  }

  const range = max - min || 1
  return data.map((row) => row.map((val) => (val - min) / range))
}

/**
 * Normalize by absolute value (for diverging color scales)
 */
export function normalizeMatrixDiverging(data: number[][]): number[][] {
  let maxAbs = 0

  for (const row of data) {
    for (const val of row) {
      maxAbs = Math.max(maxAbs, Math.abs(val))
    }
  }

  if (maxAbs === 0) return data.map((row) => row.map(() => 0.5))

  // Map to [0, 1] where 0.5 is zero
  return data.map((row) => row.map((val) => (val / maxAbs + 1) / 2))
}

export interface UseActivationVizOptions {
  activations: Ref<TensorData | null>
  tokens: Ref<string[]>
}

export function useActivationViz(options: UseActivationVizOptions) {
  const { activations, tokens } = options

  // Internal state
  const hoveredToken = ref<number | null>(null)
  const hoveredDimension = ref<number | null>(null)
  const selectedDimensions = ref<number[]>([])

  // Extract 2D matrix from tensor
  const activationMatrix: ComputedRef<number[][]> = computed(() => {
    if (!activations.value) return []
    return extractActivationMatrix(activations.value)
  })

  // Matrix dimensions
  const seqLen = computed(() => activationMatrix.value.length)
  const hiddenDim = computed(() => activationMatrix.value[0]?.length ?? 0)

  // Overall statistics
  const overallStats: ComputedRef<ActivationStats> = computed(() => {
    const flat = activationMatrix.value.flat()
    return computeArrayStats(flat)
  })

  // Per-token statistics
  const tokenStats: ComputedRef<TokenStats[]> = computed(() => {
    return computeTokenStats(activationMatrix.value, tokens.value)
  })

  // Token norms
  const tokenNorms: ComputedRef<number[]> = computed(() => {
    return computeTokenNorms(activationMatrix.value)
  })

  // Per-dimension statistics
  const dimensionStats: ComputedRef<DimensionStats[]> = computed(() => {
    return computeDimensionStats(activationMatrix.value)
  })

  // Top dimensions by variance
  const topDimensionsByVariance = computed(() => {
    return getTopDimensionsByVariance(dimensionStats.value, 20)
  })

  // Normalized matrix for display
  const normalizedMatrix = computed(() => {
    return normalizeMatrixDiverging(activationMatrix.value)
  })

  // Matrix data structure for rendering
  const matrixData: ComputedRef<ActivationMatrixData | null> = computed(() => {
    if (activationMatrix.value.length === 0) return null

    return {
      data: activationMatrix.value,
      seqLen: seqLen.value,
      hiddenDim: hiddenDim.value,
      tokens: tokens.value,
      stats: overallStats.value,
    }
  })

  // Methods
  function setHoveredToken(idx: number | null): void {
    hoveredToken.value = idx
  }

  function setHoveredDimension(idx: number | null): void {
    hoveredDimension.value = idx
  }

  function toggleDimensionSelection(idx: number): void {
    const index = selectedDimensions.value.indexOf(idx)
    if (index >= 0) {
      selectedDimensions.value.splice(index, 1)
    } else {
      selectedDimensions.value.push(idx)
    }
  }

  function clearDimensionSelection(): void {
    selectedDimensions.value = []
  }

  function getTokenActivations(tokenIdx: number): number[] {
    return activationMatrix.value[tokenIdx] ?? []
  }

  function getDimensionActivations(dimIdx: number): number[] {
    return activationMatrix.value.map((row) => row[dimIdx] ?? 0)
  }

  return {
    // State
    hoveredToken,
    hoveredDimension,
    selectedDimensions,

    // Computed
    activationMatrix,
    seqLen,
    hiddenDim,
    overallStats,
    tokenStats,
    tokenNorms,
    dimensionStats,
    topDimensionsByVariance,
    normalizedMatrix,
    matrixData,

    // Methods
    setHoveredToken,
    setHoveredDimension,
    toggleDimensionSelection,
    clearDimensionSelection,
    getTokenActivations,
    getDimensionActivations,
  }
}
