/**
 * Attention Visualization Composable
 *
 * Provides utilities for attention pattern visualization.
 */

import { computed, ref, type Ref, type ComputedRef } from 'vue'
import type { TensorData, HeadAnalysis, HeadClassification } from '@/types'

export interface AttentionCell {
  queryIdx: number
  keyIdx: number
  value: number
  queryToken: string
  keyToken: string
}

export interface AttentionRow {
  queryIdx: number
  queryToken: string
  values: number[]
}

export interface AttentionMatrixData {
  rows: AttentionRow[]
  tokens: string[]
  seqLen: number
  minValue: number
  maxValue: number
}

export interface HeadInfo {
  layer: number
  head: number
  classification?: HeadClassification
  entropy?: number
  selected: boolean
}

/**
 * Reshape a flat array into a 2D matrix
 */
function reshapeFlat2D(flat: number[], rows: number, cols: number): number[][] {
  const result: number[][] = []
  for (let i = 0; i < rows; i++) {
    result.push(flat.slice(i * cols, (i + 1) * cols))
  }
  return result
}

/**
 * Extract 2D attention matrix for a specific head from tensor data.
 * Handles multiple shape formats:
 * - 4D: [batch, heads, seq_q, seq_k] -> [seq_q, seq_k]
 * - 3D: [heads, seq_q, seq_k] -> [seq_q, seq_k]
 * - 2D: [seq_q, seq_k] -> [seq_q, seq_k] (passthrough)
 */
export function extractHeadAttention(
  data: number[] | number[][][][] | number[][][] | number[][],
  shape: number[],
  headIdx: number,
  batchIdx = 0
): number[][] {
  // Handle pre-structured nested arrays
  if (Array.isArray(data) && data.length > 0 && Array.isArray(data[0])) {
    if (shape.length === 2) {
      // Already [seq_q, seq_k] - return as-is
      return data as number[][]
    }
    if (shape.length === 3) {
      // [heads, seq_q, seq_k] - extract head
      const data3d = data as number[][][]
      return data3d[headIdx] ?? []
    }
    if (shape.length === 4) {
      // [batch, heads, seq_q, seq_k] - existing logic
      const data4d = data as number[][][][]
      const batchData = data4d[batchIdx]
      if (!batchData) return []
      return batchData[headIdx] ?? []
    }
    console.warn('[extractHeadAttention] Unexpected nested array with shape:', shape)
    return []
  }

  // Handle flat array - needs reshaping based on shape
  const flatData = data as number[]
  if (flatData.length === 0) return []

  if (shape.length === 2) {
    // [seq_q, seq_k] - reshape to 2D
    const [seqQ, seqK] = shape
    if (seqQ === undefined || seqK === undefined) return []
    return reshapeFlat2D(flatData, seqQ, seqK)
  }

  if (shape.length === 3) {
    // [heads, seq_q, seq_k] - extract head slice
    const [, seqQ, seqK] = shape
    if (seqQ === undefined || seqK === undefined) return []
    const headOffset = headIdx * seqQ * seqK
    return reshapeFlat2D(flatData.slice(headOffset, headOffset + seqQ * seqK), seqQ, seqK)
  }

  if (shape.length === 4) {
    // [batch, heads, seq_q, seq_k] - original logic
    const [, numHeads, seqQ, seqK] = shape
    if (numHeads === undefined || seqQ === undefined || seqK === undefined) return []
    const batchOffset = batchIdx * numHeads * seqQ * seqK
    const headOffset = headIdx * seqQ * seqK
    const start = batchOffset + headOffset
    return reshapeFlat2D(flatData.slice(start, start + seqQ * seqK), seqQ, seqK)
  }

  console.warn('[extractHeadAttention] Unsupported shape:', shape)
  return []
}

/**
 * Compute mean attention across all heads
 * Handles 3D [heads, seq_q, seq_k] and 4D [batch, heads, seq_q, seq_k] shapes
 */
export function computeMeanAttention(
  data: number[] | number[][][][] | number[][][] | number[][],
  shape: number[],
  batchIdx = 0
): number[][] {
  let numHeads: number
  let seqQ: number
  let seqK: number

  if (shape.length === 2) {
    // Already single head - return as-is
    return extractHeadAttention(data, shape, 0, batchIdx)
  } else if (shape.length === 3) {
    numHeads = shape[0] ?? 0
    seqQ = shape[1] ?? 0
    seqK = shape[2] ?? 0
  } else if (shape.length === 4) {
    numHeads = shape[1] ?? 0
    seqQ = shape[2] ?? 0
    seqK = shape[3] ?? 0
  } else {
    console.warn('[computeMeanAttention] Unsupported shape:', shape)
    return []
  }

  if (numHeads === 0 || seqQ === 0 || seqK === 0) return []

  // Initialize result matrix with zeros
  const result: number[][] = Array.from({ length: seqQ }, () =>
    Array(seqK).fill(0) as number[]
  )

  // Sum all heads and divide by count
  for (let h = 0; h < numHeads; h++) {
    const headMatrix = extractHeadAttention(data, shape, h, batchIdx)
    for (let q = 0; q < seqQ; q++) {
      const row = result[q]
      if (!row) continue
      for (let k = 0; k < seqK; k++) {
        row[k] = (row[k] ?? 0) + (headMatrix[q]?.[k] ?? 0) / numHeads
      }
    }
  }

  return result
}

/**
 * Compute attention statistics for a matrix
 */
export function computeAttentionStats(matrix: number[][]): {
  entropy: number
  sparsity: number
  maxAttention: number
  avgAttention: number
} {
  if (matrix.length === 0) {
    return { entropy: 0, sparsity: 0, maxAttention: 0, avgAttention: 0 }
  }

  let totalEntropy = 0
  let sparseCount = 0
  let maxVal = -Infinity
  let sum = 0
  let count = 0

  for (const row of matrix) {
    // Compute row entropy
    let rowEntropy = 0
    for (const val of row) {
      if (val > 1e-10) {
        rowEntropy -= val * Math.log2(val)
      }
      if (val < 0.01) sparseCount++
      maxVal = Math.max(maxVal, val)
      sum += val
      count++
    }
    totalEntropy += rowEntropy
  }

  return {
    entropy: totalEntropy / matrix.length,
    sparsity: sparseCount / count,
    maxAttention: maxVal,
    avgAttention: sum / count,
  }
}

/**
 * Get top-k attended positions for each query
 */
export function getTopAttended(matrix: number[][], k = 3): number[][] {
  return matrix.map((row) => {
    const indexed = row.map((val, idx) => ({ val, idx }))
    indexed.sort((a, b) => b.val - a.val)
    return indexed.slice(0, k).map((x) => x.idx)
  })
}

/**
 * Classify attention head pattern
 */
export function classifyAttentionPattern(matrix: number[][]): HeadClassification {
  if (matrix.length === 0) return 'other'

  const seqLen = matrix.length
  const stats = computeAttentionStats(matrix)

  // Check for uniform attention
  if (stats.entropy > Math.log2(seqLen) * 0.9) {
    return 'uniform'
  }

  // Check for BOS attention (attending to position 0)
  let bosAttention = 0
  for (const row of matrix) {
    bosAttention += row[0] ?? 0
  }
  if (bosAttention / seqLen > 0.5) {
    return 'bos'
  }

  // Check for previous token attention
  let prevTokenAttention = 0
  for (let i = 1; i < seqLen; i++) {
    const row = matrix[i]
    prevTokenAttention += row?.[i - 1] ?? 0
  }
  if (prevTokenAttention / (seqLen - 1) > 0.5) {
    return 'previous_token'
  }

  // Check for induction pattern (stripe pattern)
  // This is simplified - real detection needs more sophisticated analysis
  let inductionScore = 0
  for (let i = 2; i < seqLen; i++) {
    const row = matrix[i]
    if (row) {
      // Check if attending to position that follows a repeated token
      const maxIdx = row.indexOf(Math.max(...row))
      if (maxIdx > 0 && maxIdx < i - 1) {
        inductionScore++
      }
    }
  }
  if (inductionScore / (seqLen - 2) > 0.3) {
    return 'induction'
  }

  return 'other'
}

export interface UseAttentionVizOptions {
  pattern: Ref<TensorData<number[] | number[][][][]> | null>
  tokens: Ref<string[]>
  selectedHead: Ref<number>
  analyses?: Ref<HeadAnalysis[] | undefined>
}

export function useAttentionViz(options: UseAttentionVizOptions) {
  const { pattern, tokens, selectedHead, analyses } = options

  // Internal state
  const hoveredCell = ref<AttentionCell | null>(null)
  const zoomLevel = ref(1)

  // Extract attention matrix for selected head
  const attentionMatrix: ComputedRef<number[][]> = computed(() => {
    if (!pattern.value) return []

    return extractHeadAttention(
      pattern.value.data,
      pattern.value.shape,
      selectedHead.value
    )
  })

  // Build matrix data structure for rendering
  const matrixData: ComputedRef<AttentionMatrixData | null> = computed(() => {
    const matrix = attentionMatrix.value
    if (matrix.length === 0) return null

    const tokenList = tokens.value
    let minVal = Infinity
    let maxVal = -Infinity

    const rows: AttentionRow[] = matrix.map((row, qIdx) => {
      for (const val of row) {
        minVal = Math.min(minVal, val)
        maxVal = Math.max(maxVal, val)
      }
      return {
        queryIdx: qIdx,
        queryToken: tokenList[qIdx] ?? `[${qIdx}]`,
        values: row,
      }
    })

    return {
      rows,
      tokens: tokenList.length > 0 ? tokenList : matrix[0]?.map((_, i) => `[${i}]`) ?? [],
      seqLen: matrix.length,
      minValue: minVal === Infinity ? 0 : minVal,
      maxValue: maxVal === -Infinity ? 1 : maxVal,
    }
  })

  // Compute statistics for current head
  const statistics = computed(() => {
    return computeAttentionStats(attentionMatrix.value)
  })

  // Get classification from analyses or compute
  const headClassification = computed((): HeadClassification => {
    if (analyses?.value) {
      const analysis = analyses.value.find((a) => a.head === selectedHead.value)
      if (analysis) {
        // Handle both string and object classification formats from API
        const classification = analysis.classification
        if (typeof classification === 'string') {
          return classification
        }
        if (classification && typeof classification === 'object' && 'category' in classification) {
          return classification.category
        }
      }
    }
    return classifyAttentionPattern(attentionMatrix.value)
  })

  // Top attended positions
  const topAttendedPositions = computed(() => {
    return getTopAttended(attentionMatrix.value)
  })

  // Sequence length
  const seqLen = computed(() => attentionMatrix.value.length)

  // Methods
  function setHoveredCell(cell: AttentionCell | null): void {
    hoveredCell.value = cell
  }

  function getCellValue(queryIdx: number, keyIdx: number): number {
    const row = attentionMatrix.value[queryIdx]
    return row?.[keyIdx] ?? 0
  }

  function getRowValues(queryIdx: number): number[] {
    return attentionMatrix.value[queryIdx] ?? []
  }

  function setZoom(level: number): void {
    zoomLevel.value = Math.max(0.5, Math.min(4, level))
  }

  return {
    // State
    hoveredCell,
    zoomLevel,

    // Computed
    attentionMatrix,
    matrixData,
    statistics,
    headClassification,
    topAttendedPositions,
    seqLen,

    // Methods
    setHoveredCell,
    getCellValue,
    getRowValues,
    setZoom,
  }
}
