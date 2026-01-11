/**
 * Tensor Data Composable
 *
 * Utilities for working with tensor data from the API.
 */

import { computed, type Ref, type ComputedRef } from 'vue'
import type { TensorData } from '@/types'

export interface TensorView {
  data: number[]
  shape: number[]
  get: (indices: number[]) => number
  slice: (start: number[], end: number[]) => number[]
  reshape: (newShape: number[]) => TensorView
  toArray2D: () => number[][]
}

/**
 * Create a tensor view from TensorData
 */
function createTensorView(tensorData: TensorData): TensorView {
  const { data, shape } = tensorData

  // Calculate strides for indexing
  const strides: number[] = []
  let stride = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    strides.unshift(stride)
    stride *= shape[i] ?? 1
  }

  /**
   * Get element at multi-dimensional index
   */
  function get(indices: number[]): number {
    if (indices.length !== shape.length) {
      throw new Error(`Expected ${shape.length} indices, got ${indices.length}`)
    }

    let flatIndex = 0
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i]
      const dim = shape[i]
      const strideVal = strides[i] ?? 1
      if (idx === undefined || dim === undefined || idx < 0 || idx >= dim) {
        throw new Error(`Index ${idx} out of bounds for dimension ${i} with size ${dim}`)
      }
      flatIndex += idx * strideVal
    }

    return (data as number[])[flatIndex] ?? 0
  }

  /**
   * Slice tensor along all dimensions
   */
  function slice(start: number[], end: number[]): number[] {
    const result: number[] = []

    function recurse(dim: number, offset: number): void {
      if (dim === shape.length) {
        result.push((data as number[])[offset] ?? 0)
        return
      }

      const s = start[dim] ?? 0
      const dimSize = shape[dim] ?? 0
      const e = end[dim] ?? dimSize

      for (let i = s; i < e; i++) {
        recurse(dim + 1, offset + i * (strides[dim] ?? 1))
      }
    }

    recurse(0, 0)
    return result
  }

  /**
   * Reshape tensor (view, no copy)
   */
  function reshape(newShape: number[]): TensorView {
    const newSize = newShape.reduce((a, b) => a * b, 1)
    const dataArr = data as number[]
    if (newSize !== dataArr.length) {
      throw new Error(`Cannot reshape tensor of size ${dataArr.length} to shape [${newShape.join(', ')}]`)
    }

    return createTensorView({
      data,
      shape: newShape,
      dtype: tensorData.dtype,
    })
  }

  /**
   * Convert to 2D array (for matrices)
   */
  function toArray2D(): number[][] {
    if (shape.length !== 2) {
      throw new Error(`Cannot convert ${shape.length}D tensor to 2D array`)
    }

    const rows = shape[0] ?? 0
    const cols = shape[1] ?? 0
    const dataArr = data as number[]
    const result: number[][] = []

    for (let i = 0; i < rows; i++) {
      const row: number[] = []
      for (let j = 0; j < cols; j++) {
        row.push(dataArr[i * cols + j] ?? 0)
      }
      result.push(row)
    }

    return result
  }

  return {
    data: data as number[],
    shape,
    get,
    slice,
    reshape,
    toArray2D,
  }
}

export function useTensorData(tensorDataRef: Ref<TensorData | null>) {
  const tensor: ComputedRef<TensorView | null> = computed(() => {
    if (!tensorDataRef.value) return null
    return createTensorView(tensorDataRef.value)
  })

  const shape = computed(() => tensor.value?.shape ?? [])

  const ndim = computed(() => shape.value.length)

  const size = computed(() => tensor.value?.data.length ?? 0)

  /**
   * Get element at index
   */
  function get(...indices: number[]): number {
    return tensor.value?.get(indices) ?? 0
  }

  /**
   * Slice tensor
   */
  function slice(start: number[], end: number[]): number[] {
    return tensor.value?.slice(start, end) ?? []
  }

  /**
   * Convert to 2D array
   */
  function toArray2D(): number[][] {
    return tensor.value?.toArray2D() ?? []
  }

  /**
   * Get row from 2D tensor
   */
  function getRow(rowIndex: number): number[] {
    if (!tensor.value || shape.value.length !== 2) return []
    const cols = shape.value[1] ?? 0
    const start = rowIndex * cols
    return tensor.value.data.slice(start, start + cols)
  }

  /**
   * Get column from 2D tensor
   */
  function getColumn(colIndex: number): number[] {
    if (!tensor.value || shape.value.length !== 2) return []
    const rows = shape.value[0] ?? 0
    const cols = shape.value[1] ?? 0
    const result: number[] = []
    for (let i = 0; i < rows; i++) {
      result.push(tensor.value.data[i * cols + colIndex] ?? 0)
    }
    return result
  }

  /**
   * Compute row-wise statistics
   */
  function rowStats(): Array<{ min: number; max: number; mean: number }> {
    if (!tensor.value || shape.value.length !== 2) return []
    const rows = shape.value[0] ?? 0
    const cols = shape.value[1] ?? 0
    const result: Array<{ min: number; max: number; mean: number }> = []

    for (let i = 0; i < rows; i++) {
      const row = getRow(i)
      const min = Math.min(...row)
      const max = Math.max(...row)
      const mean = row.reduce((a, b) => a + b, 0) / cols
      result.push({ min, max, mean })
    }

    return result
  }

  /**
   * Normalize data to [0, 1] range
   */
  function normalize(): number[] {
    if (!tensor.value) return []
    const { data } = tensor.value
    const minVal = Math.min(...data)
    const maxVal = Math.max(...data)
    const range = maxVal - minVal || 1

    return data.map((v) => (v - minVal) / range)
  }

  /**
   * Apply softmax to data (useful for attention)
   */
  function softmax(): number[] {
    if (!tensor.value) return []
    const { data } = tensor.value

    const maxVal = Math.max(...data)
    const exp = data.map((v) => Math.exp(v - maxVal))
    const sum = exp.reduce((a, b) => a + b, 0)

    return exp.map((v) => v / sum)
  }

  return {
    tensor,
    shape,
    ndim,
    size,
    get,
    slice,
    toArray2D,
    getRow,
    getColumn,
    rowStats,
    normalize,
    softmax,
  }
}

/**
 * Reshape flat array to 2D matrix
 */
export function reshapeTo2D(data: number[], rows: number, cols: number): number[][] {
  if (data.length !== rows * cols) {
    throw new Error(`Cannot reshape ${data.length} elements to ${rows}x${cols}`)
  }

  const result: number[][] = []
  for (let i = 0; i < rows; i++) {
    result.push(data.slice(i * cols, (i + 1) * cols))
  }
  return result
}

/**
 * Flatten 2D array to 1D
 */
export function flatten2D(data: number[][]): number[] {
  return data.flat()
}
