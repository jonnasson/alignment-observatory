/**
 * Tensor-related types for handling multi-dimensional data
 */

/** Serialized tensor data from API (JSON format) */
export interface TensorData<T = number[]> {
  /** Flattened array data */
  data: T
  /** Shape of the tensor (e.g., [batch, seq, hidden]) */
  shape: number[]
  /** Data type */
  dtype: 'float32' | 'float64' | 'int32' | 'int64'
}

/** Tensor metadata without the actual data */
export interface TensorMetadata {
  shape: number[]
  dtype: string
  size: number
  /** Optional URL to fetch binary data */
  dataUrl?: string
}

/** Statistics computed on a tensor */
export interface TensorStats {
  min: number
  max: number
  mean: number
  variance: number
  /** Top-k dimensions by magnitude: [dimension_index, value] */
  topDimensions?: [number, number][]
}

/** Typed array types for different precisions */
export type TypedArray = Float32Array | Float64Array | Int32Array | Uint32Array

/** Helper type for 2D array (e.g., [seq, hidden]) */
export type Array2D = number[][]

/** Helper type for 3D array (e.g., [batch, seq, hidden]) */
export type Array3D = number[][][]

/** Helper type for 4D array (e.g., [batch, heads, seq_q, seq_k]) */
export type Array4D = number[][][][]
