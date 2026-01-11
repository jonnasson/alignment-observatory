/**
 * API types for HTTP and WebSocket communication
 */

/** Standard API response wrapper */
export interface ApiResponse<T> {
  data: T
  status: 'success' | 'error'
  message?: string
}

/** Paginated API response */
export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  total: number
  page: number
  pageSize: number
  hasMore: boolean
}

/** API error response */
export interface ApiError {
  code: ErrorCode
  message: string
  details?: Record<string, unknown>
  traceId?: string
}

/** Error codes for structured error handling */
export const ErrorCode = {
  // Resource errors
  TRACE_NOT_FOUND: 'TRACE_NOT_FOUND',
  CIRCUIT_NOT_FOUND: 'CIRCUIT_NOT_FOUND',
  MODEL_NOT_FOUND: 'MODEL_NOT_FOUND',
  SAE_NOT_FOUND: 'SAE_NOT_FOUND',

  // Validation errors
  INVALID_LAYER: 'INVALID_LAYER',
  INVALID_COMPONENT: 'INVALID_COMPONENT',
  INVALID_SHAPE: 'INVALID_SHAPE',
  INVALID_REQUEST: 'INVALID_REQUEST',

  // Computation errors
  TRACING_FAILED: 'TRACING_FAILED',
  DETECTION_FAILED: 'DETECTION_FAILED',
  OOM_ERROR: 'OOM_ERROR',

  // Infrastructure errors
  CACHE_ERROR: 'CACHE_ERROR',
  STORAGE_ERROR: 'STORAGE_ERROR',
  WEBSOCKET_ERROR: 'WEBSOCKET_ERROR',
} as const

export type ErrorCode = (typeof ErrorCode)[keyof typeof ErrorCode]

/** WebSocket message types */
export type WSMessageType =
  | 'activation'
  | 'attention'
  | 'progress'
  | 'complete'
  | 'error'
  | 'ping'
  | 'pong'
  | 'subscribe'
  | 'unsubscribe'

/** WebSocket message from server */
export interface WSServerMessage<T = unknown> {
  type: WSMessageType
  traceId?: string
  timestamp: number
  data: T
}

/** WebSocket message from client */
export interface WSClientMessage {
  type: 'subscribe' | 'unsubscribe' | 'request' | 'ping'
  channel?: string
  data?: unknown
}

/** Progress update during tracing */
export interface ProgressData {
  phase: 'tokenizing' | 'loading_model' | 'forward_pass' | 'capturing' | 'analyzing'
  layer?: number
  totalLayers: number
  percentComplete: number
  message?: string
}

/** Activation update during streaming */
export interface ActivationStreamData {
  layer: number
  component: string
  shape: number[]
  stats: {
    mean: number
    variance: number
    norm: number
  }
  /** URL to fetch full data if needed */
  chunkUrl?: string
}

/** Model information */
export interface ModelInfo {
  id: string
  name: string
  architecture: string
  numLayers: number
  numHeads: number
  hiddenSize: number
  vocabSize: number
  maxSeqLength?: number
  loaded: boolean
  memoryUsageMb?: number
  loadedAt?: string
}

/** List of available models */
export interface ModelsListResponse {
  models: ModelInfo[]
  currentModel?: string
}

/** Request to load a model */
export interface LoadModelRequest {
  modelId: string
  device?: 'cpu' | 'cuda' | 'mps'
  dtype?: 'float32' | 'float16' | 'bfloat16'
}

/** Memory estimation for tracing */
export interface MemoryEstimate {
  estimatedBytes: number
  estimatedMb: number
  strategy: 'full' | 'streaming' | 'selective'
  recommendation: string
  keyLayers?: number[]
}

/** Health check response */
export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy'
  version: string
  uptime: number
  modelLoaded: boolean
  gpuAvailable: boolean
  gpuMemoryUsedMb?: number
}
