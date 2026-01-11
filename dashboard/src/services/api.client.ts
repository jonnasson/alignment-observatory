/**
 * HTTP API Client
 *
 * Axios-based client for REST API communication with the FastAPI backend.
 */

import type {
  ApiResponse,
  AttentionResponse,
  ActivationResponse,
  CircuitDiscoveryResponse,
  DetectIOIResponse,
  EncodeResponse,
  ModelInfo,
  ModelsListResponse,
  MemoryEstimate,
} from '@/types'

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const API_VERSION = 'v1'

/**
 * API Error type
 */
export interface ApiClientError {
  code: string
  message: string
  details?: Record<string, unknown>
  status?: number
}

function createApiError(code: string, message: string, details?: Record<string, unknown>, status?: number): ApiClientError {
  return { code, message, details, status }
}

/**
 * Convert snake_case object keys to camelCase recursively
 */
function snakeToCamel(obj: unknown): unknown {
  if (obj === null || obj === undefined) return obj
  if (Array.isArray(obj)) return obj.map(snakeToCamel)
  if (typeof obj !== 'object') return obj

  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
    const camelKey = key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase())
    result[camelKey] = snakeToCamel(value)
  }
  return result
}

/**
 * Request options for API calls
 */
interface RequestOptions {
  params?: Record<string, string | number | boolean | undefined>
  signal?: AbortSignal
  timeout?: number
}

/**
 * Core fetch wrapper with error handling
 */
async function request<T>(
  method: string,
  endpoint: string,
  options: RequestOptions & { body?: unknown } = {}
): Promise<T> {
  const { params, body, signal, timeout = 30000 } = options

  // Build URL with query params
  const url = new URL(`${API_BASE_URL}/api/${API_VERSION}${endpoint}`)
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        url.searchParams.set(key, String(value))
      }
    })
  }

  // Create abort controller for timeout
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  try {
    const response = await fetch(url.toString(), {
      method,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: signal || controller.signal,
    })

    clearTimeout(timeoutId)

    // Handle non-JSON responses
    const contentType = response.headers.get('content-type')
    if (!contentType?.includes('application/json')) {
      if (!response.ok) {
        throw createApiError('REQUEST_FAILED', `Request failed: ${response.statusText}`, undefined, response.status)
      }
      return undefined as T
    }

    const data = (await response.json()) as ApiResponse<T>

    if (!response.ok || data.status === 'error') {
      throw createApiError(
        'API_ERROR',
        data.message || 'An unknown error occurred',
        undefined,
        response.status
      )
    }

    return snakeToCamel(data.data) as T
  } catch (error) {
    clearTimeout(timeoutId)

    if ((error as ApiClientError).code) {
      throw error
    }

    if (error instanceof DOMException && error.name === 'AbortError') {
      throw createApiError('TIMEOUT', 'Request timed out')
    }

    throw createApiError('NETWORK_ERROR', `Network error: ${(error as Error).message}`)
  }
}

// ============================================================================
// Trace API
// ============================================================================

export interface TraceInfo {
  traceId: string
  createdAt: string
  inputText: string
  tokens: string[]
  tokenIds: number[]
  metadata: {
    modelName: string
    numLayers: number
    numHeads: number
    hiddenSize: number
    seqLength: number
    vocabSize: number
  }
  hasAttention: boolean
  hasActivations: boolean
  layersAvailable: number[]
}

export interface CreateTraceParams {
  text: string
  modelName?: string
  includeAttention?: boolean
  includeActivations?: boolean
  layers?: number[]
}

export interface LoadTraceParams {
  path: string
}

export const traceApi = {
  /**
   * Create a new trace via live inference
   */
  async create(params: CreateTraceParams, options?: RequestOptions): Promise<TraceInfo> {
    return request<TraceInfo>('POST', '/traces', {
      ...options,
      body: {
        text: params.text,
        model_name: params.modelName,
        include_attention: params.includeAttention ?? true,
        include_activations: params.includeActivations ?? true,
        layers: params.layers,
      },
    })
  },

  /**
   * Load a pre-computed trace from disk
   */
  async load(params: LoadTraceParams, options?: RequestOptions): Promise<TraceInfo> {
    return request<TraceInfo>('POST', '/traces/load', {
      ...options,
      body: { path: params.path },
    })
  },

  /**
   * Get trace information
   */
  async get(traceId: string, options?: RequestOptions): Promise<TraceInfo> {
    return request<TraceInfo>('GET', `/traces/${traceId}`, options)
  },

  /**
   * List all traces
   */
  async list(options?: RequestOptions): Promise<TraceInfo[]> {
    return request<TraceInfo[]>('GET', '/traces', options)
  },

  /**
   * Delete a trace
   */
  async delete(traceId: string, options?: RequestOptions): Promise<void> {
    return request<void>('DELETE', `/traces/${traceId}`, options)
  },

  /**
   * Get attention patterns for a layer
   */
  async getAttention(
    traceId: string,
    layer: number,
    head?: number,
    aggregate: 'none' | 'mean' | 'max' = 'none',
    options?: RequestOptions
  ): Promise<AttentionResponse> {
    return request<AttentionResponse>('GET', `/traces/${traceId}/attention/${layer}`, {
      ...options,
      params: { head, aggregate },
    })
  },

  /**
   * Get activations for a layer and component
   */
  async getActivations(
    traceId: string,
    layer: number,
    component: 'residual' | 'attention_out' | 'mlp_out' | 'mlp_hidden',
    tokenIndices?: number[],
    options?: RequestOptions
  ): Promise<ActivationResponse> {
    return request<ActivationResponse>('GET', `/traces/${traceId}/activations/${layer}/${component}`, {
      ...options,
      params: {
        token_indices: tokenIndices?.join(','),
      },
    })
  },
}

// ============================================================================
// Circuit API
// ============================================================================

export interface CircuitDiscoveryParams {
  traceId: string
  targetTokenIdx: number
  method?: 'activation_patching' | 'edge_attribution' | 'causal_tracing'
  threshold?: number
  maxNodes?: number
  includeMlp?: boolean
  includeAttention?: boolean
  cleanInput?: string
}

export const circuitApi = {
  /**
   * Discover a circuit
   */
  async discover(params: CircuitDiscoveryParams, options?: RequestOptions): Promise<CircuitDiscoveryResponse> {
    return request<CircuitDiscoveryResponse>('POST', '/circuits/discover', {
      ...options,
      body: {
        trace_id: params.traceId,
        target_token_idx: params.targetTokenIdx,
        params: {
          method: params.method ?? 'activation_patching',
          threshold: params.threshold ?? 0.01,
          max_nodes: params.maxNodes ?? 50,
          include_mlp: params.includeMlp ?? true,
          include_attention: params.includeAttention ?? true,
        },
        clean_input: params.cleanInput,
      },
    })
  },

  /**
   * Get available discovery methods
   */
  async getMethods(options?: RequestOptions): Promise<Array<{ name: string; displayName: string; description: string }>> {
    return request('GET', '/circuits/methods', options)
  },
}

// ============================================================================
// IOI API
// ============================================================================

export interface IOISentence {
  text: string
  subjectName: string
  indirectObjectName: string
  subjectTokenIdx: number
  indirectObjectTokenIdx: number
  finalTokenIdx: number
  template?: string
}

export interface DetectIOIParams {
  traceId: string
  sentence?: IOISentence
  usePatching?: boolean
  headThreshold?: number
  validateAgainstKnown?: boolean
}

export const ioiApi = {
  /**
   * Parse an IOI sentence
   */
  async parseSentence(text: string, options?: RequestOptions): Promise<IOISentence> {
    return request<IOISentence>('POST', '/ioi/parse', {
      ...options,
      body: { text },
    })
  },

  /**
   * Detect IOI circuit
   */
  async detect(params: DetectIOIParams, options?: RequestOptions): Promise<DetectIOIResponse> {
    return request<DetectIOIResponse>('POST', '/ioi/detect', {
      ...options,
      body: {
        trace_id: params.traceId,
        sentence: params.sentence,
        config: {
          use_patching: params.usePatching ?? true,
          head_threshold: params.headThreshold ?? 0.05,
          validate_against_known: params.validateAgainstKnown ?? true,
        },
      },
    })
  },

  /**
   * Get known IOI heads for GPT-2
   */
  async getKnownHeads(options?: RequestOptions) {
    return request('GET', '/ioi/known-heads', options)
  },

  /**
   * Get IOI sentence templates
   */
  async getTemplates(options?: RequestOptions) {
    return request('GET', '/ioi/templates', options)
  },
}

// ============================================================================
// SAE API
// ============================================================================

export interface LoadSAEParams {
  layer: number
  saePath?: string
  saeName?: string
}

export interface EncodeParams {
  traceId: string
  layer: number
  topK?: number
  threshold?: number
}

export const saeApi = {
  /**
   * Load an SAE model
   */
  async load(params: LoadSAEParams, options?: RequestOptions) {
    return request('POST', '/sae/load', {
      ...options,
      body: {
        layer: params.layer,
        sae_path: params.saePath,
        sae_name: params.saeName,
      },
    })
  },

  /**
   * Encode activations through SAE
   */
  async encode(params: EncodeParams, options?: RequestOptions): Promise<EncodeResponse> {
    return request<EncodeResponse>('POST', '/sae/encode', {
      ...options,
      body: {
        trace_id: params.traceId,
        layer: params.layer,
        top_k: params.topK ?? 10,
        threshold: params.threshold ?? 0,
      },
    })
  },

  /**
   * Get top features for a layer
   */
  async getFeatures(layer: number, topK = 100, options?: RequestOptions) {
    return request('GET', `/sae/features/${layer}`, {
      ...options,
      params: { top_k: topK },
    })
  },

  /**
   * Get feature co-activations
   */
  async getCoactivations(traceId: string, layer: number, topK = 50, options?: RequestOptions) {
    return request('GET', `/sae/coactivations/${layer}`, {
      ...options,
      params: { trace_id: traceId, top_k: topK },
    })
  },

  /**
   * List available SAEs
   */
  async listAvailable(options?: RequestOptions) {
    return request('GET', '/sae/available', options)
  },
}

// ============================================================================
// Model API
// ============================================================================

export interface LoadModelParams {
  modelName: string
  device?: string
  dtype?: 'float32' | 'float16' | 'bfloat16'
  forceReload?: boolean
}

export const modelApi = {
  /**
   * List available models
   */
  async list(options?: RequestOptions): Promise<ModelsListResponse> {
    return request<ModelsListResponse>('GET', '/models', options)
  },

  /**
   * Load a model
   */
  async load(params: LoadModelParams, options?: RequestOptions): Promise<ModelInfo> {
    return request<ModelInfo>('POST', '/models/load', {
      ...options,
      body: {
        model_name: params.modelName,
        device: params.device,
        dtype: params.dtype,
        force_reload: params.forceReload ?? false,
      },
    })
  },

  /**
   * Unload current model
   */
  async unload(options?: RequestOptions): Promise<void> {
    return request<void>('POST', '/models/unload', options)
  },

  /**
   * Get current model info
   */
  async getCurrent(options?: RequestOptions): Promise<ModelInfo | null> {
    return request<ModelInfo | null>('GET', '/models/current', options)
  },

  /**
   * Estimate memory requirements
   */
  async estimateMemory(modelName: string, options?: RequestOptions): Promise<MemoryEstimate> {
    return request<MemoryEstimate>('GET', `/models/estimate/${modelName}`, options)
  },
}

// ============================================================================
// Health API
// ============================================================================

export const healthApi = {
  /**
   * Check API health
   */
  async check(): Promise<{ status: string; version: string; environment: string }> {
    const response = await fetch(`${API_BASE_URL}/api/health`)
    return response.json()
  },

  /**
   * Get API configuration
   */
  async getConfig(): Promise<{ defaultModel: string; maxSequenceLength: number; availableFeatures: string[] }> {
    const response = await fetch(`${API_BASE_URL}/api/${API_VERSION}/config`)
    const data = await response.json()
    return data.data
  },
}

// Export all APIs
export const api = {
  trace: traceApi,
  circuit: circuitApi,
  ioi: ioiApi,
  sae: saeApi,
  model: modelApi,
  health: healthApi,
}

export default api
