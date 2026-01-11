/**
 * Trace Store
 *
 * Manages activation traces, attention patterns, and activation data.
 */

import { defineStore } from 'pinia'
import { ref, computed, shallowRef } from 'vue'
import { traceApi, type TraceInfo } from '@/services/api.client'
import type { TensorData, HeadAnalysis } from '@/types'

export interface AttentionData {
  pattern: TensorData<number[] | number[][][][]>
  tokens: string[]
  analysis?: HeadAnalysis[]
}

export interface ActivationData {
  activations: TensorData
  tokenNorms: number[]
}

export const useTraceStore = defineStore('trace', () => {
  // ============================================================================
  // State
  // ============================================================================

  // All loaded traces
  const traces = ref<Map<string, TraceInfo>>(new Map())

  // Currently active trace
  const activeTraceId = ref<string | null>(null)

  // Selected layer and head for visualization
  const selectedLayer = ref<number>(0)
  const selectedHead = ref<number | null>(null)
  const selectedComponent = ref<'residual' | 'attention_out' | 'mlp_out'>('residual')

  // Cached attention data (shallow ref for performance with large arrays)
  const attentionCache = shallowRef<Map<string, AttentionData>>(new Map())

  // Cached activation data
  const activationCache = shallowRef<Map<string, ActivationData>>(new Map())

  // Loading states
  const isLoading = ref(false)
  const isLoadingAttention = ref(false)
  const isLoadingActivations = ref(false)

  // Error state
  const error = ref<string | null>(null)

  // ============================================================================
  // Getters
  // ============================================================================

  const activeTrace = computed(() => {
    if (!activeTraceId.value) return null
    return traces.value.get(activeTraceId.value) ?? null
  })

  const traceList = computed(() => Array.from(traces.value.values()))

  const numLayers = computed(() => activeTrace.value?.metadata.numLayers ?? 0)

  const numHeads = computed(() => activeTrace.value?.metadata.numHeads ?? 0)

  const tokens = computed(() => activeTrace.value?.tokens ?? [])

  const currentAttention = computed(() => {
    if (!activeTraceId.value) return null
    const key = `${activeTraceId.value}-${selectedLayer.value}-${selectedHead.value ?? 'all'}`
    return attentionCache.value.get(key) ?? null
  })

  const currentActivations = computed(() => {
    if (!activeTraceId.value) return null
    const key = `${activeTraceId.value}-${selectedLayer.value}-${selectedComponent.value}`
    return activationCache.value.get(key) ?? null
  })

  // ============================================================================
  // Actions
  // ============================================================================

  /**
   * Create a new trace via live inference
   */
  async function createTrace(
    text: string,
    options?: {
      modelName?: string
      includeAttention?: boolean
      includeActivations?: boolean
      layers?: number[]
    }
  ): Promise<TraceInfo> {
    isLoading.value = true
    error.value = null

    try {
      const trace = await traceApi.create({
        text,
        modelName: options?.modelName,
        includeAttention: options?.includeAttention ?? true,
        includeActivations: options?.includeActivations ?? true,
        layers: options?.layers,
      })

      traces.value.set(trace.traceId, trace)
      activeTraceId.value = trace.traceId
      selectedLayer.value = 0
      selectedHead.value = null

      return trace
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to create trace'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Load a pre-computed trace from disk
   */
  async function loadTrace(path: string): Promise<TraceInfo> {
    isLoading.value = true
    error.value = null

    try {
      const trace = await traceApi.load({ path })
      traces.value.set(trace.traceId, trace)
      activeTraceId.value = trace.traceId
      selectedLayer.value = 0
      selectedHead.value = null

      return trace
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to load trace'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Fetch all traces from API
   */
  async function fetchTraces(): Promise<void> {
    isLoading.value = true
    error.value = null

    try {
      const traceList = await traceApi.list()
      traces.value.clear()
      traceList.forEach((trace) => {
        traces.value.set(trace.traceId, trace)
      })
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch traces'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Set the active trace
   */
  function setActiveTrace(traceId: string | null): void {
    activeTraceId.value = traceId
    if (traceId) {
      selectedLayer.value = 0
      selectedHead.value = null
    }
  }

  /**
   * Delete a trace
   */
  async function deleteTrace(traceId: string): Promise<void> {
    try {
      await traceApi.delete(traceId)
      traces.value.delete(traceId)

      // Clear caches for this trace
      const newAttentionCache = new Map(attentionCache.value)
      const newActivationCache = new Map(activationCache.value)

      for (const key of newAttentionCache.keys()) {
        if (key.startsWith(traceId)) {
          newAttentionCache.delete(key)
        }
      }
      for (const key of newActivationCache.keys()) {
        if (key.startsWith(traceId)) {
          newActivationCache.delete(key)
        }
      }

      attentionCache.value = newAttentionCache
      activationCache.value = newActivationCache

      if (activeTraceId.value === traceId) {
        activeTraceId.value = null
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to delete trace'
      throw err
    }
  }

  /**
   * Set selected layer
   */
  function setSelectedLayer(layer: number): void {
    selectedLayer.value = layer
  }

  /**
   * Set selected head
   */
  function setSelectedHead(head: number | null): void {
    selectedHead.value = head
  }

  /**
   * Set selected component
   */
  function setSelectedComponent(component: 'residual' | 'attention_out' | 'mlp_out'): void {
    selectedComponent.value = component
  }

  /**
   * Fetch attention pattern for current selection
   */
  async function fetchAttention(
    layer?: number,
    head?: number | null,
    aggregate: 'none' | 'mean' | 'max' = 'none'
  ): Promise<AttentionData | null> {
    if (!activeTraceId.value) return null

    const targetLayer = layer ?? selectedLayer.value
    const targetHead = head !== undefined ? head : selectedHead.value

    const cacheKey = `${activeTraceId.value}-${targetLayer}-${targetHead ?? 'all'}`

    // Return cached if available
    if (attentionCache.value.has(cacheKey)) {
      return attentionCache.value.get(cacheKey)!
    }

    isLoadingAttention.value = true

    try {
      const response = await traceApi.getAttention(
        activeTraceId.value,
        targetLayer,
        targetHead ?? undefined,
        aggregate
      )

      // Extract pattern from response (API returns single pattern, not indexed by layer)
      const patternData = response.pattern
      const analysisData = response.analysis

      // Debug logging for attention data
      if (!patternData?.pattern?.data || (Array.isArray(patternData.pattern.data) && patternData.pattern.data.length === 0)) {
        console.warn('[trace.store] Empty attention pattern received:', {
          layer: targetLayer,
          head: targetHead,
          hasPattern: !!patternData?.pattern,
          shape: patternData?.pattern?.shape,
        })
      }

      const data: AttentionData = {
        pattern: patternData?.pattern ?? { data: [], shape: [], dtype: 'float32' },
        tokens: patternData?.tokens ?? [],
        analysis: analysisData,
      }

      // Update cache
      const newCache = new Map(attentionCache.value)
      newCache.set(cacheKey, data)
      attentionCache.value = newCache

      return data
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch attention'
      throw err
    } finally {
      isLoadingAttention.value = false
    }
  }

  /**
   * Fetch activations for current selection
   */
  async function fetchActivations(
    layer?: number,
    component?: 'residual' | 'attention_out' | 'mlp_out',
    tokenIndices?: number[]
  ): Promise<ActivationData | null> {
    if (!activeTraceId.value) return null

    const targetLayer = layer ?? selectedLayer.value
    const targetComponent = component ?? selectedComponent.value

    const cacheKey = `${activeTraceId.value}-${targetLayer}-${targetComponent}`

    // Return cached if available (unless specific tokens requested)
    if (!tokenIndices && activationCache.value.has(cacheKey)) {
      return activationCache.value.get(cacheKey)!
    }

    isLoadingActivations.value = true

    try {
      const response = await traceApi.getActivations(
        activeTraceId.value,
        targetLayer,
        targetComponent,
        tokenIndices
      )

      // Build TensorData from response
      const tensorData: TensorData = {
        data: response.data ?? [],
        shape: response.shape,
        dtype: response.dtype as 'float32' | 'float64' | 'int32' | 'int64',
      }

      const data: ActivationData = {
        activations: tensorData,
        tokenNorms: [], // Compute from data if needed
      }

      // Update cache (only if not filtered by token indices)
      if (!tokenIndices) {
        const newCache = new Map(activationCache.value)
        newCache.set(cacheKey, data)
        activationCache.value = newCache
      }

      return data
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch activations'
      throw err
    } finally {
      isLoadingActivations.value = false
    }
  }

  /**
   * Clear all caches
   */
  function clearCaches(): void {
    attentionCache.value = new Map()
    activationCache.value = new Map()
  }

  /**
   * Clear error
   */
  function clearError(): void {
    error.value = null
  }

  return {
    // State
    traces,
    activeTraceId,
    selectedLayer,
    selectedHead,
    selectedComponent,
    isLoading,
    isLoadingAttention,
    isLoadingActivations,
    error,

    // Getters
    activeTrace,
    traceList,
    numLayers,
    numHeads,
    tokens,
    currentAttention,
    currentActivations,

    // Actions
    createTrace,
    loadTrace,
    fetchTraces,
    setActiveTrace,
    deleteTrace,
    setSelectedLayer,
    setSelectedHead,
    setSelectedComponent,
    fetchAttention,
    fetchActivations,
    clearCaches,
    clearError,
  }
})
