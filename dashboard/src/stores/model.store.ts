/**
 * Model Store
 *
 * Manages model loading, selection, and lifecycle.
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { modelApi } from '@/services/api.client'
import type { ModelInfo, MemoryEstimate } from '@/types'

export const useModelStore = defineStore('model', () => {
  // ============================================================================
  // State
  // ============================================================================

  // Available models
  const availableModels = ref<ModelInfo[]>([])

  // Currently loaded model
  const currentModel = ref<ModelInfo | null>(null)

  // Memory estimates cache
  const memoryEstimates = ref<Map<string, MemoryEstimate>>(new Map())

  // Loading states
  const isLoading = ref(false)
  const isLoadingModel = ref(false)

  // Error state
  const error = ref<string | null>(null)

  // ============================================================================
  // Getters
  // ============================================================================

  const isModelLoaded = computed(() => currentModel.value !== null)

  const currentModelName = computed(() => currentModel.value?.name ?? null)

  const modelConfig = computed(() => {
    if (!currentModel.value) return null
    return {
      numLayers: currentModel.value.numLayers,
      numHeads: currentModel.value.numHeads,
      hiddenSize: currentModel.value.hiddenSize,
      vocabSize: currentModel.value.vocabSize,
      maxSeqLength: currentModel.value.maxSeqLength ?? 1024,
    }
  })

  // ============================================================================
  // Actions
  // ============================================================================

  /**
   * Fetch available models from API
   */
  async function fetchModels(): Promise<void> {
    isLoading.value = true
    error.value = null

    try {
      const response = await modelApi.list()
      availableModels.value = response.models

      // Update current model if one is loaded
      if (response.currentModel) {
        currentModel.value = response.models.find((m) => m.name === response.currentModel) ?? null
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch models'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Load a model
   */
  async function loadModel(
    modelName: string,
    options?: {
      device?: string
      dtype?: 'float32' | 'float16' | 'bfloat16'
      forceReload?: boolean
    }
  ): Promise<ModelInfo> {
    isLoadingModel.value = true
    error.value = null

    try {
      const model = await modelApi.load({
        modelName,
        device: options?.device,
        dtype: options?.dtype,
        forceReload: options?.forceReload ?? false,
      })

      currentModel.value = model

      // Update available models list
      const idx = availableModels.value.findIndex((m) => m.name === modelName)
      if (idx >= 0) {
        availableModels.value[idx] = model
      }

      return model
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to load model'
      throw err
    } finally {
      isLoadingModel.value = false
    }
  }

  /**
   * Unload current model
   */
  async function unloadModel(): Promise<void> {
    if (!currentModel.value) return

    isLoadingModel.value = true
    error.value = null

    try {
      await modelApi.unload()
      currentModel.value = null

      // Update available models list
      availableModels.value = availableModels.value.map((m) => ({
        ...m,
        loaded: false,
        loadedAt: undefined,
      }))
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to unload model'
      throw err
    } finally {
      isLoadingModel.value = false
    }
  }

  /**
   * Get memory estimate for a model
   */
  async function getMemoryEstimate(modelName: string): Promise<MemoryEstimate> {
    // Check cache
    if (memoryEstimates.value.has(modelName)) {
      return memoryEstimates.value.get(modelName)!
    }

    try {
      const estimate = await modelApi.estimateMemory(modelName)
      memoryEstimates.value.set(modelName, estimate)
      return estimate
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to estimate memory'
      throw err
    }
  }

  /**
   * Check current model status
   */
  async function checkCurrentModel(): Promise<void> {
    try {
      const model = await modelApi.getCurrent()
      currentModel.value = model
    } catch {
      currentModel.value = null
    }
  }

  /**
   * Clear error
   */
  function clearError(): void {
    error.value = null
  }

  return {
    // State
    availableModels,
    currentModel,
    memoryEstimates,
    isLoading,
    isLoadingModel,
    error,

    // Getters
    isModelLoaded,
    currentModelName,
    modelConfig,

    // Actions
    fetchModels,
    loadModel,
    unloadModel,
    getMemoryEstimate,
    checkCurrentModel,
    clearError,
  }
})
