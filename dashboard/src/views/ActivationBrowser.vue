<script setup lang="ts">
/**
 * Activation Browser View
 *
 * Browse and analyze layer activations across the model.
 * Shows activation heatmaps, token norms, layer statistics, and top dimensions.
 */

import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useTraceStore } from '@/stores/trace.store'
import { BaseCard, BaseSelect, BaseButton, LoadingSpinner } from '@/components/common'
import {
  ActivationHeatmap,
  TokenNormChart,
  LayerStats,
  DimensionList,
} from '@/components/activation'
import { ColorLegend } from '@/components/attention'
import {
  computeTokenNorms,
  computeDimensionStats,
  extractActivationMatrix,
} from '@/composables'
import type { ColorScaleName } from '@/composables'

// Store
const traceStore = useTraceStore()
const {
  activeTrace,
  tokens,
  currentActivations,
  isLoading,
  isLoadingActivations,
  error,
  numLayers,
} = storeToRefs(traceStore)

// Local state
const selectedLayer = ref<number>(0)
const selectedComponent = ref<'residual' | 'attention_out' | 'mlp_out'>('residual')
const selectedToken = ref<number | null>(null)
const selectedDimension = ref<number | null>(null)
const hoveredToken = ref<number | null>(null)
const hoveredDimension = ref<number | null>(null)
const colorScale = ref<ColorScaleName>('rdbu')
const dimSortBy = ref<'variance' | 'maxAbs'>('variance')
const dimStart = ref(0)
const maxDisplayDims = ref(768)

// Options
const layerOptions = computed(() => {
  const layers = numLayers.value || 12
  return Array.from({ length: layers }, (_, i) => ({
    value: i,
    label: `Layer ${i}`,
  }))
})

const componentOptions = [
  { value: 'residual', label: 'Residual Stream' },
  { value: 'attention_out', label: 'Attention Output' },
  { value: 'mlp_out', label: 'MLP Output' },
]

const colorScaleOptions = [
  { value: 'rdbu', label: 'Red-Blue (Diverging)' },
  { value: 'coolwarm', label: 'Cool-Warm (Diverging)' },
  { value: 'viridis', label: 'Viridis' },
  { value: 'inferno', label: 'Inferno' },
  { value: 'blues', label: 'Blues' },
]

const sortByOptions = [
  { value: 'variance', label: 'Variance' },
  { value: 'maxAbs', label: 'Max Absolute Value' },
]

// Get raw activation data as 2D matrix
const activationMatrix = computed((): number[][] | null => {
  if (!currentActivations.value?.activations) return null
  const tensorData = currentActivations.value.activations
  return extractActivationMatrix(tensorData)
})

// Slice for display (virtual scrolling)
const displayMatrix = computed((): number[][] | null => {
  if (!activationMatrix.value) return null
  return activationMatrix.value.map(row =>
    row.slice(dimStart.value, dimStart.value + maxDisplayDims.value)
  )
})

// Token norms
const tokenNorms = computed(() => {
  if (!activationMatrix.value) return []
  return computeTokenNorms(activationMatrix.value)
})

// Layer statistics
const layerStats = computed(() => {
  if (!activationMatrix.value) return null

  let totalSum = 0
  let totalSumSq = 0
  let totalMin = Infinity
  let totalMax = -Infinity
  let totalCount = 0
  let zeroCount = 0

  for (const row of activationMatrix.value) {
    for (const val of row) {
      totalSum += val
      totalSumSq += val * val
      totalMin = Math.min(totalMin, val)
      totalMax = Math.max(totalMax, val)
      totalCount++
      if (Math.abs(val) < 1e-6) zeroCount++
    }
  }

  const mean = totalSum / totalCount
  const variance = totalSumSq / totalCount - mean * mean
  const l2Norm = Math.sqrt(totalSumSq)
  const sparsity = zeroCount / totalCount

  return {
    mean,
    variance,
    min: totalMin,
    max: totalMax,
    l2Norm,
    sparsity,
  }
})

// Dimension statistics for list
const dimensionStats = computed(() => {
  if (!activationMatrix.value) return []
  const stats = computeDimensionStats(activationMatrix.value)
  // Sort by selected metric
  return [...stats].sort((a, b) => {
    if (dimSortBy.value === 'maxAbs') {
      return b.maxAbs - a.maxAbs
    }
    return b.variance - a.variance
  })
})

// Data range for color legend
const dataRange = computed(() => {
  if (!displayMatrix.value) return { min: -1, max: 1 }
  let min = Infinity
  let max = -Infinity
  for (const row of displayMatrix.value) {
    for (const val of row) {
      min = Math.min(min, val)
      max = Math.max(max, val)
    }
  }
  // Make symmetric for diverging scale
  const absMax = Math.max(Math.abs(min), Math.abs(max))
  return { min: -absMax, max: absMax }
})

// Hidden dimension info
const hiddenDim = computed(() => {
  if (!activationMatrix.value?.[0]) return 0
  return activationMatrix.value[0].length
})

// Load activations when layer/component changes
watch(
  [selectedLayer, selectedComponent],
  async ([layer, component]) => {
    if (activeTrace.value) {
      traceStore.setSelectedLayer(layer)
      traceStore.setSelectedComponent(component)
      await traceStore.fetchActivations(layer, component)
    }
  },
  { immediate: true }
)

// Event handlers
function handleCellHover(cell: { tokenIdx: number; dimIdx: number; value: number } | null): void {
  if (cell) {
    hoveredToken.value = cell.tokenIdx
    hoveredDimension.value = cell.dimIdx
  } else {
    hoveredToken.value = null
    hoveredDimension.value = null
  }
}

function handleCellClick(cell: { tokenIdx: number; dimIdx: number; value: number }): void {
  selectedToken.value = cell.tokenIdx
  selectedDimension.value = cell.dimIdx
}

function handleTokenClick(tokenIdx: number): void {
  selectedToken.value = tokenIdx
}

function handleTokenHover(tokenIdx: number | null): void {
  hoveredToken.value = tokenIdx
}

function handleDimensionClick(dimIdx: number): void {
  selectedDimension.value = dimIdx
}

function handleDimensionHover(dimIdx: number | null): void {
  hoveredDimension.value = dimIdx
}

// Keyboard navigation
function handleKeyDown(event: KeyboardEvent): void {
  // Skip if user is typing in an input
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
    return
  }

  if (event.key === 'ArrowLeft') {
    event.preventDefault()
    if (selectedLayer.value > 0) {
      selectedLayer.value--
    }
  } else if (event.key === 'ArrowRight') {
    event.preventDefault()
    const maxLayer = (numLayers.value || 12) - 1
    if (selectedLayer.value < maxLayer) {
      selectedLayer.value++
    }
  } else if (event.key === 'r') {
    selectedComponent.value = 'residual'
  } else if (event.key === 'a') {
    selectedComponent.value = 'attention_out'
  } else if (event.key === 'm') {
    selectedComponent.value = 'mlp_out'
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
})

// Scroll dimension range
function scrollDimensions(direction: 'left' | 'right'): void {
  const step = Math.floor(maxDisplayDims.value / 2)
  if (direction === 'left') {
    dimStart.value = Math.max(0, dimStart.value - step)
  } else {
    dimStart.value = Math.min(hiddenDim.value - maxDisplayDims.value, dimStart.value + step)
  }
}
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
        Activation Browser
      </h1>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Browse and analyze layer activations across the model.
        <span class="text-gray-400 dark:text-gray-500">
          Use arrow keys to navigate layers, R/A/M to switch components.
        </span>
      </p>
    </div>

    <!-- Controls -->
    <BaseCard title="Controls" padding="md">
      <div class="flex flex-wrap gap-4 items-end">
        <div class="w-32">
          <BaseSelect
            v-model="selectedLayer"
            :options="layerOptions"
            label="Layer"
            placeholder="Select layer"
          />
        </div>
        <div class="w-44">
          <BaseSelect
            v-model="selectedComponent"
            :options="componentOptions"
            label="Component"
            placeholder="Select component"
          />
        </div>
        <div class="w-48">
          <BaseSelect
            v-model="colorScale"
            :options="colorScaleOptions"
            label="Color Scale"
          />
        </div>
        <div class="w-40">
          <BaseSelect
            v-model="dimSortBy"
            :options="sortByOptions"
            label="Sort Dimensions By"
          />
        </div>
      </div>
    </BaseCard>

    <!-- Loading / Error states -->
    <div v-if="isLoading || isLoadingActivations" class="flex justify-center py-12">
      <LoadingSpinner size="lg" />
    </div>

    <div v-else-if="error" class="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
      <p class="text-red-600 dark:text-red-400">{{ error }}</p>
    </div>

    <div v-else-if="!activeTrace" class="p-8 text-center text-gray-500 dark:text-gray-400">
      <p class="text-lg">No trace loaded</p>
      <p class="text-sm mt-2">Load a trace from the Dashboard to visualize activations</p>
    </div>

    <!-- Main content -->
    <template v-else>
      <!-- Activation Heatmap -->
      <BaseCard
        title="Activation Heatmap"
        :subtitle="`Layer ${selectedLayer} - ${componentOptions.find(o => o.value === selectedComponent)?.label ?? selectedComponent}`"
        padding="none"
      >
        <template #actions>
          <div class="flex items-center gap-2">
            <!-- Dimension range controls -->
            <div class="flex items-center gap-1 text-xs text-gray-500">
              <BaseButton
                variant="ghost"
                size="sm"
                :disabled="dimStart === 0"
                @click="scrollDimensions('left')"
              >
                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
              </BaseButton>
              <span>{{ dimStart }}-{{ Math.min(dimStart + maxDisplayDims, hiddenDim) }}</span>
              <BaseButton
                variant="ghost"
                size="sm"
                :disabled="dimStart + maxDisplayDims >= hiddenDim"
                @click="scrollDimensions('right')"
              >
                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                </svg>
              </BaseButton>
            </div>

            <!-- Color legend -->
            <ColorLegend
              :color-scale="colorScale"
              :min="dataRange.min"
              :max="dataRange.max"
              orientation="horizontal"
              :width="120"
              :height="16"
            />
          </div>
        </template>

        <div v-if="displayMatrix" class="p-4">
          <ActivationHeatmap
            :matrix="displayMatrix"
            :tokens="tokens"
            :color-scale="colorScale"
            :diverging="true"
            :cell-height="20"
            :cell-width="2"
            :show-labels="true"
            :dim-start="dimStart"
            :max-dimensions="maxDisplayDims"
            @cell-hover="handleCellHover"
            @cell-click="handleCellClick"
            @token-click="handleTokenClick"
          />
        </div>
        <div v-else class="h-64 flex items-center justify-center text-gray-400">
          <p>No activation data available for this layer/component</p>
        </div>
      </BaseCard>

      <!-- Token Norms & Layer Stats Row -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <!-- Token Norms Chart -->
        <BaseCard title="Token L2 Norms" subtitle="Activation magnitude per token" class="lg:col-span-2">
          <div v-if="tokenNorms.length > 0">
            <TokenNormChart
              :norms="tokenNorms"
              :tokens="tokens"
              :height="200"
              :selected-token="selectedToken ?? undefined"
              :show-labels="tokens.length <= 50"
              @token-click="handleTokenClick"
              @token-hover="handleTokenHover"
            />
          </div>
          <div v-else class="h-48 flex items-center justify-center text-gray-400">
            <p>No data available</p>
          </div>
        </BaseCard>

        <!-- Layer Statistics -->
        <BaseCard title="Layer Statistics">
          <LayerStats
            v-if="layerStats"
            :stats="layerStats"
            :layer="selectedLayer"
            :component="selectedComponent"
            :seq-len="tokens.length"
            :hidden-dim="hiddenDim"
          />
          <div v-else class="text-sm text-gray-400">
            No statistics available
          </div>
        </BaseCard>
      </div>

      <!-- Dimension Analysis -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <!-- Top Dimensions -->
        <BaseCard title="Top Dimensions">
          <DimensionList
            v-if="dimensionStats.length > 0"
            :dimensions="dimensionStats"
            :sort-by="dimSortBy"
            :selected-dimension="selectedDimension ?? undefined"
            :max-items="15"
            @dimension-click="handleDimensionClick"
            @dimension-hover="handleDimensionHover"
          />
          <div v-else class="text-sm text-gray-400">
            No dimension data available
          </div>
        </BaseCard>

        <!-- Selected Info -->
        <BaseCard title="Selection Info">
          <div class="space-y-4">
            <!-- Selected Token -->
            <div v-if="selectedToken !== null" class="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20">
              <div class="text-sm font-medium text-blue-900 dark:text-blue-100">
                Selected Token
              </div>
              <div class="mt-1 font-mono text-lg text-blue-700 dark:text-blue-300">
                {{ tokens[selectedToken] ?? `[${selectedToken}]` }}
              </div>
              <div class="mt-1 text-xs text-blue-600 dark:text-blue-400">
                Position: {{ selectedToken }} | L2 Norm: {{ tokenNorms[selectedToken]?.toFixed(3) ?? 'N/A' }}
              </div>
            </div>

            <!-- Selected Dimension -->
            <div v-if="selectedDimension !== null" class="p-3 rounded-lg bg-purple-50 dark:bg-purple-900/20">
              <div class="text-sm font-medium text-purple-900 dark:text-purple-100">
                Selected Dimension
              </div>
              <div class="mt-1 font-mono text-lg text-purple-700 dark:text-purple-300">
                d{{ selectedDimension }}
              </div>
              <div v-if="dimensionStats.find(d => d.dimIdx === selectedDimension)" class="mt-1 text-xs text-purple-600 dark:text-purple-400">
                Variance: {{ dimensionStats.find(d => d.dimIdx === selectedDimension)?.variance.toFixed(4) }}
                | Max: {{ dimensionStats.find(d => d.dimIdx === selectedDimension)?.maxAbs.toFixed(4) }}
              </div>
            </div>

            <!-- Hover Info -->
            <div v-if="hoveredToken !== null || hoveredDimension !== null" class="p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
              <div class="text-xs text-gray-500 dark:text-gray-400">
                Hovering
              </div>
              <div v-if="hoveredToken !== null" class="text-sm text-gray-700 dark:text-gray-300">
                Token: {{ tokens[hoveredToken] ?? `[${hoveredToken}]` }}
              </div>
              <div v-if="hoveredDimension !== null" class="text-sm text-gray-700 dark:text-gray-300">
                Dimension: d{{ hoveredDimension }}
              </div>
            </div>

            <!-- Empty state -->
            <div v-if="selectedToken === null && selectedDimension === null && hoveredToken === null && hoveredDimension === null" class="text-sm text-gray-400">
              Click on the heatmap or lists to select tokens and dimensions
            </div>
          </div>
        </BaseCard>
      </div>
    </template>
  </div>
</template>
