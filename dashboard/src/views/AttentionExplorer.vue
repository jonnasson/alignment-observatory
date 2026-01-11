<script setup lang="ts">
/**
 * AttentionExplorer View
 *
 * Main view for exploring attention patterns across layers and heads.
 * Supports both 2D heatmap and 3D flow visualization modes.
 */

import { ref, computed, watch, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { BaseCard, BaseSelect, BaseButton, LoadingSpinner } from '@/components/common'
import { AttentionHeatmap, HeadSelector, AttentionStats, ColorLegend } from '@/components/attention'
import { AttentionFlow3D, AttentionFlow3DControls, AttentionFlow3DLegend } from '@/components/attention3d'
import { useTraceStore } from '@/stores/trace.store'
import { useUIStore } from '@/stores/ui.store'
import { useAttentionViz, extractHeadAttention, computeMeanAttention } from '@/composables/useAttentionViz'
import { useKeyboardShortcuts, getVisualizationShortcuts } from '@/composables/useKeyboardShortcuts'
import type { ColorScaleName } from '@/composables/useColorScale'
import type { HeadAnalysis, VisualizationMode, AttentionCellInstance } from '@/types'

// Store
const traceStore = useTraceStore()
const uiStore = useUIStore()
const {
  activeTrace,
  numLayers,
  numHeads,
  tokens,
  selectedLayer,
  selectedHead,
  currentAttention,
  isLoadingAttention,
  error,
} = storeToRefs(traceStore)

// Local state
const colorScale = ref<ColorScaleName>('viridis')
const showLabels = ref(true)
const aggregateMode = ref<'none' | 'mean' | 'max'>('none')

// View mode: 2D heatmap or 3D flow visualization
const viewMode = ref<VisualizationMode>('2d')

// 3D-specific state
const threshold3D = ref(0.1)
const selectedLayers3D = ref<number[]>([])
const selectedHeads3D = ref<number[]>([])
const animate3D = ref(false)
const hoveredCell3D = ref<AttentionCellInstance | null>(null)

// Reference to 3D component
const attention3DRef = ref<InstanceType<typeof AttentionFlow3D> | null>(null)

// Layer/head options for selects
const layerOptions = computed(() =>
  Array.from({ length: numLayers.value }, (_, i) => ({
    value: i,
    label: `Layer ${i}`,
  }))
)

const headOptions = computed(() => {
  const options = [{ value: -1, label: 'All Heads (Mean)' }]
  for (let i = 0; i < numHeads.value; i++) {
    options.push({ value: i, label: `Head ${i}` })
  }
  return options
})

const colorScaleOptions = [
  { value: 'viridis', label: 'Viridis' },
  { value: 'inferno', label: 'Inferno' },
  { value: 'coolwarm', label: 'Cool-Warm' },
  { value: 'blues', label: 'Blues' },
  { value: 'rdbu', label: 'Red-Blue' },
]

// Computed attention matrix
const attentionMatrix = computed((): number[][] => {
  if (!currentAttention.value?.pattern) {
    return []
  }

  const pattern = currentAttention.value.pattern
  if (!pattern.data || pattern.shape.length === 0) {
    return []
  }

  const headIdx = selectedHead.value ?? 0

  if (headIdx < 0) {
    // Compute mean across all heads
    return computeMeanAttention(pattern.data, pattern.shape)
  }

  return extractHeadAttention(pattern.data, pattern.shape, headIdx)
})

// Head info for head selector
const headAnalysisData = computed(() => {
  if (!currentAttention.value?.analysis) return []

  return currentAttention.value.analysis.map((analysis) => ({
    layer: analysis.layer,
    head: analysis.head,
    // Extract classification string from either HeadClassificationInfo or plain string
    classification: typeof analysis.classification === 'string'
      ? analysis.classification
      : analysis.classification?.category,
  }))
})

// Attention visualization composable
const patternRef = computed(() => currentAttention.value?.pattern ?? null)
const tokensRef = computed(() => tokens.value)
const selectedHeadRef = computed(() => selectedHead.value ?? 0)
const analysesRef = computed(() => currentAttention.value?.analysis as HeadAnalysis[] | undefined)

const {
  statistics,
  headClassification,
  seqLen,
} = useAttentionViz({
  pattern: patternRef,
  tokens: tokensRef,
  selectedHead: selectedHeadRef,
  analyses: analysesRef,
})

// Fetch attention when layer/head changes
watch(
  [selectedLayer, selectedHead, () => activeTrace.value?.traceId],
  async () => {
    if (activeTrace.value) {
      try {
        await traceStore.fetchAttention(
          selectedLayer.value,
          selectedHead.value ?? undefined,
          aggregateMode.value
        )
      } catch (e) {
        console.error('Failed to fetch attention:', e)
        uiStore.notifyError(
          'Failed to Load Attention',
          e instanceof Error ? e.message : 'Could not load attention data for this layer/head.'
        )
      }
    }
  },
  { immediate: true }
)

// Handle layer/head selection from HeadSelector
function handleHeadSelect(layer: number, head: number): void {
  selectedLayer.value = layer
  selectedHead.value = head
}

// 3D visualization data (Map of layer -> Float32Array)
// For now, we create mock data based on current attention
// In production, this would fetch all layers' attention data
const attention3DData = computed((): Map<number, Float32Array> => {
  const dataMap = new Map<number, Float32Array>()

  // If we have current attention data, use it for the selected layer
  if (currentAttention.value?.pattern?.data) {
    const pattern = currentAttention.value.pattern
    const data = pattern.data as number[][][][]

    // Flatten the attention data for the current layer
    // Shape is [batch, heads, seq_q, seq_k], we take batch=0
    const seqQ = pattern.shape[2] ?? 0
    const seqK = pattern.shape[3] ?? 0
    const nHeads = pattern.shape[1] ?? 0

    const flatData = new Float32Array(nHeads * seqQ * seqK)
    let idx = 0

    const batchData = data[0]
    if (batchData) {
      for (let h = 0; h < nHeads; h++) {
        const headData = batchData[h]
        if (!headData) continue

        for (let q = 0; q < seqQ; q++) {
          const rowData = headData[q]
          if (!rowData) continue
          for (let k = 0; k < seqK; k++) {
            flatData[idx++] = rowData[k] ?? 0
          }
        }
      }
    }

    dataMap.set(selectedLayer.value, flatData)
  }

  return dataMap
})

// Handle 3D cell hover
function handleCell3DHover(cell: AttentionCellInstance | null): void {
  hoveredCell3D.value = cell
}

// Handle 3D cell click
function handleCell3DClick(cell: AttentionCellInstance): void {
  // Navigate to the 2D view for this layer
  selectedLayer.value = cell.layer
  viewMode.value = '2d'
}

// Handle 3D view preset
function handleSetView3D(_view: 'top' | 'front' | 'side' | 'isometric'): void {
  // TODO: Implement view presets in AttentionFlow3D component
  // For now, just reset camera
  attention3DRef.value?.resetCamera()
}

// Reset 3D view
function resetView3D(): void {
  attention3DRef.value?.resetCamera()
}

// Keyboard shortcuts
const { registerAll } = useKeyboardShortcuts()

onMounted(() => {
  registerAll(
    getVisualizationShortcuts({
      toggleLabels: () => {
        showLabels.value = !showLabels.value
      },
      nextLayer: () => {
        if (selectedLayer.value < numLayers.value - 1) {
          selectedLayer.value++
        }
      },
      prevLayer: () => {
        if (selectedLayer.value > 0) {
          selectedLayer.value--
        }
      },
      nextHead: () => {
        const currentHead = selectedHead.value ?? -1
        if (currentHead < numHeads.value - 1) {
          selectedHead.value = currentHead + 1
        }
      },
      prevHead: () => {
        const currentHead = selectedHead.value ?? 0
        if (currentHead > 0) {
          selectedHead.value = currentHead - 1
        }
      },
    })
  )
})
</script>

<template>
  <div class="space-y-6">
    <!-- Page header -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
        Attention Explorer
      </h1>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Visualize and analyze attention patterns across layers and heads.
      </p>
    </div>

    <!-- No trace loaded message -->
    <BaseCard v-if="!activeTrace" padding="lg">
      <div class="text-center py-8">
        <svg
          class="mx-auto h-12 w-12 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="1.5"
            d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
        <h3 class="mt-4 text-lg font-medium text-gray-900 dark:text-gray-100">
          No trace loaded
        </h3>
        <p class="mt-2 text-sm text-gray-500 dark:text-gray-400">
          Load a trace from the dashboard to visualize attention patterns.
        </p>
        <div class="mt-6">
          <BaseButton variant="primary" @click="$router.push('/')">
            Go to Dashboard
          </BaseButton>
        </div>
      </div>
    </BaseCard>

    <!-- Main content when trace is loaded -->
    <template v-else>
      <!-- Controls row -->
      <BaseCard title="Controls" padding="md">
        <div class="flex flex-wrap gap-4 items-end">
          <!-- View mode toggle -->
          <div class="flex items-center gap-1 border border-gray-300 dark:border-gray-600 rounded-lg p-1">
            <button
              :class="[
                'px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
                viewMode === '2d'
                  ? 'bg-primary-600 text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800',
              ]"
              @click="viewMode = '2d'"
            >
              2D
            </button>
            <button
              :class="[
                'px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
                viewMode === '3d'
                  ? 'bg-primary-600 text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800',
              ]"
              @click="viewMode = '3d'"
            >
              3D
            </button>
          </div>

          <!-- 2D-specific controls -->
          <template v-if="viewMode === '2d'">
            <div class="w-36">
              <BaseSelect
                v-model="selectedLayer"
                :options="layerOptions"
                label="Layer"
                placeholder="Select layer"
              />
            </div>
            <div class="w-40">
              <BaseSelect
                v-model="selectedHead"
                :options="headOptions"
                label="Head"
                placeholder="Select head"
              />
            </div>
            <div class="w-36">
              <BaseSelect
                v-model="colorScale"
                :options="colorScaleOptions"
                label="Color Scale"
              />
            </div>
            <div class="flex items-center gap-2">
              <input
                id="show-labels"
                v-model="showLabels"
                type="checkbox"
                class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <label for="show-labels" class="text-sm text-gray-700 dark:text-gray-300">
                Show labels
              </label>
            </div>
          </template>
        </div>

        <!-- 3D controls (shown when in 3D mode) -->
        <AttentionFlow3DControls
          v-if="viewMode === '3d'"
          class="mt-4"
          :threshold="threshold3D"
          :num-layers="numLayers"
          :num-heads="numHeads"
          :selected-layers="selectedLayers3D"
          :selected-heads="selectedHeads3D"
          :animate="animate3D"
          @update:threshold="threshold3D = $event"
          @update:selected-layers="selectedLayers3D = $event"
          @update:selected-heads="selectedHeads3D = $event"
          @update:animate="animate3D = $event"
          @reset-view="resetView3D"
          @set-view="handleSetView3D"
        />
      </BaseCard>

      <!-- Main visualization area -->
      <div class="grid grid-cols-1 xl:grid-cols-4 gap-4">
        <!-- Visualization panel (3 columns) -->
        <BaseCard
          class="xl:col-span-3"
          :title="viewMode === '2d' ? 'Attention Pattern' : '3D Attention Flow'"
          :subtitle="viewMode === '2d' ? `Layer ${selectedLayer}, Head ${selectedHead ?? 'All'}` : 'All layers'"
          padding="none"
        >
          <!-- Loading state -->
          <div v-if="isLoadingAttention" class="h-96 flex items-center justify-center">
            <LoadingSpinner size="lg" />
          </div>

          <!-- Error state -->
          <div v-else-if="error" class="h-96 flex items-center justify-center text-red-500">
            <div class="text-center">
              <p class="font-medium">Error loading attention data</p>
              <p class="text-sm mt-1">{{ error }}</p>
            </div>
          </div>

          <!-- 2D Heatmap view -->
          <template v-else-if="viewMode === '2d'">
            <div v-if="attentionMatrix.length > 0" class="relative max-h-[80vh] overflow-auto">
              <AttentionHeatmap
                :matrix="attentionMatrix"
                :tokens="tokens"
                :color-scale="colorScale"
                :show-labels="showLabels"
                :cell-size="seqLen > 50 ? 12 : seqLen > 20 ? 18 : 24"
              />

              <!-- Color legend -->
              <div class="sticky bottom-4 float-right mr-4">
                <ColorLegend
                  :color-scale="colorScale"
                  :min="0"
                  :max="1"
                  :width="120"
                  label="Attention"
                />
              </div>
            </div>

            <!-- No data state -->
            <div v-else class="h-96 flex items-center justify-center text-gray-400">
              <div class="text-center">
                <p class="text-lg font-medium">No attention data</p>
                <p class="text-sm mt-2">Select a layer and head to visualize</p>
              </div>
            </div>
          </template>

          <!-- 3D Flow view -->
          <template v-else>
            <div class="p-4">
              <AttentionFlow3D
                ref="attention3DRef"
                :attention-data="attention3DData"
                :tokens="tokens"
                :num-layers="numLayers"
                :num-heads="numHeads"
                :seq-len="seqLen"
                :threshold="threshold3D"
                :selected-layers="selectedLayers3D"
                :selected-heads="selectedHeads3D"
                :color-scale="colorScale"
                :animate="animate3D"
                height="500px"
                show-performance
                @cell-hover="handleCell3DHover"
                @cell-click="handleCell3DClick"
              />

              <!-- 3D Legend -->
              <AttentionFlow3DLegend
                class="mt-4"
                :color-scale="colorScale"
              />

              <!-- Hovered cell info -->
              <div
                v-if="hoveredCell3D"
                class="mt-2 p-2 bg-gray-100 dark:bg-gray-800 rounded text-sm"
              >
                <span class="font-medium">Hovered:</span>
                Layer {{ hoveredCell3D.layer }},
                Query {{ hoveredCell3D.query }},
                Key {{ hoveredCell3D.key }},
                Value: {{ hoveredCell3D.value.toFixed(3) }}
              </div>
            </div>
          </template>
        </BaseCard>

        <!-- Side panel (1 column) -->
        <div class="space-y-4">
          <!-- Head selector grid -->
          <BaseCard title="Head Overview" padding="sm">
            <HeadSelector
              :num-layers="numLayers"
              :num-heads="numHeads"
              :selected-layer="selectedLayer"
              :selected-head="selectedHead ?? 0"
              :head-analyses="headAnalysisData"
              compact
              @select="handleHeadSelect"
            />
          </BaseCard>

          <!-- Statistics -->
          <BaseCard title="Statistics" padding="md">
            <AttentionStats
              v-if="attentionMatrix.length > 0"
              :stats="statistics"
              :classification="headClassification"
              :layer="selectedLayer"
              :head="selectedHead ?? 0"
              :seq-len="seqLen"
            />
            <div v-else class="text-sm text-gray-500 dark:text-gray-400">
              Select a head to view statistics
            </div>
          </BaseCard>

          <!-- Keyboard shortcuts hint -->
          <BaseCard title="Shortcuts" padding="sm">
            <div class="text-xs text-gray-500 dark:text-gray-400 space-y-1">
              <div class="flex justify-between">
                <span>Toggle labels</span>
                <kbd class="px-1 bg-gray-100 dark:bg-gray-800 rounded">L</kbd>
              </div>
              <div class="flex justify-between">
                <span>Navigate layers</span>
                <kbd class="px-1 bg-gray-100 dark:bg-gray-800 rounded">Up/Down</kbd>
              </div>
              <div class="flex justify-between">
                <span>Navigate heads</span>
                <kbd class="px-1 bg-gray-100 dark:bg-gray-800 rounded">Left/Right</kbd>
              </div>
            </div>
          </BaseCard>
        </div>
      </div>

      <!-- Token info row -->
      <BaseCard v-if="tokens.length > 0" title="Tokens" padding="md">
        <div class="flex flex-wrap gap-1">
          <span
            v-for="(token, idx) in tokens"
            :key="idx"
            class="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300"
            :title="`Position ${idx}`"
          >
            {{ token }}
          </span>
        </div>
      </BaseCard>
    </template>
  </div>
</template>
