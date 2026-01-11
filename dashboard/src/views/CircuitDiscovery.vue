<script setup lang="ts">
/**
 * Circuit Discovery View
 *
 * Discover and visualize computational circuits in transformer models.
 */

import { ref, computed, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useCircuitStore } from '@/stores/circuit.store'
import { useTraceStore } from '@/stores/trace.store'
import { useUIStore } from '@/stores/ui.store'
import { BaseCard, BaseSelect, BaseButton, LoadingSpinner } from '@/components/common'
import {
  CircuitGraph,
  CircuitControls,
  CircuitDetails,
  CircuitStats,
} from '@/components/circuit'
import { generateDotGraph } from '@/composables'
import type { Circuit, CircuitNode, CircuitEdge } from '@/types'

// Stores
const circuitStore = useCircuitStore()
const traceStore = useTraceStore()
const uiStore = useUIStore()
const { activeCircuit, circuitList, isDiscovering, error } = storeToRefs(circuitStore)
const { traceList, activeTraceId } = storeToRefs(traceStore)

// Local state for demo circuit (since store expects different flow)
const demoCircuit = ref<Circuit | null>(null)

// Local state
const circuitName = ref('Unnamed Circuit')
const behaviorDesc = ref('')
const cleanTraceId = ref<string | null>(null)
// corruptTraceId removed - not currently used in simplified discovery
const targetTokenIdx = ref(0)
const threshold = ref(0.1)
const layout = ref<'dagre' | 'breadthfirst' | 'cose'>('dagre')
const showLabels = ref(true)
const selectedNode = ref<CircuitNode | null>(null)
const selectedEdge = ref<CircuitEdge | null>(null)
const hoveredNode = ref<CircuitNode | null>(null)
const hoveredEdge = ref<CircuitEdge | null>(null)

// Graph ref for export
const graphRef = ref<InstanceType<typeof CircuitGraph> | null>(null)

// Current circuit (from store or demo)
const currentCircuit = computed((): Circuit | undefined => {
  if (demoCircuit.value) return demoCircuit.value
  return activeCircuit.value?.circuit ?? undefined
})

// Trace options for selectors
const traceOptions = computed(() => {
  return traceList.value.map((trace) => ({
    value: trace.traceId,
    label: trace.inputText.slice(0, 40) + (trace.inputText.length > 40 ? '...' : ''),
  }))
})

// Saved circuit options
const circuitOptions = computed(() => {
  return circuitList.value.map((discovered) => ({
    value: discovered.id,
    label: discovered.circuit.name,
  }))
})

// Filtered edge count
const filteredEdgeCount = computed(() => {
  if (!currentCircuit.value) return 0
  return currentCircuit.value.edges.filter((e: CircuitEdge) => e.importance >= threshold.value).length
})

// Total edge count
const totalEdgeCount = computed(() => {
  return currentCircuit.value?.edges.length ?? 0
})

// Whether discovery can run
const canDiscover = computed(() => {
  return cleanTraceId.value && circuitName.value.trim()
})

// Auto-select active trace
watch(activeTraceId, (id) => {
  if (id && !cleanTraceId.value) {
    cleanTraceId.value = id
  }
})

// Handle circuit discovery
async function handleDiscover(): Promise<void> {
  if (!canDiscover.value || !cleanTraceId.value) return

  // Clear demo circuit when discovering
  demoCircuit.value = null

  try {
    await circuitStore.discoverCircuit(
      cleanTraceId.value,
      targetTokenIdx.value,
      {
        threshold: threshold.value,
      }
    )
    uiStore.notifySuccess('Circuit Discovered', 'Circuit analysis complete.')
  } catch (err) {
    console.error('Circuit discovery failed:', err)
    uiStore.notifyError(
      'Discovery Failed',
      err instanceof Error ? err.message : 'Unable to discover circuit. Please try again.'
    )
  }
}

// Handle DOT export
function handleExportDot(): void {
  if (!currentCircuit.value) return

  try {
    const dotString = generateDotGraph(currentCircuit.value, threshold.value)
    const blob = new Blob([dotString], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.download = `${currentCircuit.value.name || 'circuit'}.dot`
    link.href = url
    link.click()
    URL.revokeObjectURL(url)
    uiStore.notifySuccess('Export Complete', 'DOT file downloaded successfully.')
  } catch (err) {
    uiStore.notifyError('Export Failed', 'Could not export circuit as DOT file.')
  }
}

// Handle PNG export
function handleExportPng(): void {
  try {
    graphRef.value?.exportPng()
    uiStore.notifySuccess('Export Complete', 'PNG image downloaded successfully.')
  } catch (err) {
    uiStore.notifyError('Export Failed', 'Could not export circuit as PNG image.')
  }
}

// Handle clear
function handleClear(): void {
  demoCircuit.value = null
  circuitStore.setActiveCircuit(null)
  selectedNode.value = null
  selectedEdge.value = null
}

// Handle node selection
function handleNodeSelect(node: CircuitNode | null): void {
  selectedNode.value = node
  if (node) selectedEdge.value = null
}

// Handle edge selection
function handleEdgeSelect(edge: CircuitEdge | null): void {
  selectedEdge.value = edge
  if (edge) selectedNode.value = null
}

// Handle node hover
function handleNodeHover(node: CircuitNode | null): void {
  hoveredNode.value = node
}

// Handle edge hover
function handleEdgeHover(edge: CircuitEdge | null): void {
  hoveredEdge.value = edge
}

// Load a saved circuit
function handleLoadCircuit(value: string | number | null): void {
  if (!value || typeof value !== 'string') return
  demoCircuit.value = null
  circuitStore.setActiveCircuit(value)
}

// Demo circuit for testing without backend
function loadDemoCircuit(): void {
  // Clear store circuit
  circuitStore.setActiveCircuit(null)

  demoCircuit.value = {
    id: 'demo-induction',
    name: 'Induction Circuit',
    description: 'A simple induction head circuit for in-context learning',
    behavior: 'Copy patterns seen in context',
    nodes: [
      { id: 'embed', layer: -1, component: 'embed', label: 'Embed' },
      { id: 'L0H0', layer: 0, component: 'attention', head: 0, importance: 0.3 },
      { id: 'L0H1', layer: 0, component: 'attention', head: 1, importance: 0.5 },
      { id: 'L0-mlp', layer: 0, component: 'mlp', importance: 0.2 },
      { id: 'L1H0', layer: 1, component: 'attention', head: 0, importance: 0.7 },
      { id: 'L1H1', layer: 1, component: 'attention', head: 1, importance: 0.9 },
      { id: 'L1-mlp', layer: 1, component: 'mlp', importance: 0.4 },
      { id: 'L2H0', layer: 2, component: 'attention', head: 0, importance: 0.6 },
      { id: 'L2-mlp', layer: 2, component: 'mlp', importance: 0.3 },
      { id: 'unembed', layer: 3, component: 'unembed', label: 'Unembed' },
    ],
    edges: [
      { source: 'embed', target: 'L0H0', importance: 0.4 },
      { source: 'embed', target: 'L0H1', importance: 0.6 },
      { source: 'embed', target: 'L0-mlp', importance: 0.3 },
      { source: 'L0H0', target: 'L1H0', importance: 0.5, type: 'q-composition' },
      { source: 'L0H1', target: 'L1H1', importance: 0.8, type: 'k-composition' },
      { source: 'L0-mlp', target: 'L1H0', importance: 0.3 },
      { source: 'L1H0', target: 'L2H0', importance: 0.6, type: 'v-composition' },
      { source: 'L1H1', target: 'L2H0', importance: 0.9, type: 'induction' },
      { source: 'L1-mlp', target: 'L2H0', importance: 0.2 },
      { source: 'L2H0', target: 'unembed', importance: 0.85 },
      { source: 'L2-mlp', target: 'unembed', importance: 0.4 },
    ],
    avgImportance: 0.52,
  }
}
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
        Circuit Discovery
      </h1>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Discover and visualize computational circuits in transformer models.
      </p>
    </div>

    <!-- Main layout -->
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-4">
      <!-- Left sidebar - Controls -->
      <div class="lg:col-span-1 space-y-4">
        <!-- Discovery inputs -->
        <BaseCard title="Discovery Setup" padding="md">
          <div class="space-y-4">
            <!-- Circuit name -->
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Circuit Name
              </label>
              <input
                v-model="circuitName"
                type="text"
                class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="e.g., Induction Circuit"
              />
            </div>

            <!-- Behavior description -->
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Behavior Description
              </label>
              <textarea
                v-model="behaviorDesc"
                class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                rows="2"
                placeholder="What does this circuit compute?"
              />
            </div>

            <!-- Trace selector -->
            <div>
              <BaseSelect
                v-model="cleanTraceId"
                :options="traceOptions"
                label="Trace"
                placeholder="Select trace"
              />
            </div>

            <!-- Target token index -->
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Target Token Index
              </label>
              <input
                v-model.number="targetTokenIdx"
                type="number"
                min="0"
                class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
        </BaseCard>

        <!-- Visualization controls -->
        <BaseCard title="Visualization" padding="md">
          <CircuitControls
            v-model:threshold="threshold"
            v-model:layout="layout"
            v-model:show-labels="showLabels"
            :is-loading="isDiscovering"
            :has-circuit="!!currentCircuit"
            :edge-count="filteredEdgeCount"
            :total-edges="totalEdgeCount"
            @discover="handleDiscover"
            @export-dot="handleExportDot"
            @export-png="handleExportPng"
            @clear="handleClear"
          />
        </BaseCard>

        <!-- Load saved circuits -->
        <BaseCard v-if="circuitOptions.length > 0" title="Saved Circuits" padding="md">
          <BaseSelect
            :model-value="null"
            :options="circuitOptions"
            placeholder="Load a circuit"
            @update:model-value="handleLoadCircuit"
          />
        </BaseCard>

        <!-- Demo button -->
        <BaseButton
          variant="ghost"
          class="w-full"
          @click="loadDemoCircuit"
        >
          Load Demo Circuit
        </BaseButton>
      </div>

      <!-- Main content - Graph -->
      <div class="lg:col-span-2">
        <BaseCard title="Circuit Graph" subtitle="Interactive circuit visualization" padding="none">
          <!-- Error state -->
          <div v-if="error" class="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg m-4">
            <p class="text-red-600 dark:text-red-400 text-sm">{{ error }}</p>
          </div>

          <!-- Loading state -->
          <div v-else-if="isDiscovering" class="h-125 flex items-center justify-center">
            <div class="text-center">
              <LoadingSpinner size="lg" />
              <p class="mt-2 text-sm text-gray-500">Discovering circuit...</p>
            </div>
          </div>

          <!-- Graph -->
          <CircuitGraph
            v-else
            ref="graphRef"
            :circuit="currentCircuit"
            :threshold="threshold"
            :show-labels="showLabels"
            :layout="layout"
            height="500px"
            @node-select="handleNodeSelect"
            @edge-select="handleEdgeSelect"
            @node-hover="handleNodeHover"
            @edge-hover="handleEdgeHover"
          />
        </BaseCard>

        <!-- Circuit stats -->
        <BaseCard v-if="currentCircuit" title="Circuit Statistics" padding="md" class="mt-4">
          <CircuitStats
            :circuit="currentCircuit"
            :threshold="threshold"
          />
        </BaseCard>
      </div>

      <!-- Right sidebar - Details -->
      <div class="lg:col-span-1">
        <BaseCard title="Details" padding="md">
          <CircuitDetails
            :circuit="currentCircuit"
            :selected-node="selectedNode ?? undefined"
            :selected-edge="selectedEdge ?? undefined"
            :hovered-node="hoveredNode ?? undefined"
            :hovered-edge="hoveredEdge ?? undefined"
          />
        </BaseCard>
      </div>
    </div>
  </div>
</template>
