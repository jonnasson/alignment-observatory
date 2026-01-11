<script setup lang="ts">
/**
 * CircuitGraph Component
 *
 * Interactive circuit visualization using Cytoscape.js with dagre layout.
 */

import { ref, computed, watch, type PropType } from 'vue'
import {
  useCircuitGraph,
  componentColors,
  componentLabels,
} from '@/composables/useCircuitGraph'
import type { Circuit, CircuitNode, CircuitEdge } from '@/types'

const props = defineProps({
  /** Circuit data to visualize */
  circuit: {
    type: Object as PropType<Circuit>,
    default: null,
  },
  /** Edge importance threshold */
  threshold: {
    type: Number,
    default: 0,
  },
  /** Show node labels */
  showLabels: {
    type: Boolean,
    default: true,
  },
  /** Layout algorithm */
  layout: {
    type: String as PropType<'dagre' | 'breadthfirst' | 'cose'>,
    default: 'dagre',
  },
  /** Container height */
  height: {
    type: String,
    default: '500px',
  },
})

const emit = defineEmits<{
  (e: 'nodeSelect', node: CircuitNode | null): void
  (e: 'edgeSelect', edge: CircuitEdge | null): void
  (e: 'nodeHover', node: CircuitNode | null): void
  (e: 'edgeHover', edge: CircuitEdge | null): void
}>()

// Container ref
const containerRef = ref<HTMLElement | null>(null)

// Reactive props for composable
const circuitRef = computed(() => props.circuit)
const thresholdRef = computed(() => props.threshold)
const showLabelsRef = computed(() => props.showLabels)
const layoutRef = computed(() => props.layout)

// Use circuit graph composable
const {
  selectedNode,
  selectedEdge,
  hoveredNode,
  hoveredEdge,
  stats,
  isInitialized,
  initGraph,
  fitGraph,
  resetView,
  exportPng,
} = useCircuitGraph({
  container: containerRef,
  circuit: circuitRef,
  edgeThreshold: thresholdRef,
  showLabels: showLabelsRef,
  layout: layoutRef,
})

// Watch selections and emit events
watch(selectedNode, (node) => emit('nodeSelect', node))
watch(selectedEdge, (edge) => emit('edgeSelect', edge))
watch(hoveredNode, (node) => emit('nodeHover', node))
watch(hoveredEdge, (edge) => emit('edgeHover', edge))

// Initialize graph when circuit changes
watch(() => props.circuit, (newCircuit) => {
  if (newCircuit && containerRef.value && !isInitialized.value) {
    initGraph()
  }
}, { immediate: true })

// Component legend
const legendItems = computed(() => {
  if (!props.circuit) return []

  // Get unique component types in the circuit
  const types = new Set(props.circuit.nodes.map(n => n.component))

  return Array.from(types).map(type => ({
    type,
    label: componentLabels[type],
    color: componentColors[type],
  }))
})

// Export handlers
function handleExportPng(): void {
  const png = exportPng()
  if (png) {
    const link = document.createElement('a')
    link.download = `circuit-${props.circuit?.name || 'graph'}.png`
    link.href = `data:image/png;base64,${png}`
    link.click()
  }
}

// Expose methods for parent component
defineExpose({
  fitGraph,
  resetView,
  exportPng,
})
</script>

<template>
  <div class="circuit-graph relative">
    <!-- Controls overlay -->
    <div class="absolute top-2 right-2 z-10 flex gap-1">
      <button
        class="p-1.5 rounded bg-white/80 dark:bg-gray-800/80 hover:bg-white dark:hover:bg-gray-800 shadow-sm"
        title="Fit to view"
        @click="fitGraph"
      >
        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
        </svg>
      </button>
      <button
        class="p-1.5 rounded bg-white/80 dark:bg-gray-800/80 hover:bg-white dark:hover:bg-gray-800 shadow-sm"
        title="Reset view"
        @click="resetView"
      >
        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </button>
      <button
        class="p-1.5 rounded bg-white/80 dark:bg-gray-800/80 hover:bg-white dark:hover:bg-gray-800 shadow-sm"
        title="Export as PNG"
        @click="handleExportPng"
      >
        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
      </button>
    </div>

    <!-- Stats overlay -->
    <div v-if="stats" class="absolute top-2 left-2 z-10 px-2 py-1 rounded bg-white/80 dark:bg-gray-800/80 shadow-sm text-xs">
      <span class="text-gray-600 dark:text-gray-400">
        {{ stats.nodeCount }} nodes, {{ stats.edgeCount }} edges
      </span>
    </div>

    <!-- Legend -->
    <div v-if="legendItems.length > 0" class="absolute bottom-2 left-2 z-10 px-2 py-1.5 rounded bg-white/90 dark:bg-gray-800/90 shadow-sm">
      <div class="flex flex-wrap gap-2">
        <div
          v-for="item in legendItems"
          :key="item.type"
          class="flex items-center gap-1"
        >
          <div
            class="w-3 h-3 rounded"
            :style="{ backgroundColor: item.color }"
          />
          <span class="text-xs text-gray-600 dark:text-gray-400">
            {{ item.label }}
          </span>
        </div>
      </div>
    </div>

    <!-- Cytoscape container -->
    <div
      ref="containerRef"
      class="cytoscape-container w-full bg-gray-50 dark:bg-gray-900/50 rounded-lg"
      :style="{ height }"
    />

    <!-- Empty state -->
    <div
      v-if="!circuit"
      class="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-500 pointer-events-none"
    >
      <div class="text-center">
        <svg class="w-12 h-12 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
        </svg>
        <p class="text-sm font-medium">No circuit to display</p>
        <p class="text-xs mt-1">Discover a circuit to visualize it here</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cytoscape-container {
  min-height: 300px;
}
</style>
