<script setup lang="ts">
/**
 * CircuitStats Component
 *
 * Displays summary statistics for a circuit.
 */

import { computed, type PropType } from 'vue'
import type { Circuit } from '@/types'
import { componentLabels } from '@/composables/useCircuitGraph'

const props = defineProps({
  /** Circuit to show stats for */
  circuit: {
    type: Object as PropType<Circuit>,
    required: true,
  },
  /** Current edge threshold */
  threshold: {
    type: Number,
    default: 0,
  },
})

// Compute stats
const stats = computed(() => {
  const c = props.circuit
  const filteredEdges = c.edges.filter(e => e.importance >= props.threshold)
  const importances = filteredEdges.map(e => e.importance)

  // Layer range
  const layers = c.nodes.map(n => n.layer)
  const minLayer = Math.min(...layers)
  const maxLayer = Math.max(...layers)

  // Component breakdown
  const componentCounts = new Map<string, number>()
  for (const node of c.nodes) {
    const count = componentCounts.get(node.component) || 0
    componentCounts.set(node.component, count + 1)
  }

  return {
    nodeCount: c.nodes.length,
    edgeCount: filteredEdges.length,
    totalEdges: c.edges.length,
    avgImportance: importances.length > 0 ? importances.reduce((a, b) => a + b, 0) / importances.length : 0,
    maxImportance: importances.length > 0 ? Math.max(...importances) : 0,
    minImportance: importances.length > 0 ? Math.min(...importances) : 0,
    layerRange: { min: minLayer, max: maxLayer },
    componentBreakdown: Array.from(componentCounts.entries()).map(([type, count]) => ({
      type,
      label: componentLabels[type as keyof typeof componentLabels] || type,
      count,
    })),
  }
})

// Format percentage
function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}
</script>

<template>
  <div class="circuit-stats">
    <!-- Primary stats grid -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
      <div class="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20">
        <div class="text-2xl font-bold text-blue-700 dark:text-blue-300">
          {{ stats.nodeCount }}
        </div>
        <div class="text-xs text-blue-600 dark:text-blue-400">Nodes</div>
      </div>

      <div class="p-3 rounded-lg bg-green-50 dark:bg-green-900/20">
        <div class="text-2xl font-bold text-green-700 dark:text-green-300">
          {{ stats.edgeCount }}
        </div>
        <div class="text-xs text-green-600 dark:text-green-400">
          Edges <span v-if="stats.edgeCount !== stats.totalEdges" class="opacity-60">(of {{ stats.totalEdges }})</span>
        </div>
      </div>

      <div class="p-3 rounded-lg bg-purple-50 dark:bg-purple-900/20">
        <div class="text-2xl font-bold text-purple-700 dark:text-purple-300">
          {{ stats.layerRange.max - stats.layerRange.min + 1 }}
        </div>
        <div class="text-xs text-purple-600 dark:text-purple-400">
          Layers ({{ stats.layerRange.min }}-{{ stats.layerRange.max }})
        </div>
      </div>

      <div class="p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20">
        <div class="text-2xl font-bold text-amber-700 dark:text-amber-300">
          {{ formatPercent(stats.avgImportance) }}
        </div>
        <div class="text-xs text-amber-600 dark:text-amber-400">Avg Importance</div>
      </div>
    </div>

    <!-- Importance range -->
    <div class="mb-4">
      <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
        <span>Importance Range</span>
        <span>{{ formatPercent(stats.minImportance) }} - {{ formatPercent(stats.maxImportance) }}</span>
      </div>
      <div class="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          class="h-full bg-gradient-to-r from-blue-300 to-blue-600 rounded-full"
          :style="{
            marginLeft: `${stats.minImportance * 100}%`,
            width: `${(stats.maxImportance - stats.minImportance) * 100}%`,
          }"
        />
      </div>
    </div>

    <!-- Component breakdown -->
    <div>
      <div class="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
        Component Breakdown
      </div>
      <div class="flex flex-wrap gap-2">
        <div
          v-for="item in stats.componentBreakdown"
          :key="item.type"
          class="px-2 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-xs"
        >
          <span class="font-medium text-gray-700 dark:text-gray-300">{{ item.count }}</span>
          <span class="text-gray-500 dark:text-gray-400 ml-1">{{ item.label }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
