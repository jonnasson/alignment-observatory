<script setup lang="ts">
/**
 * CircuitDetails Component
 *
 * Shows detailed information about selected nodes or edges.
 */

import { computed, type PropType } from 'vue'
import { componentColors, componentLabels } from '@/composables/useCircuitGraph'
import type { Circuit, CircuitNode, CircuitEdge, CircuitComponentType } from '@/types'

const props = defineProps({
  /** The full circuit for context */
  circuit: {
    type: Object as PropType<Circuit>,
    default: null,
  },
  /** Currently selected node */
  selectedNode: {
    type: Object as PropType<CircuitNode>,
    default: null,
  },
  /** Currently selected edge */
  selectedEdge: {
    type: Object as PropType<CircuitEdge>,
    default: null,
  },
  /** Currently hovered node */
  hoveredNode: {
    type: Object as PropType<CircuitNode>,
    default: null,
  },
  /** Currently hovered edge */
  hoveredEdge: {
    type: Object as PropType<CircuitEdge>,
    default: null,
  },
})

// Get node connections
const nodeConnections = computed(() => {
  const node = props.selectedNode || props.hoveredNode
  if (!node || !props.circuit) return null

  const incoming = props.circuit.edges.filter(e => e.target === node.id)
  const outgoing = props.circuit.edges.filter(e => e.source === node.id)

  return {
    incoming: incoming.map(e => ({
      ...e,
      node: props.circuit!.nodes.find(n => n.id === e.source),
    })),
    outgoing: outgoing.map(e => ({
      ...e,
      node: props.circuit!.nodes.find(n => n.id === e.target),
    })),
  }
})

// Get edge endpoints
const edgeEndpoints = computed(() => {
  const edge = props.selectedEdge || props.hoveredEdge
  if (!edge || !props.circuit) return null

  return {
    source: props.circuit.nodes.find(n => n.id === edge.source),
    target: props.circuit.nodes.find(n => n.id === edge.target),
  }
})

// Format node label
function formatNodeLabel(node: CircuitNode): string {
  if (node.label) return node.label
  if (node.component === 'attention' && node.head !== undefined) {
    return `Layer ${node.layer}, Head ${node.head}`
  }
  return `Layer ${node.layer} ${componentLabels[node.component]}`
}

// Get component color
function getComponentColor(type: CircuitComponentType): string {
  return componentColors[type] || '#6b7280'
}

// Format importance as percentage
function formatImportance(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}
</script>

<template>
  <div class="circuit-details space-y-4">
    <!-- Circuit overview (when nothing selected) -->
    <div v-if="!selectedNode && !selectedEdge && !hoveredNode && !hoveredEdge && circuit">
      <div class="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
        {{ circuit.name }}
      </div>
      <p v-if="circuit.description" class="text-sm text-gray-600 dark:text-gray-400 mb-3">
        {{ circuit.description }}
      </p>

      <div class="grid grid-cols-2 gap-2">
        <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
          <div class="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {{ circuit.nodes.length }}
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">Nodes</div>
        </div>
        <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
          <div class="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {{ circuit.edges.length }}
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">Edges</div>
        </div>
        <div v-if="circuit.avgImportance" class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
          <div class="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {{ formatImportance(circuit.avgImportance) }}
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">Avg Importance</div>
        </div>
        <div v-if="circuit.behavior" class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50 col-span-2">
          <div class="text-sm text-gray-900 dark:text-gray-100">
            {{ circuit.behavior }}
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">Behavior</div>
        </div>
      </div>

      <p class="mt-4 text-xs text-gray-400 dark:text-gray-500">
        Click on a node or edge to see details
      </p>
    </div>

    <!-- Selected/Hovered Node -->
    <div v-else-if="selectedNode || hoveredNode">
      <div class="flex items-center gap-2 mb-3">
        <div
          class="w-4 h-4 rounded"
          :style="{ backgroundColor: getComponentColor((selectedNode || hoveredNode)!.component) }"
        />
        <span class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ formatNodeLabel((selectedNode || hoveredNode)!) }}
        </span>
        <span
          v-if="hoveredNode && !selectedNode"
          class="text-xs px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-500"
        >
          hover
        </span>
      </div>

      <!-- Node properties -->
      <div class="space-y-2 mb-4">
        <div class="flex justify-between text-sm">
          <span class="text-gray-500 dark:text-gray-400">Type</span>
          <span class="text-gray-900 dark:text-gray-100">
            {{ componentLabels[(selectedNode || hoveredNode)!.component] }}
          </span>
        </div>
        <div class="flex justify-between text-sm">
          <span class="text-gray-500 dark:text-gray-400">Layer</span>
          <span class="text-gray-900 dark:text-gray-100">
            {{ (selectedNode || hoveredNode)!.layer }}
          </span>
        </div>
        <div v-if="(selectedNode || hoveredNode)!.head !== undefined" class="flex justify-between text-sm">
          <span class="text-gray-500 dark:text-gray-400">Head</span>
          <span class="text-gray-900 dark:text-gray-100">
            {{ (selectedNode || hoveredNode)!.head }}
          </span>
        </div>
        <div v-if="(selectedNode || hoveredNode)!.importance" class="flex justify-between text-sm">
          <span class="text-gray-500 dark:text-gray-400">Importance</span>
          <span class="text-gray-900 dark:text-gray-100">
            {{ formatImportance((selectedNode || hoveredNode)!.importance!) }}
          </span>
        </div>
        <div class="flex justify-between text-sm">
          <span class="text-gray-500 dark:text-gray-400">ID</span>
          <span class="text-gray-900 dark:text-gray-100 font-mono text-xs">
            {{ (selectedNode || hoveredNode)!.id }}
          </span>
        </div>
      </div>

      <!-- Connections -->
      <div v-if="nodeConnections" class="space-y-3">
        <!-- Incoming edges -->
        <div v-if="nodeConnections.incoming.length > 0">
          <div class="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Incoming ({{ nodeConnections.incoming.length }})
          </div>
          <div class="space-y-1 max-h-32 overflow-y-auto">
            <div
              v-for="conn in nodeConnections.incoming"
              :key="conn.source"
              class="flex items-center justify-between text-xs p-1.5 rounded bg-gray-50 dark:bg-gray-800/50"
            >
              <div class="flex items-center gap-1">
                <div
                  class="w-2 h-2 rounded"
                  :style="{ backgroundColor: conn.node ? getComponentColor(conn.node.component) : '#6b7280' }"
                />
                <span class="text-gray-700 dark:text-gray-300">
                  {{ conn.node ? formatNodeLabel(conn.node) : conn.source }}
                </span>
              </div>
              <span class="text-gray-500 font-mono">
                {{ formatImportance(conn.importance) }}
              </span>
            </div>
          </div>
        </div>

        <!-- Outgoing edges -->
        <div v-if="nodeConnections.outgoing.length > 0">
          <div class="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Outgoing ({{ nodeConnections.outgoing.length }})
          </div>
          <div class="space-y-1 max-h-32 overflow-y-auto">
            <div
              v-for="conn in nodeConnections.outgoing"
              :key="conn.target"
              class="flex items-center justify-between text-xs p-1.5 rounded bg-gray-50 dark:bg-gray-800/50"
            >
              <div class="flex items-center gap-1">
                <div
                  class="w-2 h-2 rounded"
                  :style="{ backgroundColor: conn.node ? getComponentColor(conn.node.component) : '#6b7280' }"
                />
                <span class="text-gray-700 dark:text-gray-300">
                  {{ conn.node ? formatNodeLabel(conn.node) : conn.target }}
                </span>
              </div>
              <span class="text-gray-500 font-mono">
                {{ formatImportance(conn.importance) }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Selected/Hovered Edge -->
    <div v-else-if="selectedEdge || hoveredEdge">
      <div class="flex items-center gap-2 mb-3">
        <svg class="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
        </svg>
        <span class="text-sm font-medium text-gray-900 dark:text-gray-100">
          Edge Connection
        </span>
        <span
          v-if="hoveredEdge && !selectedEdge"
          class="text-xs px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-500"
        >
          hover
        </span>
      </div>

      <!-- Edge endpoints -->
      <div v-if="edgeEndpoints" class="space-y-2 mb-4">
        <!-- Source -->
        <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
          <div class="text-xs text-gray-500 dark:text-gray-400 mb-1">Source</div>
          <div class="flex items-center gap-2">
            <div
              class="w-3 h-3 rounded"
              :style="{ backgroundColor: edgeEndpoints.source ? getComponentColor(edgeEndpoints.source.component) : '#6b7280' }"
            />
            <span class="text-sm text-gray-900 dark:text-gray-100">
              {{ edgeEndpoints.source ? formatNodeLabel(edgeEndpoints.source) : (selectedEdge || hoveredEdge)!.source }}
            </span>
          </div>
        </div>

        <!-- Arrow -->
        <div class="flex justify-center">
          <svg class="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </div>

        <!-- Target -->
        <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
          <div class="text-xs text-gray-500 dark:text-gray-400 mb-1">Target</div>
          <div class="flex items-center gap-2">
            <div
              class="w-3 h-3 rounded"
              :style="{ backgroundColor: edgeEndpoints.target ? getComponentColor(edgeEndpoints.target.component) : '#6b7280' }"
            />
            <span class="text-sm text-gray-900 dark:text-gray-100">
              {{ edgeEndpoints.target ? formatNodeLabel(edgeEndpoints.target) : (selectedEdge || hoveredEdge)!.target }}
            </span>
          </div>
        </div>
      </div>

      <!-- Edge properties -->
      <div class="space-y-2">
        <div class="flex justify-between text-sm">
          <span class="text-gray-500 dark:text-gray-400">Importance</span>
          <span class="text-gray-900 dark:text-gray-100 font-semibold">
            {{ formatImportance((selectedEdge || hoveredEdge)!.importance) }}
          </span>
        </div>
        <div v-if="(selectedEdge || hoveredEdge)!.type" class="flex justify-between text-sm">
          <span class="text-gray-500 dark:text-gray-400">Type</span>
          <span class="text-gray-900 dark:text-gray-100">
            {{ (selectedEdge || hoveredEdge)!.type }}
          </span>
        </div>

        <!-- Importance bar -->
        <div class="mt-2">
          <div class="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              class="h-full bg-blue-500 rounded-full transition-all"
              :style="{ width: `${(selectedEdge || hoveredEdge)!.importance * 100}%` }"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Empty state -->
    <div v-else class="text-sm text-gray-400 dark:text-gray-500 text-center py-4">
      No circuit loaded
    </div>
  </div>
</template>
