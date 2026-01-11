<script setup lang="ts">
/**
 * CircuitControls Component
 *
 * Controls for circuit discovery and visualization options.
 */

import { type PropType } from 'vue'
import { BaseButton, BaseSelect } from '@/components/common'

const props = defineProps({
  /** Current threshold value */
  threshold: {
    type: Number,
    default: 0.5,
  },
  /** Current layout */
  layout: {
    type: String as PropType<'dagre' | 'breadthfirst' | 'cose'>,
    default: 'dagre',
  },
  /** Show labels toggle */
  showLabels: {
    type: Boolean,
    default: true,
  },
  /** Whether discovery is in progress */
  isLoading: {
    type: Boolean,
    default: false,
  },
  /** Whether a circuit is loaded */
  hasCircuit: {
    type: Boolean,
    default: false,
  },
  /** Edge count for display */
  edgeCount: {
    type: Number,
    default: 0,
  },
  /** Total edges before threshold */
  totalEdges: {
    type: Number,
    default: 0,
  },
})

const emit = defineEmits<{
  (e: 'update:threshold', value: number): void
  (e: 'update:layout', value: 'dagre' | 'breadthfirst' | 'cose'): void
  (e: 'update:showLabels', value: boolean): void
  (e: 'discover'): void
  (e: 'exportDot'): void
  (e: 'exportPng'): void
  (e: 'clear'): void
}>()

// Layout options
const layoutOptions = [
  { value: 'dagre', label: 'Hierarchical (Dagre)' },
  { value: 'breadthfirst', label: 'Breadth-first' },
  { value: 'cose', label: 'Force-directed (CoSE)' },
]

// Handle threshold change
function handleThresholdChange(event: Event): void {
  const target = event.target as HTMLInputElement
  emit('update:threshold', parseFloat(target.value))
}

// Handle layout change
function handleLayoutChange(value: string | number | null): void {
  if (typeof value === 'string') {
    emit('update:layout', value as 'dagre' | 'breadthfirst' | 'cose')
  }
}

// Handle labels toggle
function handleLabelsToggle(): void {
  emit('update:showLabels', !props.showLabels)
}
</script>

<template>
  <div class="circuit-controls space-y-4">
    <!-- Threshold slider -->
    <div>
      <div class="flex items-center justify-between mb-1">
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Edge Threshold
        </label>
        <span class="text-sm font-mono text-gray-500 dark:text-gray-400">
          {{ threshold.toFixed(2) }}
        </span>
      </div>
      <input
        :value="threshold"
        type="range"
        min="0"
        max="1"
        step="0.01"
        class="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        @input="handleThresholdChange"
      />
      <div v-if="totalEdges > 0" class="mt-1 text-xs text-gray-500 dark:text-gray-400">
        Showing {{ edgeCount }} of {{ totalEdges }} edges
      </div>
    </div>

    <!-- Layout selector -->
    <div class="w-full">
      <BaseSelect
        :model-value="layout"
        :options="layoutOptions"
        label="Layout Algorithm"
        @update:model-value="handleLayoutChange"
      />
    </div>

    <!-- Show labels toggle -->
    <div class="flex items-center gap-2">
      <button
        :class="[
          'relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
          showLabels ? 'bg-blue-500' : 'bg-gray-200 dark:bg-gray-700',
        ]"
        role="switch"
        :aria-checked="showLabels"
        @click="handleLabelsToggle"
      >
        <span
          :class="[
            'pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
            showLabels ? 'translate-x-4' : 'translate-x-0',
          ]"
        />
      </button>
      <span class="text-sm text-gray-700 dark:text-gray-300">
        Show node labels
      </span>
    </div>

    <!-- Action buttons -->
    <div class="flex flex-wrap gap-2 pt-2">
      <BaseButton
        variant="primary"
        :loading="isLoading"
        @click="emit('discover')"
      >
        <svg class="w-4 h-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        Discover Circuit
      </BaseButton>

      <BaseButton
        v-if="hasCircuit"
        variant="secondary"
        @click="emit('exportDot')"
      >
        <svg class="w-4 h-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
        Export DOT
      </BaseButton>

      <BaseButton
        v-if="hasCircuit"
        variant="secondary"
        @click="emit('exportPng')"
      >
        <svg class="w-4 h-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        Export PNG
      </BaseButton>

      <BaseButton
        v-if="hasCircuit"
        variant="ghost"
        @click="emit('clear')"
      >
        <svg class="w-4 h-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
        Clear
      </BaseButton>
    </div>
  </div>
</template>
