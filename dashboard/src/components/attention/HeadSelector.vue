<script setup lang="ts">
/**
 * HeadSelector Component
 *
 * Grid for selecting attention heads across layers.
 */

import { computed, type PropType } from 'vue'
import type { HeadClassification } from '@/types'

interface HeadInfo {
  layer: number
  head: number
  classification?: HeadClassification
  entropy?: number
}

const props = defineProps({
  /** Number of layers */
  numLayers: {
    type: Number,
    required: true,
  },
  /** Number of heads per layer */
  numHeads: {
    type: Number,
    required: true,
  },
  /** Currently selected layer */
  selectedLayer: {
    type: Number,
    default: null,
  },
  /** Currently selected head */
  selectedHead: {
    type: Number,
    default: null,
  },
  /** Head analysis data for coloring */
  headAnalyses: {
    type: Array as PropType<HeadInfo[]>,
    default: () => [],
  },
  /** Show layer labels */
  showLayerLabels: {
    type: Boolean,
    default: true,
  },
  /** Show head labels */
  showHeadLabels: {
    type: Boolean,
    default: true,
  },
  /** Compact mode (smaller cells) */
  compact: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits<{
  (e: 'select', layer: number, head: number): void
  (e: 'update:selectedLayer', layer: number): void
  (e: 'update:selectedHead', head: number): void
}>()

// Classification colors
const classificationColors: Record<HeadClassification, string> = {
  previous_token: '#3b82f6', // blue
  bos: '#8b5cf6', // violet
  uniform: '#6b7280', // gray
  induction: '#10b981', // emerald
  name_mover: '#f59e0b', // amber
  s_inhibition: '#ef4444', // red
  backup_name_mover: '#f97316', // orange
  duplicate_token: '#06b6d4', // cyan
  mixed: '#a855f7', // purple
  other: '#9ca3af', // gray-400
}

// Create layer array
const layers = computed(() => Array.from({ length: props.numLayers }, (_, i) => i))
const heads = computed(() => Array.from({ length: props.numHeads }, (_, i) => i))

// Get head info for a specific layer/head
function getHeadInfo(layer: number, head: number): HeadInfo | undefined {
  return props.headAnalyses.find((h) => h.layer === layer && h.head === head)
}

// Get cell background color
function getCellColor(layer: number, head: number): string {
  const info = getHeadInfo(layer, head)
  if (info?.classification) {
    return classificationColors[info.classification]
  }
  return '#e5e7eb' // gray-200
}

// Check if cell is selected
function isSelected(layer: number, head: number): boolean {
  return props.selectedLayer === layer && props.selectedHead === head
}

// Handle cell click
function handleClick(layer: number, head: number): void {
  emit('select', layer, head)
  emit('update:selectedLayer', layer)
  emit('update:selectedHead', head)
}

// Cell size based on compact mode
const cellSize = computed(() => (props.compact ? 'w-4 h-4' : 'w-6 h-6'))
</script>

<template>
  <div class="head-selector">
    <!-- Header labels -->
    <div v-if="showHeadLabels" class="flex mb-1" :class="showLayerLabels ? 'ml-8' : ''">
      <div
        v-for="head in heads"
        :key="`header-${head}`"
        :class="[cellSize, 'flex items-center justify-center text-[10px] text-gray-500']"
      >
        {{ head }}
      </div>
    </div>

    <!-- Grid rows -->
    <div
      v-for="layer in layers"
      :key="`layer-${layer}`"
      class="flex items-center"
    >
      <!-- Layer label -->
      <div
        v-if="showLayerLabels"
        class="w-8 text-right pr-2 text-[10px] text-gray-500 font-medium"
      >
        L{{ layer }}
      </div>

      <!-- Head cells -->
      <div
        v-for="head in heads"
        :key="`cell-${layer}-${head}`"
        :class="[
          cellSize,
          'rounded-sm cursor-pointer transition-all',
          'hover:ring-2 hover:ring-offset-1 hover:ring-blue-400',
          isSelected(layer, head) ? 'ring-2 ring-offset-1 ring-blue-600' : '',
        ]"
        :style="{ backgroundColor: getCellColor(layer, head) }"
        :title="`Layer ${layer}, Head ${head}${getHeadInfo(layer, head)?.classification ? ` (${getHeadInfo(layer, head)?.classification})` : ''}`"
        @click="handleClick(layer, head)"
      />
    </div>

    <!-- Legend -->
    <div class="mt-3 flex flex-wrap gap-2 text-[10px]">
      <div
        v-for="(color, classification) in classificationColors"
        :key="classification"
        class="flex items-center gap-1"
      >
        <div class="w-3 h-3 rounded-sm" :style="{ backgroundColor: color }" />
        <span class="text-gray-600 dark:text-gray-400">{{ classification.replace(/_/g, ' ') }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.head-selector {
  user-select: none;
}
</style>
