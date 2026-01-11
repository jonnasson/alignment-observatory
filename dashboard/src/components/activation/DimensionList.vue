<script setup lang="ts">
/**
 * DimensionList Component
 *
 * Shows top dimensions by variance or other metrics.
 */

import { type PropType } from 'vue'

interface DimensionStats {
  dimIdx: number
  mean: number
  variance: number
  maxAbs: number
}

const props = defineProps({
  /** Dimension statistics */
  dimensions: {
    type: Array as PropType<DimensionStats[]>,
    required: true,
  },
  /** Title */
  title: {
    type: String,
    default: 'Top Dimensions',
  },
  /** Sort metric */
  sortBy: {
    type: String as PropType<'variance' | 'maxAbs'>,
    default: 'variance',
  },
  /** Selected dimension */
  selectedDimension: {
    type: Number,
    default: null,
  },
  /** Max items to show */
  maxItems: {
    type: Number,
    default: 10,
  },
})

const emit = defineEmits<{
  (e: 'dimensionClick', dimIdx: number): void
  (e: 'dimensionHover', dimIdx: number | null): void
}>()

// Format number
function formatNumber(val: number): string {
  if (Math.abs(val) < 0.001 && val !== 0) {
    return val.toExponential(2)
  }
  return val.toFixed(4)
}

// Get display value based on sort
function getDisplayValue(dim: DimensionStats): string {
  if (props.sortBy === 'maxAbs') {
    return formatNumber(dim.maxAbs)
  }
  return formatNumber(dim.variance)
}

// Get metric label
function getMetricLabel(): string {
  return props.sortBy === 'maxAbs' ? 'Max |val|' : 'Variance'
}

// Compute bar width percentage
function getBarWidth(dim: DimensionStats): number {
  const max = props.dimensions[0]
  if (!max) return 0
  const maxVal = props.sortBy === 'maxAbs' ? max.maxAbs : max.variance
  const val = props.sortBy === 'maxAbs' ? dim.maxAbs : dim.variance
  return (val / maxVal) * 100
}
</script>

<template>
  <div class="dimension-list">
    <!-- Header -->
    <div class="flex items-center justify-between mb-2">
      <span class="text-sm font-medium text-gray-900 dark:text-gray-100">
        {{ title }}
      </span>
      <span class="text-xs text-gray-500 dark:text-gray-400">
        {{ getMetricLabel() }}
      </span>
    </div>

    <!-- List -->
    <div class="space-y-1">
      <div
        v-for="dim in dimensions.slice(0, maxItems)"
        :key="dim.dimIdx"
        class="flex items-center gap-2 p-1.5 rounded cursor-pointer transition-colors"
        :class="[
          selectedDimension === dim.dimIdx
            ? 'bg-blue-100 dark:bg-blue-900/30'
            : 'hover:bg-gray-100 dark:hover:bg-gray-800',
        ]"
        @click="emit('dimensionClick', dim.dimIdx)"
        @mouseenter="emit('dimensionHover', dim.dimIdx)"
        @mouseleave="emit('dimensionHover', null)"
      >
        <!-- Dimension index -->
        <span class="w-12 text-xs font-mono text-gray-600 dark:text-gray-400">
          d{{ dim.dimIdx }}
        </span>

        <!-- Bar -->
        <div class="flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
          <div
            class="h-full bg-blue-500 rounded transition-all"
            :style="{ width: `${getBarWidth(dim)}%` }"
          />
        </div>

        <!-- Value -->
        <span class="w-16 text-right text-xs font-mono text-gray-700 dark:text-gray-300">
          {{ getDisplayValue(dim) }}
        </span>
      </div>
    </div>

    <!-- Show more indicator -->
    <div
      v-if="dimensions.length > maxItems"
      class="mt-2 text-xs text-center text-gray-400"
    >
      + {{ dimensions.length - maxItems }} more dimensions
    </div>
  </div>
</template>
