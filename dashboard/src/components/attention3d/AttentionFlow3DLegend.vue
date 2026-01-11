<script setup lang="ts">
/**
 * AttentionFlow3DLegend - Color scale legend for 3D attention visualization
 *
 * Shows the color mapping from attention values to colors.
 */

import { computed } from 'vue'
import { useColorScale, type ColorScaleName } from '@/composables/useColorScale'

interface Props {
  /** Color scale name */
  colorScale?: ColorScaleName
  /** Minimum value label */
  minLabel?: string
  /** Maximum value label */
  maxLabel?: string
}

const props = withDefaults(defineProps<Props>(), {
  colorScale: 'viridis',
  minLabel: '0.0',
  maxLabel: '1.0',
})

const { getGradient } = useColorScale({ scaleName: props.colorScale })

const gradientStyle = computed(() => ({
  background: getGradient('horizontal'),
}))
</script>

<template>
  <div class="attention-3d-legend flex items-center gap-2 px-3 py-2 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800">
    <span class="text-xs text-gray-500 dark:text-gray-400">Attention:</span>
    <span class="text-xs font-mono text-gray-600 dark:text-gray-300">{{ minLabel }}</span>
    <div
      class="h-3 w-32 rounded"
      :style="gradientStyle"
    />
    <span class="text-xs font-mono text-gray-600 dark:text-gray-300">{{ maxLabel }}</span>

    <!-- Axis legend -->
    <div class="ml-4 flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
      <span><span class="font-medium">X:</span> Query</span>
      <span><span class="font-medium">Y:</span> Key</span>
      <span><span class="font-medium">Z:</span> Layer</span>
    </div>
  </div>
</template>
