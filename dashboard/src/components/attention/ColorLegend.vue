<script setup lang="ts">
/**
 * ColorLegend Component
 *
 * Displays a color scale legend for heatmaps.
 */

import { computed, type PropType } from 'vue'
import { useColorScale, type ColorScaleName } from '@/composables/useColorScale'

const props = defineProps({
  /** Color scale name */
  colorScale: {
    type: String as PropType<ColorScaleName>,
    default: 'viridis',
  },
  /** Minimum value */
  min: {
    type: Number,
    default: 0,
  },
  /** Maximum value */
  max: {
    type: Number,
    default: 1,
  },
  /** Orientation */
  orientation: {
    type: String as PropType<'horizontal' | 'vertical'>,
    default: 'horizontal',
  },
  /** Width in pixels (for horizontal) */
  width: {
    type: Number,
    default: 200,
  },
  /** Height in pixels (for vertical) */
  height: {
    type: Number,
    default: 200,
  },
  /** Show tick values */
  showTicks: {
    type: Boolean,
    default: true,
  },
  /** Number of ticks to show */
  numTicks: {
    type: Number,
    default: 5,
  },
  /** Label text */
  label: {
    type: String,
    default: '',
  },
})

const colorScaleOptions = computed(() => ({
  scaleName: props.colorScale,
  min: props.min,
  max: props.max,
}))

const { getGradient } = useColorScale(colorScaleOptions)

// Compute gradient style
const gradientStyle = computed(() => {
  const gradient = getGradient(props.orientation)
  return {
    background: gradient,
    width: props.orientation === 'horizontal' ? `${props.width}px` : '16px',
    height: props.orientation === 'horizontal' ? '16px' : `${props.height}px`,
  }
})

// Generate tick values
const ticks = computed(() => {
  const tickValues: number[] = []
  const range = props.max - props.min

  for (let i = 0; i < props.numTicks; i++) {
    const value = props.min + (range * i) / (props.numTicks - 1)
    tickValues.push(value)
  }

  return tickValues
})

// Format tick value
function formatTick(value: number): string {
  if (Math.abs(value) < 0.01 && value !== 0) {
    return value.toExponential(1)
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(0)
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(2)
  }
  return value.toFixed(3)
}
</script>

<template>
  <div
    :class="[
      'color-legend flex gap-2',
      orientation === 'horizontal' ? 'flex-col' : 'flex-row',
    ]"
  >
    <!-- Label -->
    <div v-if="label" class="text-xs font-medium text-gray-600 dark:text-gray-400">
      {{ label }}
    </div>

    <!-- Legend content -->
    <div
      :class="[
        'flex gap-1',
        orientation === 'horizontal' ? 'flex-col' : 'flex-row',
      ]"
    >
      <!-- Color bar -->
      <div :style="gradientStyle" class="rounded" />

      <!-- Ticks -->
      <div
        v-if="showTicks"
        :class="[
          'flex text-[10px] text-gray-500 dark:text-gray-400',
          orientation === 'horizontal' ? 'flex-row justify-between' : 'flex-col justify-between',
        ]"
        :style="{
          width: orientation === 'horizontal' ? `${width}px` : 'auto',
          height: orientation === 'vertical' ? `${height}px` : 'auto',
        }"
      >
        <span
          v-for="(tick, idx) in ticks"
          :key="idx"
          :class="orientation === 'vertical' ? 'text-right' : ''"
        >
          {{ formatTick(tick) }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.color-legend {
  user-select: none;
}
</style>
