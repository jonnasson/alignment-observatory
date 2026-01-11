<script setup lang="ts">
  import { computed } from 'vue'

  interface Props {
    min: number
    max: number
    colorScale?: 'viridis' | 'inferno' | 'coolwarm' | 'blues'
    label?: string
    orientation?: 'horizontal' | 'vertical'
    width?: number
    height?: number
  }

  const props = withDefaults(defineProps<Props>(), {
    colorScale: 'viridis',
    label: undefined,
    orientation: 'vertical',
    width: 20,
    height: 200,
  })

  // Generate gradient stops for different color scales
  const gradientStops = computed(() => {
    const stops: { offset: string; color: string }[] = []

    // Viridis-inspired colors
    const colorMaps: Record<string, string[]> = {
      viridis: ['#440154', '#482878', '#3e4989', '#31688e', '#26838f', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
      inferno: ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4'],
      coolwarm: ['#3b4cc0', '#6688ee', '#88bbff', '#b0d0ff', '#d8daeb', '#f7d9c4', '#f4a582', '#d6604d', '#b2182b'],
      blues: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
    }

    const colors = (colorMaps[props.colorScale] ?? colorMaps.viridis) as string[]
    const numColors = colors.length

    colors.forEach((color, i) => {
      stops.push({
        offset: `${(i / (numColors - 1)) * 100}%`,
        color,
      })
    })

    return stops
  })

  const formatValue = (value: number) => {
    if (Math.abs(value) >= 1000) {
      return value.toExponential(1)
    }
    if (Math.abs(value) < 0.01 && value !== 0) {
      return value.toExponential(1)
    }
    return value.toFixed(2)
  }
</script>

<template>
  <div
    :class="[
      'flex gap-2',
      orientation === 'horizontal' ? 'flex-col items-center' : 'flex-row items-stretch',
    ]"
  >
    <!-- Label -->
    <span
      v-if="label"
      class="text-xs font-medium text-gray-600 dark:text-gray-400"
      :class="orientation === 'horizontal' ? 'mb-1' : 'writing-mode-vertical rotate-180'"
    >
      {{ label }}
    </span>

    <!-- Color bar -->
    <svg
      :width="orientation === 'horizontal' ? height : width"
      :height="orientation === 'horizontal' ? width : height"
      class="rounded"
    >
      <defs>
        <linearGradient
          :id="`gradient-${colorScale}`"
          :x1="orientation === 'horizontal' ? '0%' : '0%'"
          :y1="orientation === 'horizontal' ? '0%' : '100%'"
          :x2="orientation === 'horizontal' ? '100%' : '0%'"
          :y2="orientation === 'horizontal' ? '0%' : '0%'"
        >
          <stop
            v-for="(stop, i) in gradientStops"
            :key="i"
            :offset="stop.offset"
            :stop-color="stop.color"
          />
        </linearGradient>
      </defs>
      <rect
        x="0"
        y="0"
        :width="orientation === 'horizontal' ? height : width"
        :height="orientation === 'horizontal' ? width : height"
        :fill="`url(#gradient-${colorScale})`"
      />
    </svg>

    <!-- Value labels -->
    <div
      :class="[
        'flex text-xs text-gray-500 dark:text-gray-400',
        orientation === 'horizontal' ? 'flex-row justify-between w-full' : 'flex-col justify-between',
      ]"
      :style="orientation === 'vertical' ? { height: `${height}px` } : {}"
    >
      <span>{{ formatValue(orientation === 'vertical' ? max : min) }}</span>
      <span>{{ formatValue(orientation === 'vertical' ? min : max) }}</span>
    </div>
  </div>
</template>

<style scoped>
  .writing-mode-vertical {
    writing-mode: vertical-rl;
  }
</style>
