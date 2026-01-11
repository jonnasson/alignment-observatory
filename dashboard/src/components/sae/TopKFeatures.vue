<script setup lang="ts">
/**
 * TopKFeatures Component
 *
 * Display of top-K features with activation bars.
 */

import { computed, type PropType } from 'vue'
import type { FeatureFrequency, FeatureActivation } from '@/types'
import { formatFeatureIdx, formatActivation, getFeatureColor } from '@/composables/useSAE'

const props = defineProps({
  /** Top features (either global frequency or position-specific) */
  features: {
    type: Array as PropType<FeatureFrequency[] | FeatureActivation[]>,
    required: true,
  },
  /** Maximum value for bar scaling */
  maxValue: {
    type: Number,
    default: 1,
  },
  /** Display mode */
  mode: {
    type: String as PropType<'frequency' | 'activation'>,
    default: 'activation',
  },
  /** Title */
  title: {
    type: String,
    default: 'Top Features',
  },
  /** Highlighted feature indices */
  highlighted: {
    type: Array as PropType<number[]>,
    default: () => [],
  },
})

const emit = defineEmits<{
  (e: 'featureSelect', featureIdx: number): void
  (e: 'featureHover', featureIdx: number | null): void
}>()

// Normalize features to common format
const normalizedFeatures = computed(() => {
  return props.features.map((f) => {
    if ('frequency' in f) {
      return {
        featureIdx: f.featureIdx,
        value: props.mode === 'frequency' ? f.frequency : f.meanActivation,
        secondary: f.maxActivation,
      }
    } else {
      return {
        featureIdx: f.featureIdx,
        value: f.activation,
        secondary: null,
      }
    }
  })
})

// Get bar width percentage
function getBarWidth(value: number): string {
  const pct = Math.min(100, (value / props.maxValue) * 100)
  return `${pct}%`
}

// Check if feature is highlighted
function isHighlighted(featureIdx: number): boolean {
  return props.highlighted.includes(featureIdx)
}
</script>

<template>
  <div class="top-k-features">
    <h4 v-if="title" class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
      {{ title }}
    </h4>

    <div v-if="features.length === 0" class="text-sm text-gray-500">
      No features to display
    </div>

    <div v-else class="space-y-1.5">
      <div
        v-for="feat in normalizedFeatures"
        :key="feat.featureIdx"
        class="flex items-center gap-2 group cursor-pointer"
        :class="{ 'bg-blue-50 dark:bg-blue-900/20 -mx-1 px-1 rounded': isHighlighted(feat.featureIdx) }"
        @click="emit('featureSelect', feat.featureIdx)"
        @mouseenter="emit('featureHover', feat.featureIdx)"
        @mouseleave="emit('featureHover', null)"
      >
        <!-- Feature index -->
        <span class="w-16 text-xs font-mono text-gray-600 dark:text-gray-400 shrink-0">
          {{ formatFeatureIdx(feat.featureIdx) }}
        </span>

        <!-- Bar -->
        <div class="flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
          <div
            class="h-full rounded transition-all duration-200 group-hover:opacity-80"
            :style="{
              width: getBarWidth(feat.value),
              backgroundColor: getFeatureColor(feat.value, maxValue),
            }"
          />
        </div>

        <!-- Value -->
        <span class="w-12 text-xs text-right text-gray-600 dark:text-gray-400 shrink-0">
          {{ mode === 'frequency' ? (feat.value * 100).toFixed(0) + '%' : formatActivation(feat.value) }}
        </span>
      </div>
    </div>

    <!-- Legend for frequency mode -->
    <div v-if="mode === 'frequency'" class="mt-2 text-xs text-gray-500">
      Percentage of positions where feature is active
    </div>
  </div>
</template>
