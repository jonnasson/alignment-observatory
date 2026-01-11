<script setup lang="ts">
/**
 * FeatureCoactivation Component
 *
 * Visualizes co-activation relationships between SAE features.
 */

import { computed, type PropType } from 'vue'
import type { FeatureCoactivation as CoactivationType } from '@/types'
import { formatFeatureIdx } from '@/composables/useSAE'

const props = defineProps({
  /** Co-activation pairs */
  coactivations: {
    type: Array as PropType<CoactivationType[]>,
    required: true,
  },
  /** Maximum number of pairs to show */
  maxPairs: {
    type: Number,
    default: 20,
  },
  /** Highlighted feature (show only its pairs) */
  highlightedFeature: {
    type: Number,
    default: null,
  },
})

const emit = defineEmits<{
  (e: 'pairSelect', featureA: number, featureB: number): void
  (e: 'pairHover', featureA: number | null, featureB: number | null): void
}>()

// Filtered and sorted coactivations
const displayedPairs = computed(() => {
  let pairs = [...props.coactivations]

  if (props.highlightedFeature !== null) {
    pairs = pairs.filter(
      (p) =>
        p.featureA === props.highlightedFeature ||
        p.featureB === props.highlightedFeature
    )
  }

  return pairs.slice(0, props.maxPairs)
})

// Get bar width for score
function getBarWidth(score: number): string {
  return `${Math.min(100, score * 100)}%`
}

// Get color based on score
function getScoreColor(score: number): string {
  if (score > 0.5) return 'bg-red-500'
  if (score > 0.3) return 'bg-orange-500'
  if (score > 0.1) return 'bg-yellow-500'
  return 'bg-blue-500'
}
</script>

<template>
  <div class="feature-coactivation">
    <div v-if="coactivations.length === 0" class="text-sm text-gray-500">
      No co-activation data available
    </div>

    <div v-else class="space-y-2">
      <div
        v-for="pair in displayedPairs"
        :key="`${pair.featureA}-${pair.featureB}`"
        class="flex items-center gap-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 -mx-1 px-1 py-0.5 rounded"
        @click="emit('pairSelect', pair.featureA, pair.featureB)"
        @mouseenter="emit('pairHover', pair.featureA, pair.featureB)"
        @mouseleave="emit('pairHover', null, null)"
      >
        <!-- Feature pair labels -->
        <div class="flex items-center gap-1 w-32 shrink-0">
          <span
            class="text-xs font-mono px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-800"
            :class="{ 'bg-blue-100 dark:bg-blue-900': pair.featureA === highlightedFeature }"
          >
            {{ formatFeatureIdx(pair.featureA) }}
          </span>
          <span class="text-gray-400">↔</span>
          <span
            class="text-xs font-mono px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-800"
            :class="{ 'bg-blue-100 dark:bg-blue-900': pair.featureB === highlightedFeature }"
          >
            {{ formatFeatureIdx(pair.featureB) }}
          </span>
        </div>

        <!-- Score bar -->
        <div class="flex-1 h-3 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
          <div
            class="h-full rounded transition-all duration-200"
            :class="getScoreColor(pair.score)"
            :style="{ width: getBarWidth(pair.score) }"
          />
        </div>

        <!-- Score value -->
        <span class="w-12 text-xs text-right text-gray-600 dark:text-gray-400">
          {{ (pair.score * 100).toFixed(0) }}%
        </span>
      </div>
    </div>

    <!-- Explanation -->
    <div class="mt-3 text-xs text-gray-500">
      Co-activation score: fraction of positions where both features are active together
    </div>
  </div>
</template>
