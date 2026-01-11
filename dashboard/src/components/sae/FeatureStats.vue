<script setup lang="ts">
/**
 * FeatureStats Component
 *
 * Displays SAE feature statistics and metrics.
 */

import { computed, type PropType } from 'vue'
import type { SAEConfig, SAEFeatures } from '@/types'
import type { SAEStats } from '@/composables/useSAE'

const props = defineProps({
  /** SAE configuration */
  config: {
    type: Object as PropType<SAEConfig>,
    default: null,
  },
  /** Computed statistics */
  stats: {
    type: Object as PropType<SAEStats>,
    default: null,
  },
  /** SAE features data */
  features: {
    type: Object as PropType<SAEFeatures>,
    default: null,
  },
})

// Format large numbers
function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return n.toLocaleString()
}

// Format percentage
function formatPct(n: number): string {
  return (n * 100).toFixed(2) + '%'
}

// Stats items to display
const statsItems = computed(() => {
  const items: { label: string; value: string; description?: string }[] = []

  if (props.config) {
    items.push(
      { label: 'Input Dim', value: formatNumber(props.config.dIn), description: 'Model hidden size' },
      { label: 'SAE Dim', value: formatNumber(props.config.dSae), description: 'Feature dimension' },
      { label: 'Expansion', value: (props.config.dSae / props.config.dIn).toFixed(0) + 'x' },
      { label: 'Activation', value: props.config.activation.toUpperCase() },
    )

    if (props.config.k) {
      items.push({ label: 'Top-K', value: props.config.k.toString() })
    }

    items.push({ label: 'Hook Point', value: props.config.hookPoint })
  }

  if (props.stats) {
    items.push(
      { label: 'Active Features', value: formatNumber(props.stats.activeFeatures) },
      { label: 'Sparsity', value: formatPct(props.stats.sparsity), description: 'Fraction of zeros' },
      { label: 'Mean Active/Pos', value: props.stats.meanActivePerPosition.toFixed(1) },
      { label: 'Max Activation', value: props.stats.maxActivation.toFixed(2) },
    )
  }

  if (props.features && !props.stats) {
    items.push(
      { label: 'Sparsity', value: formatPct(props.features.sparsity) },
      { label: 'Mean Active', value: props.features.meanActiveFeatures.toFixed(1) },
    )
  }

  return items
})
</script>

<template>
  <div class="feature-stats">
    <div v-if="!config && !stats && !features" class="text-sm text-gray-500">
      No SAE loaded
    </div>

    <div v-else class="grid grid-cols-2 gap-3">
      <div
        v-for="item in statsItems"
        :key="item.label"
        class="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-2"
      >
        <div class="text-xs text-gray-500 dark:text-gray-400">
          {{ item.label }}
        </div>
        <div
          class="text-sm font-medium text-gray-900 dark:text-gray-100 mt-0.5"
          :title="item.description"
        >
          {{ item.value }}
        </div>
      </div>
    </div>

    <!-- Sparsity visualization -->
    <div v-if="stats || features" class="mt-4">
      <div class="text-xs text-gray-500 mb-1">Sparsity Distribution</div>
      <div class="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          class="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300"
          :style="{ width: formatPct(stats?.sparsity ?? features?.sparsity ?? 0) }"
        />
      </div>
      <div class="flex justify-between text-xs text-gray-400 mt-0.5">
        <span>0% (dense)</span>
        <span>100% (sparse)</span>
      </div>
    </div>
  </div>
</template>
