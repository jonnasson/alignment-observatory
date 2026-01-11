<script setup lang="ts">
/**
 * FeatureActivationGrid Component
 *
 * Grid visualization showing top-K SAE features per token position.
 */

import { ref, computed, type PropType } from 'vue'
import type { PositionFeatures, FeatureActivation } from '@/types'
import { formatFeatureIdx, formatActivation, getFeatureColor } from '@/composables/useSAE'

const props = defineProps({
  /** Top-K features per position */
  features: {
    type: Array as PropType<PositionFeatures[]>,
    required: true,
  },
  /** Number of top features to show */
  topK: {
    type: Number,
    default: 10,
  },
  /** Maximum activation value for color scaling */
  maxActivation: {
    type: Number,
    default: 10,
  },
  /** Token strings for labels */
  tokens: {
    type: Array as PropType<string[]>,
    default: () => [],
  },
})

const emit = defineEmits<{
  (e: 'featureSelect', feature: FeatureActivation, position: number): void
  (e: 'featureHover', feature: FeatureActivation | null, position: number | null): void
}>()

// Hovered cell
const hoveredCell = ref<{ feature: number; position: number } | null>(null)

// Grid data: rows are features, columns are positions
const gridData = computed(() => {
  // Collect all unique feature indices
  const allFeatures = new Set<number>()
  for (const pos of props.features) {
    for (const feat of pos.topK.slice(0, props.topK)) {
      allFeatures.add(feat.featureIdx)
    }
  }

  // Sort by global frequency
  const featureCounts = new Map<number, number>()
  for (const pos of props.features) {
    for (const feat of pos.topK) {
      featureCounts.set(feat.featureIdx, (featureCounts.get(feat.featureIdx) ?? 0) + 1)
    }
  }

  const sortedFeatures = Array.from(allFeatures).sort(
    (a, b) => (featureCounts.get(b) ?? 0) - (featureCounts.get(a) ?? 0)
  )

  // Build grid
  const grid: { featureIdx: number; cells: (number | null)[] }[] = []

  for (const featureIdx of sortedFeatures.slice(0, 30)) {
    const cells: (number | null)[] = []
    for (const pos of props.features) {
      const feat = pos.topK.find((f) => f.featureIdx === featureIdx)
      cells.push(feat?.activation ?? null)
    }
    grid.push({ featureIdx, cells })
  }

  return grid
})

// Get cell color
function getCellColor(activation: number | null): string {
  if (activation === null) return 'transparent'
  return getFeatureColor(activation, props.maxActivation)
}

// Handle cell hover
function handleCellHover(featureIdx: number, position: number, activation: number | null): void {
  if (activation !== null) {
    hoveredCell.value = { feature: featureIdx, position }
    emit('featureHover', { featureIdx, activation }, position)
  } else {
    hoveredCell.value = null
    emit('featureHover', null, null)
  }
}

// Handle cell click
function handleCellClick(featureIdx: number, position: number, activation: number | null): void {
  if (activation !== null) {
    emit('featureSelect', { featureIdx, activation }, position)
  }
}

// Get token label
function getTokenLabel(position: number): string {
  const token = props.tokens[position] ?? `[${position}]`
  // Truncate long tokens
  if (token.length > 6) {
    return token.slice(0, 5) + '…'
  }
  return token
}
</script>

<template>
  <div class="feature-activation-grid overflow-auto">
    <div v-if="features.length === 0" class="text-center py-8 text-gray-500">
      No feature data available
    </div>

    <div v-else class="min-w-max">
      <!-- Header row with tokens -->
      <div class="flex gap-0.5 mb-1 sticky top-0 bg-white dark:bg-gray-900 z-10">
        <div class="w-20 shrink-0 px-1 text-xs font-medium text-gray-500">Feature</div>
        <div
          v-for="(pos, idx) in features"
          :key="idx"
          class="w-8 text-center text-xs text-gray-500 truncate"
          :title="tokens[pos.position] ?? `Position ${pos.position}`"
        >
          {{ getTokenLabel(pos.position) }}
        </div>
      </div>

      <!-- Grid rows -->
      <div class="space-y-0.5">
        <div
          v-for="row in gridData"
          :key="row.featureIdx"
          class="flex gap-0.5 items-center"
        >
          <!-- Feature index label -->
          <div class="w-20 shrink-0 px-1 text-xs font-mono text-gray-600 dark:text-gray-400">
            {{ formatFeatureIdx(row.featureIdx) }}
          </div>

          <!-- Cells -->
          <div
            v-for="(activation, colIdx) in row.cells"
            :key="colIdx"
            class="w-8 h-6 rounded-sm cursor-pointer transition-all duration-150"
            :class="[
              activation !== null ? 'hover:ring-2 hover:ring-blue-400' : '',
              hoveredCell?.feature === row.featureIdx && hoveredCell?.position === colIdx
                ? 'ring-2 ring-blue-500'
                : '',
            ]"
            :style="{ backgroundColor: getCellColor(activation) }"
            :title="
              activation !== null
                ? `Feature ${formatFeatureIdx(row.featureIdx)}: ${formatActivation(activation)}`
                : ''
            "
            @mouseenter="handleCellHover(row.featureIdx, colIdx, activation)"
            @mouseleave="hoveredCell = null; emit('featureHover', null, null)"
            @click="handleCellClick(row.featureIdx, colIdx, activation)"
          />
        </div>
      </div>

      <!-- Color scale legend -->
      <div class="mt-4 flex items-center gap-2">
        <span class="text-xs text-gray-500">Activation:</span>
        <div class="flex">
          <div
            v-for="i in 10"
            :key="i"
            class="w-6 h-3"
            :style="{ backgroundColor: getFeatureColor((i / 10) * maxActivation, maxActivation) }"
          />
        </div>
        <span class="text-xs text-gray-500">0</span>
        <span class="text-xs text-gray-500">→</span>
        <span class="text-xs text-gray-500">{{ maxActivation }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.feature-activation-grid {
  max-height: 400px;
}
</style>
