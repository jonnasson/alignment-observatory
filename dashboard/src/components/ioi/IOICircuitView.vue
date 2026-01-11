<script setup lang="ts">
/**
 * IOICircuitView Component
 *
 * Visualization of the IOI circuit showing heads across layers.
 */

import { ref, computed, type PropType } from 'vue'
import type { IOICircuit, IOIHead, IOIComponentType } from '@/types'
import { ioiComponentColors, ioiComponentLabels } from '@/composables/useIOI'

const props = defineProps({
  /** IOI circuit data */
  circuit: {
    type: Object as PropType<IOICircuit>,
    default: null,
  },
  /** Number of layers in model */
  numLayers: {
    type: Number,
    default: 12,
  },
  /** Number of heads per layer */
  numHeads: {
    type: Number,
    default: 12,
  },
  /** Show all heads or just detected ones */
  showAllHeads: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits<{
  (e: 'headSelect', head: IOIHead | null, layer: number, headIdx: number): void
  (e: 'headHover', head: IOIHead | null, layer: number, headIdx: number): void
}>()

// Hovered cell
const hoveredCell = ref<{ layer: number; head: number } | null>(null)

// All heads flattened with their types
const headMap = computed(() => {
  const map = new Map<string, IOIHead>()
  if (!props.circuit) return map

  const addHeads = (heads: IOIHead[]) => {
    for (const head of heads) {
      map.set(`${head.layer}-${head.head}`, head)
    }
  }

  addHeads(props.circuit.nameMoverHeads)
  addHeads(props.circuit.sInhibitionHeads)
  addHeads(props.circuit.duplicateTokenHeads)
  addHeads(props.circuit.previousTokenHeads)
  addHeads(props.circuit.backupNameMoverHeads)

  return map
})

// Get head at position
function getHead(layer: number, headIdx: number): IOIHead | null {
  return headMap.value.get(`${layer}-${headIdx}`) ?? null
}

// Get color for cell
function getCellColor(layer: number, headIdx: number): string {
  const head = getHead(layer, headIdx)
  if (!head) return 'transparent'
  return ioiComponentColors[head.componentType]
}

// Get opacity based on score
function getCellOpacity(layer: number, headIdx: number): number {
  const head = getHead(layer, headIdx)
  if (!head) return 0
  return 0.3 + head.score * 0.7
}

// Handle cell interaction
function handleCellHover(layer: number, headIdx: number): void {
  hoveredCell.value = { layer, head: headIdx }
  const head = getHead(layer, headIdx)
  emit('headHover', head, layer, headIdx)
}

function handleCellClick(layer: number, headIdx: number): void {
  const head = getHead(layer, headIdx)
  emit('headSelect', head, layer, headIdx)
}

// Legend items
const legendItems = computed(() => {
  const types: IOIComponentType[] = [
    'name_mover',
    's_inhibition',
    'duplicate_token',
    'previous_token',
    'backup_name_mover',
  ]
  return types.map((type) => ({
    type,
    label: ioiComponentLabels[type],
    color: ioiComponentColors[type],
  }))
})

// Layer labels (reversed so layer 0 is at bottom)
const layers = computed(() => {
  return Array.from({ length: props.numLayers }, (_, i) => props.numLayers - 1 - i)
})
</script>

<template>
  <div class="ioi-circuit-view">
    <div v-if="!circuit" class="h-full flex items-center justify-center text-gray-400">
      <div class="text-center">
        <svg class="w-12 h-12 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
        </svg>
        <p class="text-sm font-medium">No IOI circuit detected</p>
        <p class="text-xs mt-1">Run detection to visualize heads</p>
      </div>
    </div>

    <div v-else class="space-y-4">
      <!-- Grid visualization -->
      <div class="overflow-auto">
        <div class="min-w-max">
          <!-- Header row with head indices -->
          <div class="flex gap-0.5 mb-1">
            <div class="w-12 shrink-0" />
            <div
              v-for="h in numHeads"
              :key="h"
              class="w-7 text-center text-xs text-gray-500"
            >
              H{{ h - 1 }}
            </div>
          </div>

          <!-- Grid rows (layers) -->
          <div class="space-y-0.5">
            <div
              v-for="layer in layers"
              :key="layer"
              class="flex gap-0.5 items-center"
            >
              <!-- Layer label -->
              <div class="w-12 text-xs text-gray-500 text-right pr-2 shrink-0">
                L{{ layer }}
              </div>

              <!-- Cells -->
              <div
                v-for="h in numHeads"
                :key="h"
                class="w-7 h-7 rounded cursor-pointer transition-all duration-150 border"
                :class="[
                  getHead(layer, h - 1)
                    ? 'border-transparent'
                    : 'border-gray-200 dark:border-gray-700',
                  hoveredCell?.layer === layer && hoveredCell?.head === h - 1
                    ? 'ring-2 ring-blue-500'
                    : '',
                ]"
                :style="{
                  backgroundColor: getCellColor(layer, h - 1),
                  opacity: getHead(layer, h - 1) ? getCellOpacity(layer, h - 1) : 1,
                }"
                :title="getHead(layer, h - 1)
                  ? `L${layer}H${h - 1}: ${ioiComponentLabels[getHead(layer, h - 1)!.componentType]} (${(getHead(layer, h - 1)!.score * 100).toFixed(0)}%)`
                  : `L${layer}H${h - 1}`"
                @mouseenter="handleCellHover(layer, h - 1)"
                @mouseleave="hoveredCell = null; emit('headHover', null, layer, h - 1)"
                @click="handleCellClick(layer, h - 1)"
              />
            </div>
          </div>
        </div>
      </div>

      <!-- Legend -->
      <div class="flex flex-wrap gap-3 pt-2 border-t border-gray-200 dark:border-gray-700">
        <div
          v-for="item in legendItems"
          :key="item.type"
          class="flex items-center gap-1.5"
        >
          <div
            class="w-3 h-3 rounded"
            :style="{ backgroundColor: item.color }"
          />
          <span class="text-xs text-gray-600 dark:text-gray-400">
            {{ item.label }}
          </span>
        </div>
      </div>

      <!-- Validity score -->
      <div class="flex items-center gap-2">
        <span class="text-xs text-gray-500">Circuit Validity:</span>
        <div class="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden max-w-xs">
          <div
            class="h-full rounded-full transition-all duration-300"
            :class="[
              circuit.validityScore > 0.7 ? 'bg-green-500' :
              circuit.validityScore > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
            ]"
            :style="{ width: `${circuit.validityScore * 100}%` }"
          />
        </div>
        <span class="text-xs font-medium">
          {{ (circuit.validityScore * 100).toFixed(0) }}%
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ioi-circuit-view {
  min-height: 300px;
}
</style>
