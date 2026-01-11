<script setup lang="ts">
/**
 * IOIValidation Component
 *
 * Displays validation results comparing detected heads to known IOI heads.
 */

import { type PropType } from 'vue'
import type { IOIValidationResult, IOIComponentType } from '@/types'
import { ioiComponentColors, ioiComponentLabels } from '@/composables/useIOI'

const props = defineProps({
  /** Validation results */
  result: {
    type: Object as PropType<IOIValidationResult>,
    default: null,
  },
  /** Model type for reference */
  modelType: {
    type: String,
    default: 'GPT-2 Small',
  },
})

// Component types for iteration
const componentTypes: IOIComponentType[] = [
  'name_mover',
  's_inhibition',
  'duplicate_token',
  'previous_token',
  'backup_name_mover',
]

// Format percentage
function formatPct(n: number): string {
  return (n * 100).toFixed(0) + '%'
}

// Get color class for score
function getScoreColor(score: number): string {
  if (score >= 0.8) return 'text-green-600 dark:text-green-400'
  if (score >= 0.5) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

// Get bar color class
function getBarColor(score: number): string {
  if (score >= 0.8) return 'bg-green-500'
  if (score >= 0.5) return 'bg-yellow-500'
  return 'bg-red-500'
}

// Format head tuple
function formatHead(head: [number, number]): string {
  return `L${head[0]}H${head[1]}`
}
</script>

<template>
  <div class="ioi-validation">
    <div v-if="!result" class="text-sm text-gray-500">
      Run validation to compare against known {{ modelType }} IOI heads
    </div>

    <div v-else class="space-y-4">
      <!-- Overall metrics -->
      <div class="grid grid-cols-3 gap-3">
        <div class="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 text-center">
          <div class="text-2xl font-bold" :class="getScoreColor(result.precision)">
            {{ formatPct(result.precision) }}
          </div>
          <div class="text-xs text-gray-500">Precision</div>
        </div>
        <div class="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 text-center">
          <div class="text-2xl font-bold" :class="getScoreColor(result.recall)">
            {{ formatPct(result.recall) }}
          </div>
          <div class="text-xs text-gray-500">Recall</div>
        </div>
        <div class="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 text-center">
          <div class="text-2xl font-bold" :class="getScoreColor(result.f1Score)">
            {{ formatPct(result.f1Score) }}
          </div>
          <div class="text-xs text-gray-500">F1 Score</div>
        </div>
      </div>

      <!-- Per-component breakdown -->
      <div class="space-y-2">
        <h4 class="text-sm font-medium text-gray-700 dark:text-gray-300">
          Per-Component Metrics
        </h4>

        <div class="space-y-2">
          <div
            v-for="type in componentTypes"
            :key="type"
            class="flex items-center gap-2"
          >
            <!-- Component label -->
            <div class="w-28 flex items-center gap-1.5 shrink-0">
              <div
                class="w-2 h-2 rounded"
                :style="{ backgroundColor: ioiComponentColors[type] }"
              />
              <span class="text-xs text-gray-600 dark:text-gray-400 truncate">
                {{ ioiComponentLabels[type] }}
              </span>
            </div>

            <!-- F1 bar -->
            <div class="flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
              <div
                class="h-full rounded transition-all duration-300"
                :class="getBarColor(result.perComponentMetrics[type][2])"
                :style="{ width: `${result.perComponentMetrics[type][2] * 100}%` }"
              />
            </div>

            <!-- Metrics -->
            <div class="w-32 flex gap-2 text-xs shrink-0">
              <span class="text-gray-500">
                P: {{ formatPct(result.perComponentMetrics[type][0]) }}
              </span>
              <span class="text-gray-500">
                R: {{ formatPct(result.perComponentMetrics[type][1]) }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- False positives/negatives -->
      <div class="grid grid-cols-2 gap-4">
        <!-- False positives -->
        <div>
          <h4 class="text-sm font-medium text-red-600 dark:text-red-400 mb-1">
            False Positives ({{ result.falsePositives.length }})
          </h4>
          <div v-if="result.falsePositives.length === 0" class="text-xs text-gray-500">
            None
          </div>
          <div v-else class="flex flex-wrap gap-1">
            <span
              v-for="head in result.falsePositives.slice(0, 10)"
              :key="`fp-${head[0]}-${head[1]}`"
              class="text-xs font-mono px-1.5 py-0.5 rounded bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
            >
              {{ formatHead(head) }}
            </span>
            <span v-if="result.falsePositives.length > 10" class="text-xs text-gray-500">
              +{{ result.falsePositives.length - 10 }} more
            </span>
          </div>
        </div>

        <!-- False negatives -->
        <div>
          <h4 class="text-sm font-medium text-amber-600 dark:text-amber-400 mb-1">
            False Negatives ({{ result.falseNegatives.length }})
          </h4>
          <div v-if="result.falseNegatives.length === 0" class="text-xs text-gray-500">
            None
          </div>
          <div v-else class="flex flex-wrap gap-1">
            <span
              v-for="head in result.falseNegatives.slice(0, 10)"
              :key="`fn-${head[0]}-${head[1]}`"
              class="text-xs font-mono px-1.5 py-0.5 rounded bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
            >
              {{ formatHead(head) }}
            </span>
            <span v-if="result.falseNegatives.length > 10" class="text-xs text-gray-500">
              +{{ result.falseNegatives.length - 10 }} more
            </span>
          </div>
        </div>
      </div>

      <!-- Reference -->
      <div class="text-xs text-gray-500 border-t border-gray-200 dark:border-gray-700 pt-2">
        Validated against known {{ modelType }} heads from Wang et al. 2022
      </div>
    </div>
  </div>
</template>
