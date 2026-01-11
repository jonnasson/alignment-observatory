<script setup lang="ts">
/**
 * LayerStats Component
 *
 * Displays statistics for layer activations.
 */

import { computed, type PropType } from 'vue'

interface ActivationStats {
  mean: number
  variance: number
  min: number
  max: number
  l2Norm: number
  sparsity: number
}

const props = defineProps({
  /** Activation statistics */
  stats: {
    type: Object as PropType<ActivationStats>,
    required: true,
  },
  /** Layer number */
  layer: {
    type: Number,
    default: null,
  },
  /** Component type */
  component: {
    type: String as PropType<'residual' | 'attention_out' | 'mlp_out'>,
    default: 'residual',
  },
  /** Sequence length */
  seqLen: {
    type: Number,
    default: null,
  },
  /** Hidden dimension size */
  hiddenDim: {
    type: Number,
    default: null,
  },
})

// Component display names
const componentNames: Record<string, string> = {
  residual: 'Residual Stream',
  attention_out: 'Attention Output',
  mlp_out: 'MLP Output',
}

const componentName = computed(() => componentNames[props.component] ?? props.component)

// Format numbers
function formatNumber(val: number, decimals = 4): string {
  if (Math.abs(val) < 0.0001 && val !== 0) {
    return val.toExponential(2)
  }
  return val.toFixed(decimals)
}

function formatPercent(val: number): string {
  return `${(val * 100).toFixed(1)}%`
}

// Standard deviation from variance
const stdDev = computed(() => Math.sqrt(props.stats.variance))

// Value range
const range = computed(() => props.stats.max - props.stats.min)
</script>

<template>
  <div class="layer-stats space-y-4">
    <!-- Header -->
    <div v-if="layer !== null" class="flex items-center justify-between">
      <span class="text-sm font-medium text-gray-900 dark:text-gray-100">
        Layer {{ layer }}
      </span>
      <span class="text-xs px-2 py-0.5 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">
        {{ componentName }}
      </span>
    </div>

    <!-- Dimensions info -->
    <div v-if="seqLen && hiddenDim" class="text-xs text-gray-500 dark:text-gray-400">
      Shape: {{ seqLen }} tokens x {{ hiddenDim }} dimensions
    </div>

    <!-- Statistics grid -->
    <div class="grid grid-cols-2 gap-3">
      <!-- Mean -->
      <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ formatNumber(stats.mean) }}
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Mean</div>
      </div>

      <!-- Std Dev -->
      <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ formatNumber(stdDev) }}
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Std Dev</div>
      </div>

      <!-- Min -->
      <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ formatNumber(stats.min) }}
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Min</div>
      </div>

      <!-- Max -->
      <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ formatNumber(stats.max) }}
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Max</div>
      </div>

      <!-- L2 Norm -->
      <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ formatNumber(stats.l2Norm, 2) }}
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">L2 Norm</div>
      </div>

      <!-- Sparsity -->
      <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ formatPercent(stats.sparsity) }}
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Sparsity</div>
        <div class="mt-1">
          <div class="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              class="h-full bg-emerald-500 rounded-full transition-all"
              :style="{ width: `${stats.sparsity * 100}%` }"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Range visualization -->
    <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
      <div class="text-xs text-gray-500 dark:text-gray-400 mb-2">Value Range</div>
      <div class="relative h-6 bg-gradient-to-r from-blue-500 via-gray-300 to-red-500 rounded">
        <!-- Zero marker if in range -->
        <div
          v-if="stats.min < 0 && stats.max > 0"
          class="absolute top-0 bottom-0 w-0.5 bg-black"
          :style="{ left: `${((0 - stats.min) / range) * 100}%` }"
        />
        <!-- Mean marker -->
        <div
          class="absolute top-0 bottom-0 w-1 bg-yellow-400 rounded"
          :style="{ left: `${((stats.mean - stats.min) / range) * 100}%` }"
          title="Mean"
        />
      </div>
      <div class="flex justify-between text-[10px] text-gray-400 mt-1">
        <span>{{ formatNumber(stats.min, 2) }}</span>
        <span>{{ formatNumber(stats.max, 2) }}</span>
      </div>
    </div>
  </div>
</template>
