<script setup lang="ts">
/**
 * AttentionStats Component
 *
 * Displays attention pattern statistics and metrics.
 */

import { computed, type PropType } from 'vue'
import type { HeadClassification } from '@/types'

interface AttentionStatistics {
  entropy: number
  sparsity: number
  maxAttention: number
  avgAttention: number
}

const props = defineProps({
  /** Attention statistics */
  stats: {
    type: Object as PropType<AttentionStatistics>,
    required: true,
  },
  /** Head classification */
  classification: {
    type: String as PropType<HeadClassification>,
    default: 'other',
  },
  /** Layer index */
  layer: {
    type: Number,
    default: null,
  },
  /** Head index */
  head: {
    type: Number,
    default: null,
  },
  /** Sequence length for context */
  seqLen: {
    type: Number,
    default: null,
  },
})

// Classification display info
const classificationInfo: Record<
  HeadClassification,
  { label: string; description: string; color: string }
> = {
  previous_token: {
    label: 'Previous Token',
    description: 'Attends primarily to the previous position',
    color: 'text-blue-600 bg-blue-100 dark:bg-blue-900/30',
  },
  bos: {
    label: 'BOS',
    description: 'Attends to beginning of sequence token',
    color: 'text-violet-600 bg-violet-100 dark:bg-violet-900/30',
  },
  uniform: {
    label: 'Uniform',
    description: 'Roughly uniform attention distribution',
    color: 'text-gray-600 bg-gray-100 dark:bg-gray-800',
  },
  induction: {
    label: 'Induction',
    description: 'Pattern matching for in-context learning',
    color: 'text-emerald-600 bg-emerald-100 dark:bg-emerald-900/30',
  },
  name_mover: {
    label: 'Name Mover',
    description: 'Copies names from context (IOI circuit)',
    color: 'text-amber-600 bg-amber-100 dark:bg-amber-900/30',
  },
  s_inhibition: {
    label: 'S-Inhibition',
    description: 'Inhibits subject name (IOI circuit)',
    color: 'text-red-600 bg-red-100 dark:bg-red-900/30',
  },
  backup_name_mover: {
    label: 'Backup Name Mover',
    description: 'Secondary name moving head',
    color: 'text-orange-600 bg-orange-100 dark:bg-orange-900/30',
  },
  duplicate_token: {
    label: 'Duplicate Token',
    description: 'Attends to duplicate tokens',
    color: 'text-cyan-600 bg-cyan-100 dark:bg-cyan-900/30',
  },
  mixed: {
    label: 'Mixed',
    description: 'Multiple attention patterns detected',
    color: 'text-purple-600 bg-purple-100 dark:bg-purple-900/30',
  },
  other: {
    label: 'Other',
    description: 'No specific pattern detected',
    color: 'text-gray-500 bg-gray-100 dark:bg-gray-800',
  },
}

const classInfo = computed(() =>
  classificationInfo[props.classification] ?? classificationInfo.other
)

// Max theoretical entropy for uniform distribution
const maxEntropy = computed(() => {
  if (!props.seqLen) return null
  return Math.log2(props.seqLen)
})

// Entropy as percentage of max
const entropyPercent = computed(() => {
  if (!maxEntropy.value) return null
  return (props.stats.entropy / maxEntropy.value) * 100
})

// Format number
function formatNumber(val: number, decimals = 3): string {
  return val.toFixed(decimals)
}

function formatPercent(val: number): string {
  return `${(val * 100).toFixed(1)}%`
}
</script>

<template>
  <div class="attention-stats space-y-4">
    <!-- Head identification -->
    <div v-if="layer !== null && head !== null" class="flex items-center gap-2">
      <span class="text-sm text-gray-500 dark:text-gray-400">
        Layer {{ layer }}, Head {{ head }}
      </span>
    </div>

    <!-- Classification badge -->
    <div class="flex items-start gap-3">
      <span
        :class="[
          'inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium',
          classInfo.color,
        ]"
      >
        {{ classInfo.label }}
      </span>
      <p class="text-sm text-gray-600 dark:text-gray-400 flex-1">
        {{ classInfo.description }}
      </p>
    </div>

    <!-- Statistics grid -->
    <div class="grid grid-cols-2 gap-4">
      <!-- Entropy -->
      <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ formatNumber(stats.entropy) }}
          <span v-if="maxEntropy" class="text-gray-400 text-xs">
            / {{ formatNumber(maxEntropy) }}
          </span>
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Entropy</div>
        <div v-if="entropyPercent !== null" class="mt-1">
          <div class="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              class="h-full bg-blue-500 rounded-full transition-all"
              :style="{ width: `${Math.min(100, entropyPercent)}%` }"
            />
          </div>
        </div>
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

      <!-- Max attention -->
      <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ formatNumber(stats.maxAttention) }}
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Max Attention</div>
      </div>

      <!-- Avg attention -->
      <div class="p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ formatNumber(stats.avgAttention, 4) }}
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Avg Attention</div>
      </div>
    </div>

    <!-- Additional info -->
    <div v-if="seqLen" class="text-xs text-gray-500 dark:text-gray-400">
      Sequence length: {{ seqLen }} tokens
    </div>
  </div>
</template>

