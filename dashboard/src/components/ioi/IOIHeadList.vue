<script setup lang="ts">
/**
 * IOIHeadList Component
 *
 * Displays detected IOI circuit heads organized by component type.
 */

import { computed, type PropType } from 'vue'
import type { IOIHead, IOIComponentType } from '@/types'
import { ioiComponentColors, ioiComponentLabels } from '@/composables/useIOI'

const props = defineProps({
  /** Name mover heads */
  nameMoverHeads: {
    type: Array as PropType<IOIHead[]>,
    default: () => [],
  },
  /** S-inhibition heads */
  sInhibitionHeads: {
    type: Array as PropType<IOIHead[]>,
    default: () => [],
  },
  /** Duplicate token heads */
  duplicateTokenHeads: {
    type: Array as PropType<IOIHead[]>,
    default: () => [],
  },
  /** Previous token heads */
  previousTokenHeads: {
    type: Array as PropType<IOIHead[]>,
    default: () => [],
  },
  /** Backup name mover heads */
  backupNameMoverHeads: {
    type: Array as PropType<IOIHead[]>,
    default: () => [],
  },
  /** Show scores */
  showScores: {
    type: Boolean,
    default: true,
  },
  /** Highlighted heads */
  highlighted: {
    type: Array as PropType<[number, number][]>,
    default: () => [],
  },
})

const emit = defineEmits<{
  (e: 'headSelect', head: IOIHead): void
  (e: 'headHover', head: IOIHead | null): void
}>()

// Component sections
const sections = computed(() => [
  {
    type: 'name_mover' as IOIComponentType,
    label: ioiComponentLabels.name_mover,
    color: ioiComponentColors.name_mover,
    heads: props.nameMoverHeads,
    description: 'Copy IO name to output',
  },
  {
    type: 's_inhibition' as IOIComponentType,
    label: ioiComponentLabels.s_inhibition,
    color: ioiComponentColors.s_inhibition,
    heads: props.sInhibitionHeads,
    description: 'Suppress subject copying',
  },
  {
    type: 'duplicate_token' as IOIComponentType,
    label: ioiComponentLabels.duplicate_token,
    color: ioiComponentColors.duplicate_token,
    heads: props.duplicateTokenHeads,
    description: 'Detect repeated subject',
  },
  {
    type: 'previous_token' as IOIComponentType,
    label: ioiComponentLabels.previous_token,
    color: ioiComponentColors.previous_token,
    heads: props.previousTokenHeads,
    description: 'Track local context',
  },
  {
    type: 'backup_name_mover' as IOIComponentType,
    label: ioiComponentLabels.backup_name_mover,
    color: ioiComponentColors.backup_name_mover,
    heads: props.backupNameMoverHeads,
    description: 'Secondary name movers',
  },
])

// Check if head is highlighted
function isHighlighted(head: IOIHead): boolean {
  return props.highlighted.some(([l, h]) => l === head.layer && h === head.head)
}

// Format head
function formatHead(head: IOIHead): string {
  return `L${head.layer}H${head.head}`
}

// Total head count
const totalHeads = computed(() => {
  return sections.value.reduce((sum, s) => sum + s.heads.length, 0)
})
</script>

<template>
  <div class="ioi-head-list space-y-4">
    <div v-if="totalHeads === 0" class="text-sm text-gray-500 text-center py-4">
      No IOI circuit detected yet
    </div>

    <div v-for="section in sections" :key="section.type" class="space-y-2">
      <!-- Section header -->
      <div class="flex items-center gap-2">
        <div
          class="w-3 h-3 rounded"
          :style="{ backgroundColor: section.color }"
        />
        <span class="text-sm font-medium text-gray-700 dark:text-gray-300">
          {{ section.label }}
        </span>
        <span class="text-xs text-gray-400">
          ({{ section.heads.length }})
        </span>
      </div>

      <!-- Heads list -->
      <div v-if="section.heads.length > 0" class="flex flex-wrap gap-1.5 pl-5">
        <div
          v-for="head in section.heads"
          :key="`${head.layer}-${head.head}`"
          class="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-mono cursor-pointer transition-all"
          :class="[
            isHighlighted(head)
              ? 'ring-2 ring-blue-500'
              : 'hover:ring-2 hover:ring-gray-300 dark:hover:ring-gray-600',
          ]"
          :style="{
            backgroundColor: section.color + '20',
            color: section.color,
          }"
          @click="emit('headSelect', head)"
          @mouseenter="emit('headHover', head)"
          @mouseleave="emit('headHover', null)"
        >
          <span>{{ formatHead(head) }}</span>
          <span v-if="showScores" class="text-gray-500 dark:text-gray-400">
            {{ (head.score * 100).toFixed(0) }}%
          </span>
        </div>
      </div>

      <div v-else class="text-xs text-gray-400 pl-5">
        None detected
      </div>

      <!-- Section description -->
      <div class="text-xs text-gray-500 pl-5">
        {{ section.description }}
      </div>
    </div>

    <!-- Total count -->
    <div v-if="totalHeads > 0" class="text-xs text-gray-500 border-t border-gray-200 dark:border-gray-700 pt-2">
      Total: {{ totalHeads }} heads detected
    </div>
  </div>
</template>
