<script setup lang="ts">
/**
 * AttentionFlow3DControls - Control panel for 3D attention visualization
 *
 * Provides controls for:
 * - Attention threshold slider
 * - Layer selection
 * - Head selection
 * - View presets
 * - Animation toggle
 */

import { computed } from 'vue'
import {
  ArrowPathIcon,
  PlayIcon,
  PauseIcon,
} from '@heroicons/vue/24/outline'

interface Props {
  /** Current threshold value */
  threshold: number
  /** Number of layers */
  numLayers: number
  /** Number of heads */
  numHeads: number
  /** Currently selected layers */
  selectedLayers: number[]
  /** Currently selected heads */
  selectedHeads: number[]
  /** Animation enabled */
  animate: boolean
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'update:threshold', value: number): void
  (e: 'update:selectedLayers', value: number[]): void
  (e: 'update:selectedHeads', value: number[]): void
  (e: 'update:animate', value: boolean): void
  (e: 'resetView'): void
  (e: 'setView', view: 'top' | 'front' | 'side' | 'isometric'): void
}>()

// Layer options
const layerOptions = computed(() => [
  { value: '', label: 'All Layers' },
  ...Array.from({ length: props.numLayers }, (_, i) => ({
    value: String(i),
    label: `Layer ${i}`,
  })),
])

// Head options
const headOptions = computed(() => [
  { value: '', label: 'All Heads' },
  ...Array.from({ length: props.numHeads }, (_, i) => ({
    value: String(i),
    label: `Head ${i}`,
  })),
])

// Selected layer value for select
const selectedLayerValue = computed({
  get: () => (props.selectedLayers.length === 1 ? String(props.selectedLayers[0]) : ''),
  set: (val) => {
    emit('update:selectedLayers', val ? [parseInt(val)] : [])
  },
})

// Selected head value for select
const selectedHeadValue = computed({
  get: () => (props.selectedHeads.length === 1 ? String(props.selectedHeads[0]) : ''),
  set: (val) => {
    emit('update:selectedHeads', val ? [parseInt(val)] : [])
  },
})

// View presets
const viewPresets = [
  { id: 'isometric', label: 'Iso' },
  { id: 'top', label: 'Top' },
  { id: 'front', label: 'Front' },
  { id: 'side', label: 'Side' },
] as const
</script>

<template>
  <div class="attention-3d-controls flex flex-wrap items-center gap-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800">
    <!-- Threshold slider -->
    <div class="flex items-center gap-2">
      <label class="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">
        Threshold
      </label>
      <input
        type="range"
        :value="threshold"
        min="0"
        max="1"
        step="0.01"
        class="w-24 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
        @input="emit('update:threshold', parseFloat(($event.target as HTMLInputElement).value))"
      />
      <span class="text-xs font-mono text-gray-500 w-10">
        {{ threshold.toFixed(2) }}
      </span>
    </div>

    <!-- Divider -->
    <div class="h-6 w-px bg-gray-300 dark:bg-gray-700" />

    <!-- Layer selection -->
    <div class="flex items-center gap-2">
      <label class="text-sm text-gray-600 dark:text-gray-400">Layer</label>
      <select
        :value="selectedLayerValue"
        class="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
        @change="selectedLayerValue = ($event.target as HTMLSelectElement).value"
      >
        <option
          v-for="opt in layerOptions"
          :key="opt.value"
          :value="opt.value"
        >
          {{ opt.label }}
        </option>
      </select>
    </div>

    <!-- Head selection -->
    <div class="flex items-center gap-2">
      <label class="text-sm text-gray-600 dark:text-gray-400">Head</label>
      <select
        :value="selectedHeadValue"
        class="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
        @change="selectedHeadValue = ($event.target as HTMLSelectElement).value"
      >
        <option
          v-for="opt in headOptions"
          :key="opt.value"
          :value="opt.value"
        >
          {{ opt.label }}
        </option>
      </select>
    </div>

    <!-- Divider -->
    <div class="h-6 w-px bg-gray-300 dark:bg-gray-700" />

    <!-- View presets -->
    <div class="flex items-center gap-1">
      <span class="text-sm text-gray-600 dark:text-gray-400 mr-1">View:</span>
      <button
        v-for="preset in viewPresets"
        :key="preset.id"
        class="px-2 py-1 text-xs rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
        :title="`${preset.label} view`"
        @click="emit('setView', preset.id)"
      >
        {{ preset.label }}
      </button>
    </div>

    <!-- Reset button -->
    <button
      class="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
      title="Reset view"
      @click="emit('resetView')"
    >
      <ArrowPathIcon class="w-4 h-4" />
    </button>

    <!-- Animation toggle -->
    <button
      :class="[
        'p-1.5 rounded',
        animate
          ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
          : 'hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400',
      ]"
      title="Toggle animation"
      @click="emit('update:animate', !animate)"
    >
      <component :is="animate ? PauseIcon : PlayIcon" class="w-4 h-4" />
    </button>
  </div>
</template>

<style scoped>
input[type='range']::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #6366f1;
  cursor: pointer;
}

input[type='range']::-moz-range-thumb {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #6366f1;
  cursor: pointer;
  border: none;
}
</style>
