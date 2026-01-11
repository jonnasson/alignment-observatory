<script setup lang="ts">
/**
 * AttentionHeatmap Component
 *
 * Renders an interactive attention pattern heatmap.
 */

import { ref, computed, watch, onMounted, type PropType } from 'vue'
import { useColorScale, type ColorScaleName } from '@/composables/useColorScale'

interface AttentionCell {
  queryIdx: number
  keyIdx: number
  value: number
  queryToken: string
  keyToken: string
}

const props = defineProps({
  /** 2D attention matrix [seq_q, seq_k] */
  matrix: {
    type: Array as PropType<number[][]>,
    required: true,
  },
  /** Token labels for rows (queries) and columns (keys) */
  tokens: {
    type: Array as PropType<string[]>,
    default: () => [],
  },
  /** Color scale name */
  colorScale: {
    type: String as PropType<ColorScaleName>,
    default: 'viridis',
  },
  /** Cell size in pixels */
  cellSize: {
    type: Number,
    default: 24,
  },
  /** Show token labels */
  showLabels: {
    type: Boolean,
    default: true,
  },
  /** Show values on hover */
  showTooltip: {
    type: Boolean,
    default: true,
  },
  /** Min value for color scale (auto if undefined) */
  minValue: {
    type: Number,
    default: undefined,
  },
  /** Max value for color scale (auto if undefined) */
  maxValue: {
    type: Number,
    default: undefined,
  },
})

const emit = defineEmits<{
  (e: 'cellHover', cell: AttentionCell | null): void
  (e: 'cellClick', cell: AttentionCell): void
}>()

// Refs
const canvasRef = ref<HTMLCanvasElement | null>(null)
const hoveredCell = ref<AttentionCell | null>(null)
const tooltipPosition = ref({ x: 0, y: 0 })


// Compute min/max from data if not provided
const dataRange = computed(() => {
  let min = props.minValue ?? Infinity
  let max = props.maxValue ?? -Infinity

  if (props.minValue === undefined || props.maxValue === undefined) {
    for (const row of props.matrix) {
      for (const val of row) {
        if (props.minValue === undefined) min = Math.min(min, val)
        if (props.maxValue === undefined) max = Math.max(max, val)
      }
    }
  }

  return {
    min: min === Infinity ? 0 : min,
    max: max === -Infinity ? 1 : max,
  }
})

// Color scale
const colorScaleOptions = computed(() => ({
  scaleName: props.colorScale,
  min: dataRange.value.min,
  max: dataRange.value.max,
}))

const { getColor } = useColorScale(colorScaleOptions)

// Dimensions
const seqLen = computed(() => props.matrix.length)
const labelWidth = computed(() => (props.showLabels ? 80 : 0))
const labelHeight = computed(() => (props.showLabels ? 24 : 0))
const gridWidth = computed(() => seqLen.value * props.cellSize)
const gridHeight = computed(() => seqLen.value * props.cellSize)
const totalWidth = computed(() => gridWidth.value + labelWidth.value)
const totalHeight = computed(() => gridHeight.value + labelHeight.value)

// Get token label with truncation
function getTokenLabel(idx: number): string {
  const token = props.tokens[idx] ?? `[${idx}]`
  return token.length > 8 ? token.slice(0, 7) + '...' : token
}

// Handle mouse move on grid
function handleMouseMove(event: MouseEvent): void {
  if (!canvasRef.value) return

  const rect = canvasRef.value.getBoundingClientRect()
  const x = event.clientX - rect.left - labelWidth.value
  const y = event.clientY - rect.top - labelHeight.value

  const keyIdx = Math.floor(x / props.cellSize)
  const queryIdx = Math.floor(y / props.cellSize)

  if (queryIdx >= 0 && queryIdx < seqLen.value && keyIdx >= 0 && keyIdx < seqLen.value) {
    const row = props.matrix[queryIdx]
    const value = row?.[keyIdx] ?? 0
    const cell: AttentionCell = {
      queryIdx,
      keyIdx,
      value,
      queryToken: props.tokens[queryIdx] ?? `[${queryIdx}]`,
      keyToken: props.tokens[keyIdx] ?? `[${keyIdx}]`,
    }
    hoveredCell.value = cell
    tooltipPosition.value = { x: event.clientX, y: event.clientY }
    emit('cellHover', cell)
  } else {
    hoveredCell.value = null
    emit('cellHover', null)
  }
}

function handleMouseLeave(): void {
  hoveredCell.value = null
  emit('cellHover', null)
}

function handleClick(): void {
  if (hoveredCell.value) {
    emit('cellClick', hoveredCell.value)
  }
}

// Draw the heatmap
function drawHeatmap(): void {
  const canvas = canvasRef.value
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  // Set canvas size
  canvas.width = totalWidth.value
  canvas.height = totalHeight.value

  // Clear
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Draw cells
  for (let qIdx = 0; qIdx < props.matrix.length; qIdx++) {
    const row = props.matrix[qIdx]
    if (!row) continue

    for (let kIdx = 0; kIdx < row.length; kIdx++) {
      const value = row[kIdx] ?? 0
      const color = getColor(value, 'hex') as string

      const x = labelWidth.value + kIdx * props.cellSize
      const y = labelHeight.value + qIdx * props.cellSize

      ctx.fillStyle = color
      ctx.fillRect(x, y, props.cellSize, props.cellSize)

      // Highlight hovered cell
      if (hoveredCell.value?.queryIdx === qIdx && hoveredCell.value?.keyIdx === kIdx) {
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.strokeRect(x + 1, y + 1, props.cellSize - 2, props.cellSize - 2)
      }
    }
  }

  // Draw labels if enabled
  if (props.showLabels) {
    ctx.fillStyle = 'currentColor'
    ctx.font = '11px Inter, system-ui, sans-serif'
    ctx.textBaseline = 'middle'

    // Column labels (key tokens) - rotated
    ctx.save()
    for (let i = 0; i < seqLen.value; i++) {
      const x = labelWidth.value + i * props.cellSize + props.cellSize / 2
      const y = labelHeight.value - 4

      ctx.save()
      ctx.translate(x, y)
      ctx.rotate(-Math.PI / 4)
      ctx.textAlign = 'left'
      ctx.fillStyle = '#6b7280'
      ctx.fillText(getTokenLabel(i), 0, 0)
      ctx.restore()
    }
    ctx.restore()

    // Row labels (query tokens)
    ctx.textAlign = 'right'
    for (let i = 0; i < seqLen.value; i++) {
      const x = labelWidth.value - 8
      const y = labelHeight.value + i * props.cellSize + props.cellSize / 2

      ctx.fillStyle = '#6b7280'
      ctx.fillText(getTokenLabel(i), x, y)
    }
  }
}

// Redraw on changes
watch([() => props.matrix, () => props.colorScale, hoveredCell], drawHeatmap, { deep: true })

onMounted(() => {
  drawHeatmap()
})

// No methods to expose since zoom/pan was removed
</script>

<template>
  <div class="attention-heatmap relative">
    <!-- Heatmap container -->
    <div
      @mousemove="handleMouseMove"
      @mouseleave="handleMouseLeave"
      @click="handleClick"
    >
      <canvas ref="canvasRef" class="block" />
    </div>

    <!-- Tooltip -->
    <Teleport to="body">
      <div
        v-if="showTooltip && hoveredCell"
        class="fixed z-50 px-2 py-1 text-xs bg-gray-900 text-white rounded shadow-lg pointer-events-none"
        :style="{
          left: `${tooltipPosition.x + 12}px`,
          top: `${tooltipPosition.y + 12}px`,
        }"
      >
        <div class="font-medium">
          {{ hoveredCell.queryToken }} → {{ hoveredCell.keyToken }}
        </div>
        <div class="text-gray-300">
          Attention: {{ hoveredCell.value.toFixed(4) }}
        </div>
        <div class="text-gray-400 text-[10px]">
          [{{ hoveredCell.queryIdx }}, {{ hoveredCell.keyIdx }}]
        </div>
      </div>
    </Teleport>
  </div>
</template>

