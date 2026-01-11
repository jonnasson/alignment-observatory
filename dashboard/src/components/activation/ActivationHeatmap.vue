<script setup lang="ts">
/**
 * ActivationHeatmap Component
 *
 * Renders a heatmap of activations (tokens x hidden dimensions).
 * Supports virtual scrolling for large hidden dimensions.
 */

import { ref, computed, watch, onMounted, type PropType } from 'vue'
import { useColorScale, type ColorScaleName } from '@/composables/useColorScale'
import { useZoomPan } from '@/composables/useZoomPan'

interface HoveredCell {
  tokenIdx: number
  dimIdx: number
  value: number
  token: string
}

const props = defineProps({
  /** 2D activation matrix [seq_len, hidden_dim] */
  matrix: {
    type: Array as PropType<number[][]>,
    required: true,
  },
  /** Token labels */
  tokens: {
    type: Array as PropType<string[]>,
    default: () => [],
  },
  /** Color scale name */
  colorScale: {
    type: String as PropType<ColorScaleName>,
    default: 'rdbu',
  },
  /** Use diverging normalization (centered at 0) */
  diverging: {
    type: Boolean,
    default: true,
  },
  /** Cell height in pixels */
  cellHeight: {
    type: Number,
    default: 20,
  },
  /** Cell width in pixels */
  cellWidth: {
    type: Number,
    default: 2,
  },
  /** Show token labels */
  showLabels: {
    type: Boolean,
    default: true,
  },
  /** Max dimensions to display (for performance) */
  maxDimensions: {
    type: Number,
    default: 768,
  },
  /** Dimension range start */
  dimStart: {
    type: Number,
    default: 0,
  },
  /** Min value for color scale */
  minValue: {
    type: Number,
    default: undefined,
  },
  /** Max value for color scale */
  maxValue: {
    type: Number,
    default: undefined,
  },
})

const emit = defineEmits<{
  (e: 'cellHover', cell: HoveredCell | null): void
  (e: 'cellClick', cell: HoveredCell): void
  (e: 'tokenClick', tokenIdx: number): void
  (e: 'dimensionClick', dimIdx: number): void
}>()

// Refs
const containerRef = ref<HTMLElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const hoveredCell = ref<HoveredCell | null>(null)
const tooltipPosition = ref({ x: 0, y: 0 })

// Zoom/pan
const { zoom, pan, transform, isDragging, reset: resetView } = useZoomPan(containerRef, {
  minZoom: 0.5,
  maxZoom: 4,
  enableWheel: true,
  enableDrag: true,
})

// Compute display range
const displayDimEnd = computed(() =>
  Math.min(props.dimStart + props.maxDimensions, props.matrix[0]?.length ?? 0)
)
const displayDimCount = computed(() => displayDimEnd.value - props.dimStart)

// Compute data range
const dataRange = computed(() => {
  if (props.minValue !== undefined && props.maxValue !== undefined) {
    return { min: props.minValue, max: props.maxValue }
  }

  let min = Infinity
  let max = -Infinity

  for (const row of props.matrix) {
    for (let d = props.dimStart; d < displayDimEnd.value; d++) {
      const val = row[d] ?? 0
      min = Math.min(min, val)
      max = Math.max(max, val)
    }
  }

  if (props.diverging) {
    const absMax = Math.max(Math.abs(min), Math.abs(max))
    return { min: -absMax, max: absMax }
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
const labelHeight = 24
const gridWidth = computed(() => displayDimCount.value * props.cellWidth)
const gridHeight = computed(() => seqLen.value * props.cellHeight)
const totalWidth = computed(() => gridWidth.value + labelWidth.value)
const totalHeight = computed(() => gridHeight.value + labelHeight)

// Get token label with truncation
function getTokenLabel(idx: number): string {
  const token = props.tokens[idx] ?? `[${idx}]`
  return token.length > 10 ? token.slice(0, 9) + '...' : token
}

// Handle mouse move
function handleMouseMove(event: MouseEvent): void {
  if (isDragging.value || !canvasRef.value) return

  const rect = canvasRef.value.getBoundingClientRect()
  const x = (event.clientX - rect.left - labelWidth.value) / zoom.value - pan.value.x / zoom.value
  const y = (event.clientY - rect.top - labelHeight) / zoom.value - pan.value.y / zoom.value

  const dimIdx = Math.floor(x / props.cellWidth) + props.dimStart
  const tokenIdx = Math.floor(y / props.cellHeight)

  if (
    tokenIdx >= 0 &&
    tokenIdx < seqLen.value &&
    dimIdx >= props.dimStart &&
    dimIdx < displayDimEnd.value
  ) {
    const row = props.matrix[tokenIdx]
    const value = row?.[dimIdx] ?? 0
    const cell: HoveredCell = {
      tokenIdx,
      dimIdx,
      value,
      token: props.tokens[tokenIdx] ?? `[${tokenIdx}]`,
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
  for (let tIdx = 0; tIdx < props.matrix.length; tIdx++) {
    const row = props.matrix[tIdx]
    if (!row) continue

    for (let d = props.dimStart; d < displayDimEnd.value; d++) {
      const value = row[d] ?? 0
      const color = getColor(value, 'hex') as string

      const x = labelWidth.value + (d - props.dimStart) * props.cellWidth
      const y = labelHeight + tIdx * props.cellHeight

      ctx.fillStyle = color
      ctx.fillRect(x, y, props.cellWidth, props.cellHeight)
    }
  }

  // Draw row separators
  ctx.strokeStyle = 'rgba(0,0,0,0.1)'
  ctx.lineWidth = 0.5
  for (let tIdx = 1; tIdx < props.matrix.length; tIdx++) {
    const y = labelHeight + tIdx * props.cellHeight
    ctx.beginPath()
    ctx.moveTo(labelWidth.value, y)
    ctx.lineTo(totalWidth.value, y)
    ctx.stroke()
  }

  // Draw labels if enabled
  if (props.showLabels) {
    ctx.font = '11px Inter, system-ui, sans-serif'
    ctx.textBaseline = 'middle'
    ctx.textAlign = 'right'

    for (let i = 0; i < seqLen.value; i++) {
      const x = labelWidth.value - 8
      const y = labelHeight + i * props.cellHeight + props.cellHeight / 2

      ctx.fillStyle = '#6b7280'
      ctx.fillText(getTokenLabel(i), x, y)
    }

    // Dimension axis label
    ctx.textAlign = 'center'
    ctx.fillStyle = '#9ca3af'
    ctx.fillText(
      `Dimensions ${props.dimStart}-${displayDimEnd.value - 1}`,
      labelWidth.value + gridWidth.value / 2,
      12
    )
  }

  // Highlight hovered row
  if (hoveredCell.value) {
    const y = labelHeight + hoveredCell.value.tokenIdx * props.cellHeight
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.strokeRect(labelWidth.value, y, gridWidth.value, props.cellHeight)
  }
}

// Redraw on changes
watch(
  [() => props.matrix, () => props.colorScale, () => props.dimStart, hoveredCell],
  drawHeatmap,
  { deep: true }
)

onMounted(() => {
  drawHeatmap()
})

defineExpose({
  resetView,
  zoom,
  pan,
})
</script>

<template>
  <div class="activation-heatmap relative">
    <!-- Controls -->
    <div class="absolute top-2 right-2 z-10 flex gap-1">
      <button
        class="p-1.5 rounded bg-white/80 dark:bg-gray-800/80 hover:bg-white dark:hover:bg-gray-800 shadow-sm"
        title="Reset view"
        @click="resetView"
      >
        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </button>
    </div>

    <!-- Heatmap container -->
    <div
      ref="containerRef"
      class="overflow-hidden cursor-grab active:cursor-grabbing"
      :class="{ 'cursor-grabbing': isDragging }"
      @mousemove="handleMouseMove"
      @mouseleave="handleMouseLeave"
      @click="handleClick"
    >
      <div :style="{ transform, transformOrigin: '0 0' }">
        <canvas ref="canvasRef" class="block" />
      </div>
    </div>

    <!-- Tooltip -->
    <Teleport to="body">
      <div
        v-if="hoveredCell"
        class="fixed z-50 px-2 py-1 text-xs bg-gray-900 text-white rounded shadow-lg pointer-events-none"
        :style="{
          left: `${tooltipPosition.x + 12}px`,
          top: `${tooltipPosition.y + 12}px`,
        }"
      >
        <div class="font-medium">{{ hoveredCell.token }}</div>
        <div class="text-gray-300">
          Dim {{ hoveredCell.dimIdx }}: {{ hoveredCell.value.toFixed(4) }}
        </div>
        <div class="text-gray-400 text-[10px]">
          Token {{ hoveredCell.tokenIdx }}
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.activation-heatmap {
  min-height: 200px;
}
</style>
