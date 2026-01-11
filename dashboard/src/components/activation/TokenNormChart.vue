<script setup lang="ts">
/**
 * TokenNormChart Component
 *
 * Bar chart showing L2 norms for each token.
 */

import { ref, computed, type PropType } from 'vue'

interface TokenNormData {
  tokenIdx: number
  token: string
  norm: number
}

const props = defineProps({
  /** Token norms array */
  norms: {
    type: Array as PropType<number[]>,
    required: true,
  },
  /** Token labels */
  tokens: {
    type: Array as PropType<string[]>,
    default: () => [],
  },
  /** Chart height */
  height: {
    type: Number,
    default: 200,
  },
  /** Bar color */
  barColor: {
    type: String,
    default: '#3b82f6',
  },
  /** Show token labels on x-axis */
  showLabels: {
    type: Boolean,
    default: true,
  },
  /** Selected token index */
  selectedToken: {
    type: Number,
    default: null,
  },
})

const emit = defineEmits<{
  (e: 'tokenClick', tokenIdx: number): void
  (e: 'tokenHover', tokenIdx: number | null): void
}>()

// Refs
const hoveredToken = ref<number | null>(null)

// Computed
const maxNorm = computed(() => Math.max(...props.norms, 0.001))

const data = computed((): TokenNormData[] => {
  return props.norms.map((norm, idx) => ({
    tokenIdx: idx,
    token: props.tokens[idx] ?? `[${idx}]`,
    norm,
  }))
})

// Layout
const margin = { top: 20, right: 20, bottom: 40, left: 50 }
const chartWidth = computed(() => Math.max(props.norms.length * 30, 300))
const innerWidth = computed(() => chartWidth.value - margin.left - margin.right)
const innerHeight = computed(() => props.height - margin.top - margin.bottom)
const barWidth = computed(() => Math.max(innerWidth.value / props.norms.length - 2, 4))

// Scale functions
function xScale(idx: number): number {
  return margin.left + (idx / props.norms.length) * innerWidth.value + barWidth.value / 2
}

function yScale(value: number): number {
  return margin.top + innerHeight.value - (value / maxNorm.value) * innerHeight.value
}

function barHeight(value: number): number {
  return (value / maxNorm.value) * innerHeight.value
}

// Y-axis ticks
const yTicks = computed(() => {
  const ticks: number[] = []
  const step = maxNorm.value / 4
  for (let i = 0; i <= 4; i++) {
    ticks.push(step * i)
  }
  return ticks
})

// Event handlers
function handleBarHover(idx: number | null): void {
  hoveredToken.value = idx
  emit('tokenHover', idx)
}

function handleBarClick(idx: number): void {
  emit('tokenClick', idx)
}

// Truncate token label
function truncateToken(token: string, maxLen = 6): string {
  return token.length > maxLen ? token.slice(0, maxLen - 1) + '...' : token
}
</script>

<template>
  <div class="token-norm-chart overflow-x-auto">
    <svg
      :width="chartWidth"
      :height="height"
      class="select-none"
    >
      <!-- Y-axis -->
      <g class="y-axis">
        <line
          :x1="margin.left"
          :y1="margin.top"
          :x2="margin.left"
          :y2="margin.top + innerHeight"
          stroke="#e5e7eb"
        />
        <g v-for="tick in yTicks" :key="tick">
          <line
            :x1="margin.left - 4"
            :y1="yScale(tick)"
            :x2="margin.left"
            :y2="yScale(tick)"
            stroke="#9ca3af"
          />
          <text
            :x="margin.left - 8"
            :y="yScale(tick)"
            text-anchor="end"
            dominant-baseline="middle"
            class="text-[10px] fill-gray-500"
          >
            {{ tick.toFixed(1) }}
          </text>
          <!-- Grid line -->
          <line
            :x1="margin.left"
            :y1="yScale(tick)"
            :x2="margin.left + innerWidth"
            :y2="yScale(tick)"
            stroke="#f3f4f6"
            stroke-dasharray="2,2"
          />
        </g>
        <!-- Y-axis label -->
        <text
          :x="15"
          :y="margin.top + innerHeight / 2"
          text-anchor="middle"
          dominant-baseline="middle"
          transform="rotate(-90, 15, 110)"
          class="text-xs fill-gray-500"
        >
          L2 Norm
        </text>
      </g>

      <!-- X-axis -->
      <line
        :x1="margin.left"
        :y1="margin.top + innerHeight"
        :x2="margin.left + innerWidth"
        :y2="margin.top + innerHeight"
        stroke="#e5e7eb"
      />

      <!-- Bars -->
      <g class="bars">
        <g
          v-for="(d, idx) in data"
          :key="idx"
          class="bar-group cursor-pointer"
          @mouseenter="handleBarHover(idx)"
          @mouseleave="handleBarHover(null)"
          @click="handleBarClick(idx)"
        >
          <rect
            :x="xScale(idx) - barWidth / 2"
            :y="yScale(d.norm)"
            :width="barWidth"
            :height="barHeight(d.norm)"
            :fill="selectedToken === idx ? '#1d4ed8' : hoveredToken === idx ? '#60a5fa' : barColor"
            class="transition-colors"
          />
          <!-- Token label -->
          <text
            v-if="showLabels && barWidth >= 10"
            :x="xScale(idx)"
            :y="margin.top + innerHeight + 12"
            text-anchor="middle"
            dominant-baseline="hanging"
            class="text-[9px] fill-gray-500"
            :transform="`rotate(-45, ${xScale(idx)}, ${margin.top + innerHeight + 12})`"
          >
            {{ truncateToken(d.token) }}
          </text>
        </g>
      </g>

      <!-- Hover tooltip -->
      <g v-if="hoveredToken !== null" class="tooltip">
        <rect
          :x="xScale(hoveredToken) - 40"
          :y="yScale(data[hoveredToken]?.norm ?? 0) - 35"
          width="80"
          height="30"
          rx="4"
          fill="#1f2937"
          fill-opacity="0.9"
        />
        <text
          :x="xScale(hoveredToken)"
          :y="yScale(data[hoveredToken]?.norm ?? 0) - 22"
          text-anchor="middle"
          class="text-[10px] fill-white font-medium"
        >
          {{ data[hoveredToken]?.token }}
        </text>
        <text
          :x="xScale(hoveredToken)"
          :y="yScale(data[hoveredToken]?.norm ?? 0) - 10"
          text-anchor="middle"
          class="text-[10px] fill-gray-300"
        >
          {{ data[hoveredToken]?.norm.toFixed(3) }}
        </text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.token-norm-chart {
  min-height: 100px;
}
</style>
