<script setup lang="ts">
/**
 * ThreeCanvas - Reusable Three.js canvas wrapper component
 *
 * Provides a container for Three.js visualizations with:
 * - Automatic resize handling
 * - Dark mode support
 * - Performance monitoring overlay
 * - Slot for custom overlay UI (legends, controls)
 */

import { ref, onMounted, onUnmounted, computed } from 'vue'
import * as THREE from 'three'
import { useTheme } from '@/composables/useTheme'
import { useThreeScene } from '@/composables/useThreeScene'
import { useWebGLPerformance } from '@/composables/useWebGLPerformance'

interface Props {
  /** Container height (CSS value) */
  height?: string
  /** Enable performance overlay */
  showPerformance?: boolean
  /** Enable antialiasing */
  antialias?: boolean
  /** Enable shadows */
  enableShadows?: boolean
  /** Auto-start animation loop */
  autoStart?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  height: '400px',
  showPerformance: false,
  antialias: true,
  enableShadows: false,
  autoStart: true,
})

const emit = defineEmits<{
  /** Emitted when scene is initialized */
  (e: 'ready', context: { scene: THREE.Scene; camera: THREE.Camera; renderer: THREE.WebGLRenderer }): void
  /** Emitted on each animation frame */
  (e: 'frame', delta: number): void
  /** Emitted when an error occurs */
  (e: 'error', error: Error): void
}>()

// Container ref
const container = ref<HTMLElement | null>(null)

// Dark mode
const { isDark } = useTheme()

// Three.js scene
const {
  scene,
  camera,
  renderer,
  controls,
  isInitialized,
  isAnimating,
  fps,
  init,
  startAnimation,
  stopAnimation,
  render,
  resetCamera,
  exportImage,
  getRendererInfo,
  dispose,
} = useThreeScene({
  container,
  isDark,
  antialias: props.antialias,
  enableShadows: props.enableShadows,
})

// Performance monitoring
const {
  currentFPS,
  drawCalls,
  triangles,
  currentLOD,
  performanceStatus,
  recordFrame,
} = useWebGLPerformance({ targetFPS: 30 })

// Initialize on mount
onMounted(() => {
  try {
    if (init()) {
      if (scene.value && camera.value && renderer.value) {
        emit('ready', {
          scene: scene.value,
          camera: camera.value,
          renderer: renderer.value,
        })
      }

      if (props.autoStart) {
        startAnimation((delta) => {
          emit('frame', delta)
          const info = getRendererInfo()
          if (info) {
            recordFrame(info)
          }
        })
      }
    }
  } catch (error) {
    emit('error', error instanceof Error ? error : new Error(String(error)))
  }
})

// Cleanup on unmount
onUnmounted(() => {
  dispose()
})

// Computed styles
const containerStyle = computed(() => ({
  height: props.height,
}))

// Performance status color
const statusColor = computed(() => {
  switch (performanceStatus.value) {
    case 'good':
      return 'text-green-500'
    case 'warning':
      return 'text-yellow-500'
    case 'poor':
      return 'text-red-500'
    default:
      return 'text-gray-500'
  }
})

// Expose methods for parent component
defineExpose({
  scene,
  camera,
  renderer,
  controls,
  isInitialized,
  isAnimating,
  fps,
  currentLOD,
  startAnimation,
  stopAnimation,
  render,
  resetCamera,
  exportImage,
})
</script>

<template>
  <div class="three-canvas-wrapper relative" :style="containerStyle">
    <!-- Three.js container -->
    <div
      ref="container"
      class="three-container absolute inset-0 w-full h-full"
    />

    <!-- Performance overlay -->
    <div
      v-if="showPerformance && isInitialized"
      class="absolute top-2 left-2 bg-black/70 text-white text-xs font-mono px-2 py-1 rounded pointer-events-none"
    >
      <div class="flex items-center gap-2">
        <span :class="statusColor">{{ currentFPS }} FPS</span>
        <span class="text-gray-400">|</span>
        <span>{{ drawCalls }} calls</span>
        <span class="text-gray-400">|</span>
        <span>{{ (triangles / 1000).toFixed(1) }}k tris</span>
        <span class="text-gray-400">|</span>
        <span class="uppercase">{{ currentLOD }}</span>
      </div>
    </div>

    <!-- Loading state -->
    <div
      v-if="!isInitialized"
      class="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-900"
    >
      <div class="text-gray-500 dark:text-gray-400">
        Initializing 3D view...
      </div>
    </div>

    <!-- Slot for overlay UI (legends, controls, etc.) -->
    <slot name="overlay" />
  </div>
</template>

<style scoped>
.three-canvas-wrapper {
  overflow: hidden;
  border-radius: 0.5rem;
}

.three-container :deep(canvas) {
  display: block;
  width: 100%;
  height: 100%;
}
</style>
