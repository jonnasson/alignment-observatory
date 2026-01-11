/**
 * WebGL Performance Monitoring Composable
 *
 * Monitors rendering performance and automatically adjusts LOD
 * (Level of Detail) settings for optimal frame rates.
 */

import { ref, computed } from 'vue'
import type { LODLevel, LODSettings, WebGLPerformanceMetrics } from '@/types/three.types'

export interface UseWebGLPerformanceOptions {
  /** Target frames per second (default: 30) */
  targetFPS?: number
  /** Number of frames to average for FPS calculation (default: 30) */
  sampleSize?: number
  /** Enable automatic LOD adjustment (default: true) */
  autoAdjustLOD?: boolean
  /** Minimum FPS ratio to trigger LOD decrease (default: 0.7) */
  lodDecreaseThreshold?: number
  /** Maximum FPS ratio to allow LOD increase (default: 0.95) */
  lodIncreaseThreshold?: number
  /** Cooldown between LOD changes in ms (default: 2000) */
  lodChangeCooldown?: number
}

/** LOD configuration by level */
const LOD_SETTINGS: Record<LODLevel, LODSettings> = {
  high: { maxInstances: 100000, sizeMultiplier: 1.0, shadows: true, antialias: true },
  medium: { maxInstances: 50000, sizeMultiplier: 0.8, shadows: false, antialias: true },
  low: { maxInstances: 20000, sizeMultiplier: 0.5, shadows: false, antialias: false },
}

export function useWebGLPerformance(options: UseWebGLPerformanceOptions = {}) {
  const {
    targetFPS = 30,
    sampleSize = 30,
    autoAdjustLOD = true,
    lodDecreaseThreshold = 0.7,
    lodIncreaseThreshold = 0.95,
    lodChangeCooldown = 2000,
  } = options

  // Performance tracking
  const fpsHistory = ref<number[]>([])
  const currentFPS = ref(0)
  const averageFPS = ref(0)
  const frameTime = ref(0)
  const drawCalls = ref(0)
  const triangles = ref(0)

  // LOD state
  const currentLOD = ref<LODLevel>('high')
  const lastLODChange = ref(0)

  // Frame timing
  let lastFrameTime = performance.now()

  /**
   * Record a frame and update metrics
   */
  function recordFrame(rendererInfo?: { drawCalls: number; triangles: number }) {
    const now = performance.now()
    const delta = now - lastFrameTime
    lastFrameTime = now

    // Calculate instantaneous FPS
    const instantFPS = delta > 0 ? 1000 / delta : 0
    currentFPS.value = Math.round(instantFPS)
    frameTime.value = delta

    // Update FPS history
    fpsHistory.value.push(instantFPS)
    if (fpsHistory.value.length > sampleSize) {
      fpsHistory.value.shift()
    }

    // Calculate average FPS
    if (fpsHistory.value.length > 0) {
      const sum = fpsHistory.value.reduce((a, b) => a + b, 0)
      averageFPS.value = Math.round(sum / fpsHistory.value.length)
    }

    // Update renderer stats
    if (rendererInfo) {
      drawCalls.value = rendererInfo.drawCalls
      triangles.value = rendererInfo.triangles
    }

    // Auto-adjust LOD if enabled
    if (autoAdjustLOD && fpsHistory.value.length >= sampleSize) {
      adjustLOD()
    }
  }

  /**
   * Adjust LOD based on current performance
   */
  function adjustLOD() {
    const now = performance.now()

    // Check cooldown
    if (now - lastLODChange.value < lodChangeCooldown) return

    const fpsRatio = averageFPS.value / targetFPS

    // Decrease LOD if performance is poor
    if (fpsRatio < lodDecreaseThreshold) {
      if (currentLOD.value === 'high') {
        currentLOD.value = 'medium'
        lastLODChange.value = now
      } else if (currentLOD.value === 'medium') {
        currentLOD.value = 'low'
        lastLODChange.value = now
      }
    }
    // Increase LOD if performance is good
    else if (fpsRatio > lodIncreaseThreshold) {
      if (currentLOD.value === 'low') {
        currentLOD.value = 'medium'
        lastLODChange.value = now
      } else if (currentLOD.value === 'medium') {
        currentLOD.value = 'high'
        lastLODChange.value = now
      }
    }
  }

  /**
   * Manually set LOD level
   */
  function setLOD(level: LODLevel) {
    currentLOD.value = level
    lastLODChange.value = performance.now()
  }

  /**
   * Reset performance tracking
   */
  function reset() {
    fpsHistory.value = []
    currentFPS.value = 0
    averageFPS.value = 0
    frameTime.value = 0
    drawCalls.value = 0
    triangles.value = 0
    currentLOD.value = 'high'
    lastFrameTime = performance.now()
  }

  // Computed LOD settings
  const lodSettings = computed((): LODSettings => LOD_SETTINGS[currentLOD.value])

  // Performance status
  const performanceStatus = computed((): 'good' | 'warning' | 'poor' => {
    const ratio = averageFPS.value / targetFPS
    if (ratio >= lodIncreaseThreshold) return 'good'
    if (ratio >= lodDecreaseThreshold) return 'warning'
    return 'poor'
  })

  // Full metrics object
  const metrics = computed((): WebGLPerformanceMetrics => ({
    fps: currentFPS.value,
    frameTime: frameTime.value,
    drawCalls: drawCalls.value,
    triangles: triangles.value,
    lodLevel: currentLOD.value,
  }))

  return {
    // Metrics
    currentFPS,
    averageFPS,
    frameTime,
    drawCalls,
    triangles,
    metrics,

    // LOD
    currentLOD,
    lodSettings,
    setLOD,

    // Status
    performanceStatus,
    targetFPS,

    // Methods
    recordFrame,
    reset,
  }
}
