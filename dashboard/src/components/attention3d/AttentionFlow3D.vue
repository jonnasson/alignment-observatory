<script setup lang="ts">
/**
 * AttentionFlow3D - 3D visualization of attention patterns across all layers
 *
 * Renders attention patterns as a 3D cube where:
 * - X-axis: Token position (query)
 * - Y-axis: Token position (key)
 * - Z-axis: Layer depth
 * - Color/Opacity: Attention weight
 *
 * Uses THREE.InstancedMesh for efficient rendering of many attention cells.
 */

import { ref, computed, watch, onMounted, onUnmounted, shallowRef } from 'vue'
import * as THREE from 'three'
import { useTheme } from '@/composables/useTheme'
import { useThreeScene } from '@/composables/useThreeScene'
import { useWebGLPerformance } from '@/composables/useWebGLPerformance'
import { useColorScale, type ColorScaleName } from '@/composables/useColorScale'
import type { AttentionCellInstance } from '@/types/three.types'

interface Props {
  /** Attention data per layer: Map<layer, Float32Array of shape [heads, seq_q, seq_k]> */
  attentionData: Map<number, Float32Array>
  /** Token labels */
  tokens: string[]
  /** Number of layers */
  numLayers: number
  /** Number of attention heads per layer */
  numHeads: number
  /** Sequence length */
  seqLen: number
  /** Minimum attention threshold to display (0-1) */
  threshold?: number
  /** Selected heads to highlight (empty = all) */
  selectedHeads?: number[]
  /** Selected layers to display (empty = all) */
  selectedLayers?: number[]
  /** Color scale name */
  colorScale?: ColorScaleName
  /** Enable animation */
  animate?: boolean
  /** Container height */
  height?: string
  /** Show performance stats */
  showPerformance?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  threshold: 0.1,
  selectedHeads: () => [],
  selectedLayers: () => [],
  colorScale: 'viridis',
  animate: false,
  height: '500px',
  showPerformance: false,
})

const emit = defineEmits<{
  /** Emitted when a cell is hovered */
  (e: 'cellHover', data: AttentionCellInstance | null): void
  /** Emitted when a cell is clicked */
  (e: 'cellClick', data: AttentionCellInstance): void
}>()

// Container ref
const container = ref<HTMLElement | null>(null)

// Dark mode
const { isDark } = useTheme()

// Color scale
const { getColor } = useColorScale({ scaleName: props.colorScale })

// Three.js scene
const {
  scene,
  camera,
  controls,
  isInitialized,
  init,
  startAnimation,
  resetCamera,
  getRendererInfo,
  dispose,
} = useThreeScene({
  container,
  isDark,
  antialias: true,
  enableShadows: false,
})

// Performance monitoring
const {
  currentFPS,
  drawCalls,
  triangles,
  performanceStatus,
  recordFrame,
} = useWebGLPerformance({ targetFPS: 30 })

// Instanced mesh for attention cells
const instancedMesh = shallowRef<THREE.InstancedMesh | null>(null)

// Raycaster for mouse picking
const raycaster = new THREE.Raycaster()
const mouse = new THREE.Vector2()

// Hovered cell
const hoveredIndex = ref<number | null>(null)

// Cell instance data (for raycasting lookup)
const cellInstances = ref<AttentionCellInstance[]>([])

// Computed: layers to display
const activeLayers = computed(() => {
  if (props.selectedLayers.length === 0) {
    return Array.from({ length: props.numLayers }, (_, i) => i)
  }
  return props.selectedLayers
})

// Computed: heads to highlight
const activeHeads = computed(() => {
  if (props.selectedHeads.length === 0) {
    return Array.from({ length: props.numHeads }, (_, i) => i)
  }
  return props.selectedHeads
})

// Layout constants
const CELL_SIZE = 0.8
const CELL_GAP = 0.2
const LAYER_SPACING = 3

/**
 * Create/update the instanced mesh with attention data
 */
function updateVisualization() {
  if (!scene.value || !isInitialized.value) return

  // Remove existing mesh
  if (instancedMesh.value) {
    scene.value.remove(instancedMesh.value)
    instancedMesh.value.geometry.dispose()
    if (instancedMesh.value.material instanceof THREE.Material) {
      instancedMesh.value.material.dispose()
    }
    instancedMesh.value = null
  }

  // Calculate total cells needed
  const layers = activeLayers.value
  const totalCells = layers.length * props.seqLen * props.seqLen

  if (totalCells === 0) return

  // Create geometry for each cell
  const cellGeometry = new THREE.BoxGeometry(CELL_SIZE, CELL_SIZE, CELL_SIZE * 0.3)

  // Create material with vertex colors
  const cellMaterial = new THREE.MeshPhongMaterial({
    vertexColors: false,
    transparent: true,
    opacity: 0.9,
  })

  // Create instanced mesh
  const mesh = new THREE.InstancedMesh(cellGeometry, cellMaterial, totalCells)
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)

  // Create color buffer attribute
  const colors = new Float32Array(totalCells * 3)

  // Track instances for raycasting
  const instances: AttentionCellInstance[] = []

  // Temporary objects for matrix computation
  const matrix = new THREE.Matrix4()
  const position = new THREE.Vector3()
  const quaternion = new THREE.Quaternion()
  const scale = new THREE.Vector3(1, 1, 1)

  // Center offset
  const offsetX = (props.seqLen * (CELL_SIZE + CELL_GAP)) / 2
  const offsetY = (props.seqLen * (CELL_SIZE + CELL_GAP)) / 2
  const offsetZ = (layers.length * LAYER_SPACING) / 2

  let instanceIndex = 0
  let visibleCount = 0

  // Iterate through layers and build instances
  for (const layer of layers) {
    const layerData = props.attentionData.get(layer)
    if (!layerData) continue

    const layerZ = (layers.indexOf(layer)) * LAYER_SPACING - offsetZ

    for (let q = 0; q < props.seqLen; q++) {
      for (let k = 0; k < props.seqLen; k++) {
        // Aggregate attention across heads (mean)
        let totalAttention = 0
        let headCount = 0

        for (let h = 0; h < props.numHeads; h++) {
          // Check if this head is active
          if (activeHeads.value.length > 0 && !activeHeads.value.includes(h)) {
            continue
          }

          const idx = h * props.seqLen * props.seqLen + q * props.seqLen + k
          const attention = layerData[idx] ?? 0
          totalAttention += attention
          headCount++
        }

        const avgAttention = headCount > 0 ? totalAttention / headCount : 0

        // Skip if below threshold
        if (avgAttention < props.threshold) {
          // Still need to set matrix for hidden instance
          scale.set(0, 0, 0)
          position.set(0, 0, 0)
          matrix.compose(position, quaternion, scale)
          mesh.setMatrixAt(instanceIndex, matrix)
          colors[instanceIndex * 3] = 0
          colors[instanceIndex * 3 + 1] = 0
          colors[instanceIndex * 3 + 2] = 0
          instanceIndex++
          continue
        }

        visibleCount++

        // Calculate position
        const x = q * (CELL_SIZE + CELL_GAP) - offsetX
        const y = k * (CELL_SIZE + CELL_GAP) - offsetY
        const z = layerZ

        // Set position and scale (scale by attention for height effect)
        position.set(x, y, z)
        scale.set(1, 1, 1 + avgAttention * 2)
        matrix.compose(position, quaternion, scale)
        mesh.setMatrixAt(instanceIndex, matrix)

        // Get color from scale
        const rgb = getColor(avgAttention, 'array') as number[]
        colors[instanceIndex * 3] = (rgb[0] ?? 0) / 255
        colors[instanceIndex * 3 + 1] = (rgb[1] ?? 0) / 255
        colors[instanceIndex * 3 + 2] = (rgb[2] ?? 0) / 255

        // Track instance for raycasting
        instances.push({
          layer,
          head: -1, // Aggregated
          query: q,
          key: k,
          value: avgAttention,
          matrixIndex: instanceIndex,
        })

        instanceIndex++
      }
    }
  }

  // Apply colors via custom attribute
  mesh.instanceMatrix.needsUpdate = true

  // Store references
  instancedMesh.value = mesh
  cellInstances.value = instances
  scene.value.add(mesh)

  // Add axis helpers
  addAxisHelpers()
}

/**
 * Add axis labels and guides
 */
function addAxisHelpers() {
  if (!scene.value) return

  // Remove existing helpers
  scene.value.children
    .filter((c) => c.name.startsWith('axis-'))
    .forEach((c) => scene.value?.remove(c))

  const layers = activeLayers.value
  const offsetX = (props.seqLen * (CELL_SIZE + CELL_GAP)) / 2
  const offsetY = (props.seqLen * (CELL_SIZE + CELL_GAP)) / 2
  const offsetZ = (layers.length * LAYER_SPACING) / 2

  // Query axis (X) label
  const queryLabel = createTextSprite('Query →', 0xffffff)
  queryLabel.position.set(0, -offsetY - 3, -offsetZ)
  queryLabel.name = 'axis-query'
  scene.value.add(queryLabel)

  // Key axis (Y) label
  const keyLabel = createTextSprite('Key →', 0xffffff)
  keyLabel.position.set(-offsetX - 3, 0, -offsetZ)
  keyLabel.name = 'axis-key'
  scene.value.add(keyLabel)

  // Layer axis (Z) label
  const layerLabel = createTextSprite('Layer →', 0xffffff)
  layerLabel.position.set(-offsetX - 3, -offsetY - 3, 0)
  layerLabel.name = 'axis-layer'
  scene.value.add(layerLabel)

  // Add layer markers
  for (let i = 0; i < layers.length; i++) {
    const z = i * LAYER_SPACING - offsetZ
    const marker = createTextSprite(`L${layers[i]}`, 0x888888)
    marker.position.set(-offsetX - 2, -offsetY - 2, z)
    marker.scale.set(0.5, 0.5, 0.5)
    marker.name = `axis-layer-${i}`
    scene.value.add(marker)
  }
}

/**
 * Create a text sprite for labels
 */
function createTextSprite(text: string, color: number): THREE.Sprite {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')!
  canvas.width = 256
  canvas.height = 64

  ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`
  ctx.font = 'bold 32px Arial'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, 128, 32)

  const texture = new THREE.CanvasTexture(canvas)
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true })
  const sprite = new THREE.Sprite(material)
  sprite.scale.set(4, 1, 1)

  return sprite
}

/**
 * Handle mouse move for hover detection
 */
function onMouseMove(event: MouseEvent) {
  if (!container.value || !instancedMesh.value || !camera.value) return

  const rect = container.value.getBoundingClientRect()
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  raycaster.setFromCamera(mouse, camera.value)
  const intersects = raycaster.intersectObject(instancedMesh.value)

  const firstIntersect = intersects[0]
  if (firstIntersect && firstIntersect.instanceId !== undefined) {
    const idx = firstIntersect.instanceId
    if (idx !== hoveredIndex.value) {
      hoveredIndex.value = idx
      const instance = cellInstances.value.find((c) => c.matrixIndex === idx)
      emit('cellHover', instance ?? null)
    }
  } else {
    if (hoveredIndex.value !== null) {
      hoveredIndex.value = null
      emit('cellHover', null)
    }
  }
}

/**
 * Handle click for cell selection
 */
function onClick(event: MouseEvent) {
  if (!container.value || !instancedMesh.value || !camera.value) return

  const rect = container.value.getBoundingClientRect()
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  raycaster.setFromCamera(mouse, camera.value)
  const intersects = raycaster.intersectObject(instancedMesh.value)

  const firstIntersect = intersects[0]
  if (firstIntersect && firstIntersect.instanceId !== undefined) {
    const instance = cellInstances.value.find((c) => c.matrixIndex === firstIntersect.instanceId)
    if (instance) {
      emit('cellClick', instance)
    }
  }
}

// Animation frame callback
function onFrame(delta: number) {
  if (props.animate && instancedMesh.value) {
    // Subtle rotation animation
    instancedMesh.value.rotation.y += delta * 0.1
  }

  const info = getRendererInfo()
  if (info) {
    recordFrame(info)
  }
}

// Initialize on mount
onMounted(() => {
  if (init()) {
    updateVisualization()

    // Set initial camera position
    if (camera.value && controls.value) {
      camera.value.position.set(30, 30, 30)
      controls.value.target.set(0, 0, 0)
      controls.value.update()
    }

    startAnimation(onFrame)

    // Add event listeners
    container.value?.addEventListener('mousemove', onMouseMove)
    container.value?.addEventListener('click', onClick)
  }
})

// Cleanup on unmount
onUnmounted(() => {
  container.value?.removeEventListener('mousemove', onMouseMove)
  container.value?.removeEventListener('click', onClick)
  dispose()
})

// Watch for data changes
watch(
  () => [props.attentionData, props.threshold, props.selectedHeads, props.selectedLayers, props.colorScale],
  () => {
    if (isInitialized.value) {
      updateVisualization()
    }
  },
  { deep: true }
)

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

// Expose methods
defineExpose({
  resetCamera,
  updateVisualization,
})
</script>

<template>
  <div class="attention-flow-3d relative" :style="{ height }">
    <!-- Three.js container -->
    <div
      ref="container"
      class="absolute inset-0 w-full h-full rounded-lg overflow-hidden"
    />

    <!-- Performance overlay -->
    <div
      v-if="showPerformance && isInitialized"
      class="absolute top-2 left-2 bg-black/70 text-white text-xs font-mono px-2 py-1 rounded pointer-events-none"
    >
      <div class="flex items-center gap-2">
        <span :class="statusColor">{{ currentFPS }} FPS</span>
        <span class="text-gray-400">|</span>
        <span>{{ drawCalls }} draws</span>
        <span class="text-gray-400">|</span>
        <span>{{ (triangles / 1000).toFixed(1) }}k tris</span>
      </div>
    </div>

    <!-- Controls hint -->
    <div
      v-if="isInitialized"
      class="absolute bottom-2 right-2 text-xs text-gray-400 bg-black/50 px-2 py-1 rounded"
    >
      Drag to rotate • Scroll to zoom
    </div>

    <!-- Loading state -->
    <div
      v-if="!isInitialized"
      class="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-900 rounded-lg"
    >
      <div class="flex flex-col items-center gap-2">
        <div class="animate-spin w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full" />
        <span class="text-gray-500 dark:text-gray-400 text-sm">Initializing 3D view...</span>
      </div>
    </div>

    <!-- Slot for additional overlay content -->
    <slot />
  </div>
</template>

<style scoped>
.attention-flow-3d {
  min-height: 300px;
}
</style>
