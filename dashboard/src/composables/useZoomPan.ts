/**
 * Zoom and Pan Composable
 *
 * Provides zoom and pan functionality for visualization components.
 */

import { ref, computed, onMounted, onUnmounted, type Ref } from 'vue'

export interface ZoomPanOptions {
  minZoom?: number
  maxZoom?: number
  zoomStep?: number
  initialZoom?: number
  initialPan?: { x: number; y: number }
  enableWheel?: boolean
  enablePinch?: boolean
  enableDrag?: boolean
  boundaryPadding?: number
}

export interface ZoomPanState {
  zoom: number
  pan: { x: number; y: number }
  isDragging: boolean
}

export function useZoomPan(containerRef: Ref<HTMLElement | null>, options: ZoomPanOptions = {}) {
  const {
    minZoom = 0.1,
    maxZoom = 5,
    zoomStep = 0.1,
    initialZoom = 1,
    initialPan = { x: 0, y: 0 },
    enableWheel = true,
    enablePinch = true,
    enableDrag = true,
    boundaryPadding = 50,
  } = options

  // State
  const zoom = ref(initialZoom)
  const pan = ref({ ...initialPan })
  const isDragging = ref(false)

  // Internal state
  let dragStart = { x: 0, y: 0 }
  let panStart = { x: 0, y: 0 }
  let lastPinchDistance = 0

  // Computed
  const transform = computed(() => {
    return `translate(${pan.value.x}px, ${pan.value.y}px) scale(${zoom.value})`
  })

  const transformOrigin = computed(() => 'center center')

  const state = computed<ZoomPanState>(() => ({
    zoom: zoom.value,
    pan: pan.value,
    isDragging: isDragging.value,
  }))

  // Methods
  function setZoom(newZoom: number, center?: { x: number; y: number }): void {
    const clampedZoom = Math.max(minZoom, Math.min(maxZoom, newZoom))

    if (center && containerRef.value) {
      // Zoom towards point
      const scale = clampedZoom / zoom.value
      pan.value = {
        x: center.x - (center.x - pan.value.x) * scale,
        y: center.y - (center.y - pan.value.y) * scale,
      }
    }

    zoom.value = clampedZoom
  }

  function zoomIn(center?: { x: number; y: number }): void {
    setZoom(zoom.value + zoomStep, center)
  }

  function zoomOut(center?: { x: number; y: number }): void {
    setZoom(zoom.value - zoomStep, center)
  }

  function zoomToFit(contentSize?: { width: number; height: number }): void {
    if (!containerRef.value) return

    const container = containerRef.value.getBoundingClientRect()

    if (contentSize) {
      const scaleX = (container.width - boundaryPadding * 2) / contentSize.width
      const scaleY = (container.height - boundaryPadding * 2) / contentSize.height
      zoom.value = Math.max(minZoom, Math.min(maxZoom, Math.min(scaleX, scaleY)))
    } else {
      zoom.value = 1
    }

    pan.value = { x: 0, y: 0 }
  }

  function setPan(newPan: { x: number; y: number }): void {
    pan.value = { ...newPan }
  }

  function reset(): void {
    zoom.value = initialZoom
    pan.value = { ...initialPan }
  }

  // Event handlers
  function handleWheel(event: WheelEvent): void {
    if (!enableWheel) return
    event.preventDefault()

    const delta = -event.deltaY * 0.001
    const newZoom = zoom.value + delta

    setZoom(newZoom, { x: event.clientX, y: event.clientY })
  }

  function handleMouseDown(event: MouseEvent): void {
    if (!enableDrag || event.button !== 0) return

    isDragging.value = true
    dragStart = { x: event.clientX, y: event.clientY }
    panStart = { ...pan.value }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }

  function handleMouseMove(event: MouseEvent): void {
    if (!isDragging.value) return

    pan.value = {
      x: panStart.x + (event.clientX - dragStart.x),
      y: panStart.y + (event.clientY - dragStart.y),
    }
  }

  function handleMouseUp(): void {
    isDragging.value = false
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
  }

  function handleTouchStart(event: TouchEvent): void {
    if (event.touches.length === 2 && enablePinch) {
      // Pinch zoom start
      const touch1 = event.touches[0]
      const touch2 = event.touches[1]
      if (touch1 && touch2) {
        lastPinchDistance = Math.hypot(touch2.clientX - touch1.clientX, touch2.clientY - touch1.clientY)
      }
    } else if (event.touches.length === 1 && enableDrag) {
      // Pan start
      const touch = event.touches[0]
      if (touch) {
        isDragging.value = true
        dragStart = { x: touch.clientX, y: touch.clientY }
        panStart = { ...pan.value }
      }
    }
  }

  function handleTouchMove(event: TouchEvent): void {
    if (event.touches.length === 2 && enablePinch) {
      // Pinch zoom
      event.preventDefault()
      const touch1 = event.touches[0]
      const touch2 = event.touches[1]

      if (touch1 && touch2) {
        const distance = Math.hypot(touch2.clientX - touch1.clientX, touch2.clientY - touch1.clientY)

        const scale = distance / lastPinchDistance
        const center = {
          x: (touch1.clientX + touch2.clientX) / 2,
          y: (touch1.clientY + touch2.clientY) / 2,
        }

        setZoom(zoom.value * scale, center)
        lastPinchDistance = distance
      }
    } else if (event.touches.length === 1 && isDragging.value) {
      // Pan
      const touch = event.touches[0]
      if (touch) {
        pan.value = {
          x: panStart.x + (touch.clientX - dragStart.x),
          y: panStart.y + (touch.clientY - dragStart.y),
        }
      }
    }
  }

  function handleTouchEnd(): void {
    isDragging.value = false
    lastPinchDistance = 0
  }

  // Lifecycle
  onMounted(() => {
    const el = containerRef.value
    if (!el) return

    el.addEventListener('wheel', handleWheel, { passive: false })
    el.addEventListener('mousedown', handleMouseDown)
    el.addEventListener('touchstart', handleTouchStart, { passive: true })
    el.addEventListener('touchmove', handleTouchMove, { passive: false })
    el.addEventListener('touchend', handleTouchEnd)
  })

  onUnmounted(() => {
    const el = containerRef.value
    if (!el) return

    el.removeEventListener('wheel', handleWheel)
    el.removeEventListener('mousedown', handleMouseDown)
    el.removeEventListener('touchstart', handleTouchStart)
    el.removeEventListener('touchmove', handleTouchMove)
    el.removeEventListener('touchend', handleTouchEnd)
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
  })

  return {
    // State
    zoom,
    pan,
    isDragging,
    state,

    // Computed
    transform,
    transformOrigin,

    // Methods
    setZoom,
    zoomIn,
    zoomOut,
    zoomToFit,
    setPan,
    reset,
  }
}
