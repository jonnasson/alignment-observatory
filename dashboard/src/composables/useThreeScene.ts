/**
 * Three.js Scene Composable
 *
 * Provides shared Three.js infrastructure for 3D visualizations.
 * Handles scene setup, camera, renderer, controls, and cleanup.
 */

import { ref, shallowRef, onUnmounted, watch, type Ref } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { CameraPreset } from '@/types/three.types'

export interface UseThreeSceneOptions {
  /** Container element ref */
  container: Ref<HTMLElement | null>
  /** Enable antialiasing (default: true) */
  antialias?: boolean
  /** Transparent background (default: false) */
  alpha?: boolean
  /** Pixel ratio (default: device pixel ratio, capped at 2) */
  pixelRatio?: number
  /** Enable orbit controls (default: true) */
  enableControls?: boolean
  /** Background color for light mode (hex) */
  backgroundColorLight?: number
  /** Background color for dark mode (hex) */
  backgroundColorDark?: number
  /** Enable shadows (default: false) */
  enableShadows?: boolean
  /** Dark mode ref */
  isDark?: Ref<boolean>
}

export function useThreeScene(options: UseThreeSceneOptions) {
  const {
    container,
    antialias = true,
    alpha = false,
    pixelRatio = Math.min(window.devicePixelRatio, 2),
    enableControls = true,
    backgroundColorLight = 0xfafafa,
    backgroundColorDark = 0x0a0a0f,
    enableShadows = false,
    isDark,
  } = options

  // Core Three.js objects (using shallowRef for performance)
  const scene = shallowRef<THREE.Scene | null>(null)
  const camera = shallowRef<THREE.PerspectiveCamera | null>(null)
  const renderer = shallowRef<THREE.WebGLRenderer | null>(null)
  const controls = shallowRef<OrbitControls | null>(null)
  const clock = shallowRef<THREE.Clock | null>(null)

  // State
  const isInitialized = ref(false)
  const isAnimating = ref(false)
  const fps = ref(0)

  // Internal
  let animationFrameId: number | null = null
  let resizeObserver: ResizeObserver | null = null
  let frameCount = 0
  let lastFpsUpdate = 0

  /**
   * Initialize the Three.js scene
   */
  function init(): boolean {
    if (!container.value || isInitialized.value) return false

    const width = container.value.clientWidth
    const height = container.value.clientHeight

    // Create scene
    scene.value = new THREE.Scene()
    const bgColor = isDark?.value ? backgroundColorDark : backgroundColorLight
    scene.value.background = new THREE.Color(bgColor)

    // Create camera
    camera.value = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000)
    camera.value.position.set(30, 30, 30)
    camera.value.lookAt(0, 0, 0)

    // Create renderer
    renderer.value = new THREE.WebGLRenderer({
      antialias,
      alpha,
      powerPreference: 'high-performance',
    })
    renderer.value.setPixelRatio(pixelRatio)
    renderer.value.setSize(width, height)

    if (enableShadows) {
      renderer.value.shadowMap.enabled = true
      renderer.value.shadowMap.type = THREE.PCFSoftShadowMap
    }

    container.value.appendChild(renderer.value.domElement)

    // Create controls
    if (enableControls) {
      controls.value = new OrbitControls(camera.value, renderer.value.domElement)
      controls.value.enableDamping = true
      controls.value.dampingFactor = 0.05
      controls.value.minDistance = 5
      controls.value.maxDistance = 200
      controls.value.maxPolarAngle = Math.PI * 0.9
    }

    // Create clock
    clock.value = new THREE.Clock()

    // Setup lights
    setupLights()

    // Setup resize observer
    resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(container.value)

    isInitialized.value = true
    return true
  }

  /**
   * Setup default lighting
   */
  function setupLights() {
    if (!scene.value) return

    // Ambient light for overall illumination
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.value.add(ambientLight)

    // Directional light for shadows and depth
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(10, 20, 10)

    if (enableShadows) {
      directionalLight.castShadow = true
      directionalLight.shadow.mapSize.width = 2048
      directionalLight.shadow.mapSize.height = 2048
      directionalLight.shadow.camera.near = 0.5
      directionalLight.shadow.camera.far = 100
    }

    scene.value.add(directionalLight)

    // Secondary light from opposite direction
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3)
    fillLight.position.set(-10, 5, -10)
    scene.value.add(fillLight)
  }

  /**
   * Handle container resize
   */
  function handleResize() {
    if (!container.value || !camera.value || !renderer.value) return

    const width = container.value.clientWidth
    const height = container.value.clientHeight

    camera.value.aspect = width / height
    camera.value.updateProjectionMatrix()

    renderer.value.setSize(width, height)
  }

  /**
   * Start animation loop
   */
  function startAnimation(onFrame?: (delta: number) => void) {
    if (isAnimating.value) return

    isAnimating.value = true
    lastFpsUpdate = performance.now()

    function animate() {
      if (!isAnimating.value) return

      animationFrameId = requestAnimationFrame(animate)

      const delta = clock.value?.getDelta() ?? 0

      // Update controls
      controls.value?.update()

      // Call custom frame callback
      onFrame?.(delta)

      // Render scene
      if (renderer.value && scene.value && camera.value) {
        renderer.value.render(scene.value, camera.value)
      }

      // Update FPS
      frameCount++
      const now = performance.now()
      if (now - lastFpsUpdate >= 1000) {
        fps.value = Math.round(frameCount * 1000 / (now - lastFpsUpdate))
        frameCount = 0
        lastFpsUpdate = now
      }
    }

    animate()
  }

  /**
   * Stop animation loop
   */
  function stopAnimation() {
    isAnimating.value = false
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId)
      animationFrameId = null
    }
  }

  /**
   * Render a single frame
   */
  function render() {
    if (renderer.value && scene.value && camera.value) {
      renderer.value.render(scene.value, camera.value)
    }
  }

  /**
   * Set camera to a preset position
   */
  function setCameraPreset(preset: CameraPreset, animate = true) {
    if (!camera.value || !controls.value) return

    const { position, target } = preset

    if (animate) {
      // Animate camera movement (simple linear interpolation)
      const startPos = camera.value.position.clone()
      const startTarget = controls.value.target.clone()
      const endPos = new THREE.Vector3(...position)
      const endTarget = new THREE.Vector3(...target)
      const duration = 500
      const startTime = performance.now()

      function animateCamera() {
        const elapsed = performance.now() - startTime
        const t = Math.min(elapsed / duration, 1)
        const eased = 1 - Math.pow(1 - t, 3) // Ease out cubic

        camera.value?.position.lerpVectors(startPos, endPos, eased)
        controls.value?.target.lerpVectors(startTarget, endTarget, eased)
        controls.value?.update()

        if (t < 1) {
          requestAnimationFrame(animateCamera)
        }
      }

      animateCamera()
    } else {
      camera.value.position.set(...position)
      controls.value.target.set(...target)
      controls.value.update()
    }
  }

  /**
   * Reset camera to default position
   */
  function resetCamera() {
    setCameraPreset({ position: [30, 30, 30], target: [0, 0, 0], name: 'Default' })
  }

  /**
   * Fit camera to bounding box
   */
  function fitToBox(box: THREE.Box3, padding = 1.2) {
    if (!camera.value || !controls.value) return

    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    const maxDim = Math.max(size.x, size.y, size.z) * padding

    const fov = camera.value.fov * (Math.PI / 180)
    const distance = maxDim / (2 * Math.tan(fov / 2))

    camera.value.position.set(
      center.x + distance * 0.5,
      center.y + distance * 0.5,
      center.z + distance
    )

    controls.value.target.copy(center)
    controls.value.update()
  }

  /**
   * Export canvas as image
   */
  function exportImage(format: 'png' | 'jpeg' = 'png'): string | null {
    if (!renderer.value) return null

    render()
    return renderer.value.domElement.toDataURL(`image/${format}`)
  }

  /**
   * Get WebGL renderer info
   */
  function getRendererInfo(): { drawCalls: number; triangles: number } | null {
    if (!renderer.value) return null

    const info = renderer.value.info
    return {
      drawCalls: info.render.calls,
      triangles: info.render.triangles,
    }
  }

  /**
   * Dispose all resources
   */
  function dispose() {
    stopAnimation()

    // Remove resize observer
    if (resizeObserver) {
      resizeObserver.disconnect()
      resizeObserver = null
    }

    // Dispose controls
    controls.value?.dispose()
    controls.value = null

    // Dispose renderer
    if (renderer.value) {
      renderer.value.dispose()
      renderer.value.domElement.remove()
      renderer.value = null
    }

    // Clear scene
    if (scene.value) {
      scene.value.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry?.dispose()
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose())
          } else {
            obj.material?.dispose()
          }
        }
      })
      scene.value.clear()
      scene.value = null
    }

    camera.value = null
    clock.value = null
    isInitialized.value = false
  }

  // Watch for dark mode changes
  if (isDark) {
    watch(isDark, (dark) => {
      if (scene.value) {
        scene.value.background = new THREE.Color(dark ? backgroundColorDark : backgroundColorLight)
      }
    })
  }

  // Auto-cleanup on unmount
  onUnmounted(() => {
    dispose()
  })

  return {
    // Refs
    scene,
    camera,
    renderer,
    controls,
    clock,
    isInitialized,
    isAnimating,
    fps,

    // Methods
    init,
    startAnimation,
    stopAnimation,
    render,
    setCameraPreset,
    resetCamera,
    fitToBox,
    exportImage,
    getRendererInfo,
    handleResize,
    dispose,
  }
}
