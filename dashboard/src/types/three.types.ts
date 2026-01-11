/**
 * Three.js visualization types for Alignment Observatory Dashboard
 */

import type * as THREE from 'three'
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

/** Visualization mode for toggling between 2D and 3D views */
export type VisualizationMode = '2d' | '3d'

/** 3D visualization sub-modes */
export type ThreeVisualizationMode = '3d-surface' | '3d-flow' | '3d-graph' | '3d-terrain'

/** Color scale names for 3D visualizations */
export type ThreeColorScale = 'viridis' | 'plasma' | 'inferno' | 'coolwarm'

/** Level of detail for performance optimization */
export type LODLevel = 'high' | 'medium' | 'low'

/** Three.js scene context */
export interface ThreeSceneContext {
  /** The Three.js scene */
  scene: THREE.Scene
  /** Camera (perspective or orthographic) */
  camera: THREE.PerspectiveCamera | THREE.OrthographicCamera
  /** WebGL renderer */
  renderer: THREE.WebGLRenderer
  /** Orbit controls for camera manipulation */
  controls: OrbitControls | null
  /** Animation clock */
  clock: THREE.Clock
}

/** Options for creating a Three.js scene */
export interface ThreeSceneOptions {
  /** Container element ref */
  container: HTMLElement | null
  /** Enable antialiasing (default: true) */
  antialias?: boolean
  /** Transparent background (default: false) */
  alpha?: boolean
  /** Pixel ratio (default: device pixel ratio, capped at 2) */
  pixelRatio?: number
  /** Enable orbit controls (default: true) */
  enableControls?: boolean
  /** Background color (hex) */
  backgroundColor?: number
  /** Enable shadows (default: false) */
  enableShadows?: boolean
}

/** Camera preset positions */
export interface CameraPreset {
  position: [number, number, number]
  target: [number, number, number]
  name: string
}

/** Standard camera presets for visualizations */
export const CAMERA_PRESETS: Record<string, CameraPreset> = {
  top: { position: [0, 50, 0], target: [0, 0, 0], name: 'Top View' },
  front: { position: [0, 0, 50], target: [0, 0, 0], name: 'Front View' },
  side: { position: [50, 0, 0], target: [0, 0, 0], name: 'Side View' },
  isometric: { position: [30, 30, 30], target: [0, 0, 0], name: 'Isometric' },
}

/** LOD settings based on performance */
export interface LODSettings {
  /** Maximum number of instanced objects */
  maxInstances: number
  /** Particle/cell size multiplier */
  sizeMultiplier: number
  /** Enable shadows */
  shadows: boolean
  /** Antialiasing quality */
  antialias: boolean
}

/** LOD configuration by level */
export const LOD_CONFIG: Record<LODLevel, LODSettings> = {
  high: { maxInstances: 100000, sizeMultiplier: 1.0, shadows: true, antialias: true },
  medium: { maxInstances: 50000, sizeMultiplier: 0.8, shadows: false, antialias: true },
  low: { maxInstances: 20000, sizeMultiplier: 0.5, shadows: false, antialias: false },
}

/** Performance metrics for WebGL rendering */
export interface WebGLPerformanceMetrics {
  /** Frames per second */
  fps: number
  /** Frame time in milliseconds */
  frameTime: number
  /** GPU memory usage (if available) */
  gpuMemory?: number
  /** Draw calls per frame */
  drawCalls: number
  /** Triangle count */
  triangles: number
  /** Current LOD level */
  lodLevel: LODLevel
}

/** Props for AttentionFlow3D component */
export interface AttentionFlow3DProps {
  /** Attention data per layer */
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
  colorScale?: ThreeColorScale
  /** Enable animation */
  animate?: boolean
  /** Container height */
  height?: string
}

/** Emits for AttentionFlow3D component */
export interface AttentionFlow3DEmits {
  /** Emitted when a cell is hovered */
  (e: 'cellHover', data: { layer: number; head: number; query: number; key: number; value: number } | null): void
  /** Emitted when a cell is clicked */
  (e: 'cellClick', data: { layer: number; head: number; query: number; key: number; value: number }): void
  /** Emitted when the camera view changes */
  (e: 'viewChange', data: { position: [number, number, number]; target: [number, number, number] }): void
}

/** Props for Circuit3D component */
export interface Circuit3DProps {
  /** Circuit data with nodes and edges */
  nodes: Array<{
    id: string
    layer: number
    component: string
    head?: number
    importance?: number
    label?: string
  }>
  edges: Array<{
    source: string
    target: string
    importance: number
  }>
  /** Edge importance threshold (0-1) */
  threshold?: number
  /** Layout algorithm */
  layout?: '3d-force' | '3d-hierarchical' | '3d-layered'
  /** Show node labels */
  showLabels?: boolean
  /** Enable flow animation */
  animate?: boolean
  /** Highlighted node IDs */
  highlightedNodes?: string[]
  /** Container height */
  height?: string
}

/** Props for ActivationLandscape3D component */
export interface ActivationLandscape3DProps {
  /** Activation matrix (tokens x dimensions) */
  matrix: Float32Array
  /** Matrix shape [tokens, dimensions] */
  shape: [number, number]
  /** Token labels */
  tokens: string[]
  /** Current layer being displayed */
  layer: number
  /** Height scale multiplier */
  heightScale?: number
  /** Color scale name */
  colorScale?: ThreeColorScale
  /** Enable diverging colors (centered at 0) */
  diverging?: boolean
  /** Container height */
  height?: string
}

/** Node mesh with custom userData */
export interface NodeMeshUserData {
  nodeId: string
  type: string
  layer: number
  head?: number
  importance?: number
}

/** Attention cell instance data */
export interface AttentionCellInstance {
  layer: number
  head: number
  query: number
  key: number
  value: number
  matrixIndex: number
}
