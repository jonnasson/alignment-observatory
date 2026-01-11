/**
 * Composables barrel export
 */

export { useTheme } from './useTheme'

export { useColorScale } from './useColorScale'
export type { ColorScaleName, ColorScaleOptions } from './useColorScale'

export { useZoomPan } from './useZoomPan'
export type { ZoomPanOptions, ZoomPanState } from './useZoomPan'

export { useTensorData, reshapeTo2D, flatten2D } from './useTensorData'
export type { TensorView } from './useTensorData'

export { useKeyboardShortcuts, formatShortcut, getVisualizationShortcuts } from './useKeyboardShortcuts'
export type { Shortcut, ShortcutGroup } from './useKeyboardShortcuts'

export { useAttentionViz, extractHeadAttention, computeAttentionStats, classifyAttentionPattern } from './useAttentionViz'
export type { AttentionCell, AttentionRow, AttentionMatrixData, HeadInfo, UseAttentionVizOptions } from './useAttentionViz'

export {
  useActivationViz,
  computeArrayStats,
  computeTokenNorms,
  computeTokenStats,
  computeDimensionStats,
  extractActivationMatrix,
  getTopDimensionsByVariance,
  getTopDimensionsByMaxAbs,
  normalizeMatrix,
  normalizeMatrixDiverging,
} from './useActivationViz'
export type {
  ActivationStats,
  TokenStats,
  DimensionStats,
  ActivationMatrixData,
  UseActivationVizOptions,
} from './useActivationViz'

export {
  useCircuitGraph,
  componentColors,
  componentLabels,
  getNodeLabel,
  getEdgeWidth,
  getEdgeOpacity,
  generateDotGraph,
  computeGraphStats,
} from './useCircuitGraph'
export type { GraphStats, UseCircuitGraphOptions } from './useCircuitGraph'

export {
  useSAE,
  getFeatureColor,
  generateCoactivationMatrix,
  formatFeatureIdx,
  formatActivation,
} from './useSAE'
export type { UseSAEOptions, SAEStats } from './useSAE'

export {
  useIOI,
  ioiComponentColors,
  ioiComponentLabels,
  knownGPT2Heads,
  parseIOISentence,
  generateDemoIOICircuit,
} from './useIOI'
export type { UseIOIOptions } from './useIOI'
