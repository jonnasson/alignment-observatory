/**
 * Type definitions for Alignment Observatory Dashboard
 *
 * Barrel export for all type modules
 */

// Tensor types
export type {
  TensorData,
  TensorMetadata,
  TensorStats,
  TypedArray,
  Array2D,
  Array3D,
  Array4D,
} from './tensor.types'

// Attention types
export type {
  HeadClassification,
  AttentionPattern,
  HeadAnalysis,
  LayerAttention,
  AttentionSummary,
  AttentionAnalysisRequest,
  AttentionAnalysisResponse,
  AttentionResponse,
} from './attention.types'

// Activation types
export type {
  ComponentType,
  ActivationTrace,
  ModelConfig,
  LayerActivation,
  TokenNorms,
  ActivationRequest,
  ActivationResponse,
  ResidualAnalysis,
  CreateTraceRequest,
  LoadTraceRequest,
} from './activation.types'

// Circuit types
export type {
  CircuitComponentType,
  CircuitNode,
  CircuitEdge,
  Circuit,
  CircuitDiscoveryParams,
  CircuitDiscoveryRequest,
  CircuitDiscoveryResponse,
  NodePosition,
  CircuitLayout,
  CircuitViewOptions,
} from './circuit.types'

// SAE types
export type {
  SAEActivation,
  SAEConfig,
  SAEFeatures,
  FeatureActivation,
  PositionFeatures,
  FeatureFrequency,
  FeatureCoactivation,
  BehaviorFeatures,
  LoadSAERequest,
  EncodeRequest,
  EncodeResponse,
  SAEAnalysisSession,
  SAEViewOptions,
} from './sae.types'

// IOI types
export type {
  IOITokenRole,
  IOISentence,
  IOIComponentType,
  IOIHead,
  IOICircuit,
  IOIValidationResult,
  IOIDetectionConfig,
  ParseIOISentenceRequest,
  DetectIOIRequest,
  DetectIOIResponse,
  KnownIOIHeads,
  IOIViewOptions,
} from './ioi.types'

// API types
export type {
  ApiResponse,
  PaginatedResponse,
  ApiError,
  WSMessageType,
  WSServerMessage,
  WSClientMessage,
  ProgressData,
  ActivationStreamData,
  ModelInfo,
  ModelsListResponse,
  LoadModelRequest,
  MemoryEstimate,
  HealthCheckResponse,
} from './api.types'

export { ErrorCode } from './api.types'

// Three.js visualization types
export type {
  VisualizationMode,
  ThreeVisualizationMode,
  ThreeColorScale,
  LODLevel,
  ThreeSceneContext,
  ThreeSceneOptions,
  CameraPreset,
  LODSettings,
  WebGLPerformanceMetrics,
  AttentionFlow3DProps,
  AttentionFlow3DEmits,
  Circuit3DProps,
  ActivationLandscape3DProps,
  NodeMeshUserData,
  AttentionCellInstance,
} from './three.types'

export { CAMERA_PRESETS, LOD_CONFIG } from './three.types'
