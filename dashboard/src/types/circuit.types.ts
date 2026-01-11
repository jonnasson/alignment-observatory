/**
 * Circuit types for computational graph analysis
 */

/** Component types that can be circuit nodes */
export type CircuitComponentType =
  | 'attention'
  | 'mlp'
  | 'embed'
  | 'unembed'
  | 'residual'
  | 'layer_norm'

/** A node in the computational circuit */
export interface CircuitNode {
  /** Unique identifier for this node */
  id: string
  /** Layer index (or -1 for embed, num_layers for unembed) */
  layer: number
  /** Type of component */
  component: CircuitComponentType
  /** Head index for attention nodes */
  head?: number
  /** Sequence position if position-specific */
  position?: number
  /** Human-readable label */
  label?: string
  /** Importance score from circuit discovery */
  importance?: number
}

/** An edge connecting two circuit nodes */
export interface CircuitEdge {
  /** Source node ID */
  source: string
  /** Target node ID */
  target: string
  /** Edge importance/weight (0-1) */
  importance: number
  /** Edge type description */
  type?: string
  /** Additional metadata */
  metadata?: Record<string, string | number>
}

/** A computational circuit representing a behavior */
export interface Circuit {
  /** Unique identifier */
  id?: string
  /** Human-readable name */
  name: string
  /** Description of what the circuit computes */
  description: string
  /** The behavior this circuit implements */
  behavior: string
  /** Nodes in the circuit */
  nodes: CircuitNode[]
  /** Edges connecting nodes */
  edges: CircuitEdge[]
  /** Total importance (sum of edge weights) */
  totalImportance?: number
  /** Average edge importance */
  avgImportance?: number
  /** DOT format representation for Graphviz */
  dotGraph?: string
}

/** Parameters for circuit discovery */
export interface CircuitDiscoveryParams {
  /** Name for the discovered circuit */
  name: string
  /** Description of the behavior */
  behavior: string
  /** Clean trace ID */
  cleanTraceId: string
  /** Corrupt/counterfactual trace ID */
  corruptTraceId: string
  /** Importance threshold for edge inclusion */
  threshold?: number
  /** Maximum number of nodes to include */
  maxNodes?: number
  /** Custom metric function (defined server-side) */
  metricName?: string
}

/** Request to discover a circuit */
export interface CircuitDiscoveryRequest {
  params: CircuitDiscoveryParams
}

/** Response from circuit discovery */
export interface CircuitDiscoveryResponse {
  circuit: Circuit
  /** Computation time in ms */
  computeTimeMs: number
  /** Number of edges pruned by threshold */
  prunedEdges: number
}

/** Layout position for visualization */
export interface NodePosition {
  nodeId: string
  x: number
  y: number
}

/** Graph layout for circuit visualization */
export interface CircuitLayout {
  nodes: (CircuitNode & { x: number; y: number })[]
  edges: CircuitEdge[]
  /** Layout algorithm used */
  algorithm: 'dagre' | 'force' | 'hierarchical'
  /** Bounding box */
  bounds: { width: number; height: number }
}

/** Options for circuit visualization */
export interface CircuitViewOptions {
  /** Minimum edge importance to display */
  edgeThreshold: number
  /** Show edge labels */
  showEdgeLabels: boolean
  /** Color nodes by type */
  colorByType: boolean
  /** Highlight specific nodes */
  highlightedNodes: string[]
  /** Layout algorithm */
  layout: 'dagre' | 'force' | 'hierarchical'
}
