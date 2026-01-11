/**
 * Circuit Store
 *
 * Manages discovered circuits and circuit visualization state.
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { circuitApi, type CircuitDiscoveryParams } from '@/services/api.client'
import type { Circuit, CircuitNode, CircuitEdge } from '@/types'

export interface DiscoveredCircuit {
  id: string
  traceId: string
  targetToken: string
  targetTokenIdx: number
  circuit: Circuit
  method: string
  computationTimeMs: number
  discoveredAt: Date
}

export interface CircuitViewState {
  selectedNodeId: string | null
  hoveredNodeId: string | null
  zoomLevel: number
  panOffset: { x: number; y: number }
  showLabels: boolean
  showEdgeWeights: boolean
  layoutType: 'dagre' | 'force' | 'hierarchical'
}

/** Internal discovery params state (excludes per-call params) */
export interface DiscoveryParamsState {
  method: 'activation_patching' | 'edge_attribution' | 'causal_tracing'
  threshold: number
  maxNodes: number
  includeMlp: boolean
  includeAttention: boolean
}

export const useCircuitStore = defineStore('circuit', () => {
  // ============================================================================
  // State
  // ============================================================================

  // Discovered circuits
  const circuits = ref<Map<string, DiscoveredCircuit>>(new Map())

  // Currently active circuit
  const activeCircuitId = ref<string | null>(null)

  // Circuit view state
  const viewState = ref<CircuitViewState>({
    selectedNodeId: null,
    hoveredNodeId: null,
    zoomLevel: 1,
    panOffset: { x: 0, y: 0 },
    showLabels: true,
    showEdgeWeights: true,
    layoutType: 'dagre',
  })

  // Discovery parameters (excluding traceId and targetTokenIdx which are per-call)
  const discoveryParams = ref<DiscoveryParamsState>({
    method: 'activation_patching',
    threshold: 0.01,
    maxNodes: 50,
    includeMlp: true,
    includeAttention: true,
  })

  // Loading state
  const isDiscovering = ref(false)

  // Error state
  const error = ref<string | null>(null)

  // ============================================================================
  // Getters
  // ============================================================================

  const activeCircuit = computed(() => {
    if (!activeCircuitId.value) return null
    return circuits.value.get(activeCircuitId.value) ?? null
  })

  const circuitList = computed(() => Array.from(circuits.value.values()))

  const activeNodes = computed((): CircuitNode[] => {
    return activeCircuit.value?.circuit.nodes ?? []
  })

  const activeEdges = computed((): CircuitEdge[] => {
    return activeCircuit.value?.circuit.edges ?? []
  })

  const selectedNode = computed((): CircuitNode | null => {
    if (!viewState.value.selectedNodeId || !activeCircuit.value) return null
    return activeCircuit.value.circuit.nodes.find((n) => n.id === viewState.value.selectedNodeId) ?? null
  })

  const nodesByLayer = computed(() => {
    const byLayer = new Map<number, CircuitNode[]>()
    for (const node of activeNodes.value) {
      const layer = node.layer
      if (!byLayer.has(layer)) {
        byLayer.set(layer, [])
      }
      byLayer.get(layer)!.push(node)
    }
    return byLayer
  })

  // ============================================================================
  // Actions
  // ============================================================================

  /**
   * Discover a circuit for a trace
   */
  async function discoverCircuit(
    traceId: string,
    targetTokenIdx: number,
    options?: Partial<CircuitDiscoveryParams>
  ): Promise<DiscoveredCircuit> {
    isDiscovering.value = true
    error.value = null

    const params = {
      ...discoveryParams.value,
      ...options,
      traceId,
      targetTokenIdx,
    }

    try {
      const response = await circuitApi.discover(params as CircuitDiscoveryParams)

      const discovered: DiscoveredCircuit = {
        id: `${traceId}-${targetTokenIdx}-${Date.now()}`,
        traceId: traceId,
        targetToken: `token_${targetTokenIdx}`,
        targetTokenIdx: targetTokenIdx,
        circuit: response.circuit,
        method: params.method ?? 'activation_patching',
        computationTimeMs: response.computeTimeMs,
        discoveredAt: new Date(),
      }

      circuits.value.set(discovered.id, discovered)
      activeCircuitId.value = discovered.id

      return discovered
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to discover circuit'
      throw err
    } finally {
      isDiscovering.value = false
    }
  }

  /**
   * Set active circuit
   */
  function setActiveCircuit(circuitId: string | null): void {
    activeCircuitId.value = circuitId
    // Reset view state when changing circuits
    if (circuitId) {
      viewState.value.selectedNodeId = null
      viewState.value.hoveredNodeId = null
      viewState.value.zoomLevel = 1
      viewState.value.panOffset = { x: 0, y: 0 }
    }
  }

  /**
   * Delete a circuit
   */
  function deleteCircuit(circuitId: string): void {
    circuits.value.delete(circuitId)
    if (activeCircuitId.value === circuitId) {
      activeCircuitId.value = null
    }
  }

  /**
   * Select a node
   */
  function selectNode(nodeId: string | null): void {
    viewState.value.selectedNodeId = nodeId
  }

  /**
   * Set hovered node
   */
  function setHoveredNode(nodeId: string | null): void {
    viewState.value.hoveredNodeId = nodeId
  }

  /**
   * Update zoom level
   */
  function setZoom(level: number): void {
    viewState.value.zoomLevel = Math.max(0.1, Math.min(3, level))
  }

  /**
   * Update pan offset
   */
  function setPan(offset: { x: number; y: number }): void {
    viewState.value.panOffset = offset
  }

  /**
   * Toggle labels visibility
   */
  function toggleLabels(): void {
    viewState.value.showLabels = !viewState.value.showLabels
  }

  /**
   * Toggle edge weights visibility
   */
  function toggleEdgeWeights(): void {
    viewState.value.showEdgeWeights = !viewState.value.showEdgeWeights
  }

  /**
   * Set layout type
   */
  function setLayoutType(type: 'dagre' | 'force' | 'hierarchical'): void {
    viewState.value.layoutType = type
  }

  /**
   * Update discovery parameters
   */
  function setDiscoveryParams(params: Partial<DiscoveryParamsState>): void {
    discoveryParams.value = { ...discoveryParams.value, ...params }
  }

  /**
   * Reset view to fit all nodes
   */
  function resetView(): void {
    viewState.value.zoomLevel = 1
    viewState.value.panOffset = { x: 0, y: 0 }
  }

  /**
   * Get edges connected to a node
   */
  function getNodeEdges(nodeId: string): { incoming: CircuitEdge[]; outgoing: CircuitEdge[] } {
    const incoming = activeEdges.value.filter((e) => e.target === nodeId)
    const outgoing = activeEdges.value.filter((e) => e.source === nodeId)
    return { incoming, outgoing }
  }

  /**
   * Export circuit as DOT format
   */
  function exportAsDot(): string {
    if (!activeCircuit.value) return ''

    const { nodes, edges } = activeCircuit.value.circuit
    const lines: string[] = ['digraph Circuit {', '  rankdir=TB;', '  node [shape=box];', '']

    // Add nodes
    for (const node of nodes) {
      const label = node.label ?? `${node.component} L${node.layer}${node.head !== undefined ? `H${node.head}` : ''}`
      lines.push(`  "${node.id}" [label="${label}"];`)
    }

    lines.push('')

    // Add edges
    for (const edge of edges) {
      const weight = edge.importance.toFixed(2)
      lines.push(`  "${edge.source}" -> "${edge.target}" [label="${weight}"];`)
    }

    lines.push('}')
    return lines.join('\n')
  }

  /**
   * Clear error
   */
  function clearError(): void {
    error.value = null
  }

  return {
    // State
    circuits,
    activeCircuitId,
    viewState,
    discoveryParams,
    isDiscovering,
    error,

    // Getters
    activeCircuit,
    circuitList,
    activeNodes,
    activeEdges,
    selectedNode,
    nodesByLayer,

    // Actions
    discoverCircuit,
    setActiveCircuit,
    deleteCircuit,
    selectNode,
    setHoveredNode,
    setZoom,
    setPan,
    toggleLabels,
    toggleEdgeWeights,
    setLayoutType,
    setDiscoveryParams,
    resetView,
    getNodeEdges,
    exportAsDot,
    clearError,
  }
})
