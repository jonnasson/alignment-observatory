/**
 * Circuit Graph Composable
 *
 * Provides utilities for circuit graph visualization using Cytoscape.js.
 */

import { ref, computed, watch, onMounted, onUnmounted, type Ref, type ComputedRef } from 'vue'
import cytoscape, { type Core, type NodeSingular, type EdgeSingular, type EventObject } from 'cytoscape'
import dagre from 'cytoscape-dagre'
import type { Circuit, CircuitNode, CircuitEdge, CircuitComponentType } from '@/types'

// Register dagre layout
cytoscape.use(dagre)

/** Color mapping for component types */
export const componentColors: Record<CircuitComponentType, string> = {
  attention: '#3b82f6', // blue
  mlp: '#10b981', // green
  embed: '#8b5cf6', // purple
  unembed: '#f59e0b', // amber
  residual: '#6b7280', // gray
  layer_norm: '#ec4899', // pink
}

/** Component type display names */
export const componentLabels: Record<CircuitComponentType, string> = {
  attention: 'Attention',
  mlp: 'MLP',
  embed: 'Embed',
  unembed: 'Unembed',
  residual: 'Residual',
  layer_norm: 'LayerNorm',
}

export interface GraphStats {
  nodeCount: number
  edgeCount: number
  avgImportance: number
  maxImportance: number
  layerSpan: number
}

export interface UseCircuitGraphOptions {
  container: Ref<HTMLElement | null>
  circuit: Ref<Circuit | null>
  edgeThreshold?: Ref<number>
  showLabels?: Ref<boolean>
  layout?: Ref<'dagre' | 'breadthfirst' | 'cose'>
}

/**
 * Get node label for display
 */
export function getNodeLabel(node: CircuitNode): string {
  if (node.label) return node.label
  if (node.component === 'attention' && node.head !== undefined) {
    return `L${node.layer}H${node.head}`
  }
  if (node.component === 'mlp') {
    return `L${node.layer} MLP`
  }
  if (node.component === 'embed') {
    return 'Embed'
  }
  if (node.component === 'unembed') {
    return 'Unembed'
  }
  return `L${node.layer} ${componentLabels[node.component]}`
}

/**
 * Calculate edge width based on importance
 */
export function getEdgeWidth(importance: number, minWidth = 1, maxWidth = 8): number {
  return minWidth + importance * (maxWidth - minWidth)
}

/**
 * Calculate edge opacity based on importance
 */
export function getEdgeOpacity(importance: number, minOpacity = 0.3, maxOpacity = 1): number {
  return minOpacity + importance * (maxOpacity - minOpacity)
}

/**
 * Generate DOT format graph string
 */
export function generateDotGraph(circuit: Circuit, threshold = 0): string {
  const lines: string[] = ['digraph Circuit {', '  rankdir=TB;', '  node [shape=box];', '']

  // Group nodes by layer for subgraphs
  const nodesByLayer = new Map<number, CircuitNode[]>()
  for (const node of circuit.nodes) {
    const layer = node.layer
    if (!nodesByLayer.has(layer)) {
      nodesByLayer.set(layer, [])
    }
    nodesByLayer.get(layer)!.push(node)
  }

  // Sort layers
  const sortedLayers = Array.from(nodesByLayer.keys()).sort((a, b) => a - b)

  // Add subgraphs for each layer
  for (const layer of sortedLayers) {
    const nodes = nodesByLayer.get(layer)!
    lines.push(`  subgraph cluster_layer_${layer < 0 ? 'neg' + Math.abs(layer) : layer} {`)
    lines.push(`    label="Layer ${layer}";`)
    lines.push(`    style=dashed;`)
    for (const node of nodes) {
      const color = componentColors[node.component]
      const label = getNodeLabel(node)
      lines.push(`    "${node.id}" [label="${label}", style=filled, fillcolor="${color}40", color="${color}"];`)
    }
    lines.push('  }')
    lines.push('')
  }

  // Add edges
  lines.push('  // Edges')
  for (const edge of circuit.edges) {
    if (edge.importance >= threshold) {
      const width = Math.max(1, Math.round(edge.importance * 5))
      lines.push(`  "${edge.source}" -> "${edge.target}" [penwidth=${width}, label="${edge.importance.toFixed(2)}"];`)
    }
  }

  lines.push('}')
  return lines.join('\n')
}

/**
 * Compute graph statistics
 */
export function computeGraphStats(circuit: Circuit, threshold = 0): GraphStats {
  const filteredEdges = circuit.edges.filter(e => e.importance >= threshold)
  const importances = filteredEdges.map(e => e.importance)

  const layers = circuit.nodes.map(n => n.layer)
  const minLayer = Math.min(...layers)
  const maxLayer = Math.max(...layers)

  return {
    nodeCount: circuit.nodes.length,
    edgeCount: filteredEdges.length,
    avgImportance: importances.length > 0 ? importances.reduce((a, b) => a + b, 0) / importances.length : 0,
    maxImportance: importances.length > 0 ? Math.max(...importances) : 0,
    layerSpan: maxLayer - minLayer + 1,
  }
}

/**
 * Circuit graph composable for Cytoscape.js integration
 */
export function useCircuitGraph(options: UseCircuitGraphOptions) {
  const { container, circuit, edgeThreshold, showLabels, layout } = options

  // Internal state
  const cy = ref<Core | null>(null)
  const selectedNode = ref<CircuitNode | null>(null)
  const selectedEdge = ref<CircuitEdge | null>(null)
  const hoveredNode = ref<CircuitNode | null>(null)
  const hoveredEdge = ref<CircuitEdge | null>(null)
  const isInitialized = ref(false)

  // Computed threshold
  const threshold = computed(() => edgeThreshold?.value ?? 0)

  // Filtered edges based on threshold
  const filteredEdges: ComputedRef<CircuitEdge[]> = computed(() => {
    if (!circuit.value) return []
    return circuit.value.edges.filter(e => e.importance >= threshold.value)
  })

  // Graph statistics
  const stats: ComputedRef<GraphStats | null> = computed(() => {
    if (!circuit.value) return null
    return computeGraphStats(circuit.value, threshold.value)
  })

  // DOT graph string
  const dotGraph: ComputedRef<string> = computed(() => {
    if (!circuit.value) return ''
    return generateDotGraph(circuit.value, threshold.value)
  })

  /**
   * Get Cytoscape stylesheet
   */
  function getStylesheet() {
    return [
      {
        selector: 'node',
        style: {
          'background-color': (ele: NodeSingular) => {
            const component = ele.data('component') as CircuitComponentType
            return componentColors[component] || '#6b7280'
          },
          'border-color': (ele: NodeSingular) => {
            const component = ele.data('component') as CircuitComponentType
            return componentColors[component] || '#6b7280'
          },
          'border-width': 2,
          label: showLabels?.value !== false ? 'data(label)' : '',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '11px',
          color: '#1f2937',
          'text-outline-color': '#ffffff',
          'text-outline-width': 2,
          width: 60,
          height: 36,
          shape: 'round-rectangle',
        },
      },
      {
        selector: 'node:selected',
        style: {
          'border-width': 4,
          'border-color': '#1d4ed8',
          'background-color': '#dbeafe',
        },
      },
      {
        selector: 'node.hover',
        style: {
          'border-width': 3,
          'border-color': '#3b82f6',
        },
      },
      {
        selector: 'edge',
        style: {
          width: (ele: EdgeSingular) => getEdgeWidth(ele.data('importance') ?? 0),
          'line-color': '#94a3b8',
          'target-arrow-color': '#94a3b8',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          opacity: (ele: EdgeSingular) => getEdgeOpacity(ele.data('importance') ?? 0),
        },
      },
      {
        selector: 'edge:selected',
        style: {
          'line-color': '#3b82f6',
          'target-arrow-color': '#3b82f6',
          opacity: 1,
        },
      },
      {
        selector: 'edge.hover',
        style: {
          'line-color': '#60a5fa',
          'target-arrow-color': '#60a5fa',
          opacity: 1,
        },
      },
    ]
  }

  /**
   * Get layout configuration
   */
  function getLayoutConfig() {
    const layoutType = layout?.value ?? 'dagre'

    if (layoutType === 'dagre') {
      return {
        name: 'dagre',
        rankDir: 'TB',
        nodeSep: 50,
        rankSep: 80,
        edgeSep: 20,
        animate: true,
        animationDuration: 300,
      }
    }

    if (layoutType === 'breadthfirst') {
      return {
        name: 'breadthfirst',
        directed: true,
        spacingFactor: 1.5,
        animate: true,
        animationDuration: 300,
      }
    }

    // cose
    return {
      name: 'cose',
      animate: true,
      animationDuration: 300,
      nodeRepulsion: () => 8000,
      idealEdgeLength: () => 100,
    }
  }

  /**
   * Initialize Cytoscape graph
   */
  function initGraph(): void {
    if (!container.value || !circuit.value) return

    // Build elements
    const elements: cytoscape.ElementDefinition[] = []

    // Add nodes
    for (const node of circuit.value.nodes) {
      elements.push({
        data: {
          id: node.id,
          label: getNodeLabel(node),
          component: node.component,
          layer: node.layer,
          head: node.head,
          importance: node.importance,
        },
      })
    }

    // Add edges (filtered by threshold)
    for (const edge of filteredEdges.value) {
      elements.push({
        data: {
          id: `${edge.source}-${edge.target}`,
          source: edge.source,
          target: edge.target,
          importance: edge.importance,
          type: edge.type,
        },
      })
    }

    // Destroy existing instance
    if (cy.value) {
      cy.value.destroy()
    }

    // Create Cytoscape instance
    cy.value = cytoscape({
      container: container.value,
      elements,
      style: getStylesheet() as unknown as cytoscape.StylesheetCSS[],
      layout: getLayoutConfig(),
      minZoom: 0.2,
      maxZoom: 3,
      wheelSensitivity: 0.3,
    })

    // Event handlers
    cy.value.on('tap', 'node', (evt: EventObject) => {
      const nodeData = evt.target.data()
      selectedNode.value = circuit.value?.nodes.find(n => n.id === nodeData.id) ?? null
      selectedEdge.value = null
    })

    cy.value.on('tap', 'edge', (evt: EventObject) => {
      const edgeData = evt.target.data()
      selectedEdge.value = circuit.value?.edges.find(
        e => e.source === edgeData.source && e.target === edgeData.target
      ) ?? null
      selectedNode.value = null
    })

    cy.value.on('tap', (evt: EventObject) => {
      if (evt.target === cy.value) {
        selectedNode.value = null
        selectedEdge.value = null
      }
    })

    cy.value.on('mouseover', 'node', (evt: EventObject) => {
      evt.target.addClass('hover')
      const nodeData = evt.target.data()
      hoveredNode.value = circuit.value?.nodes.find(n => n.id === nodeData.id) ?? null
    })

    cy.value.on('mouseout', 'node', (evt: EventObject) => {
      evt.target.removeClass('hover')
      hoveredNode.value = null
    })

    cy.value.on('mouseover', 'edge', (evt: EventObject) => {
      evt.target.addClass('hover')
      const edgeData = evt.target.data()
      hoveredEdge.value = circuit.value?.edges.find(
        e => e.source === edgeData.source && e.target === edgeData.target
      ) ?? null
    })

    cy.value.on('mouseout', 'edge', (evt: EventObject) => {
      evt.target.removeClass('hover')
      hoveredEdge.value = null
    })

    isInitialized.value = true
  }

  /**
   * Update graph with new circuit data or threshold
   */
  function updateGraph(): void {
    if (!cy.value || !circuit.value) return

    // Rebuild elements
    const elements: cytoscape.ElementDefinition[] = []

    for (const node of circuit.value.nodes) {
      elements.push({
        data: {
          id: node.id,
          label: getNodeLabel(node),
          component: node.component,
          layer: node.layer,
          head: node.head,
          importance: node.importance,
        },
      })
    }

    for (const edge of filteredEdges.value) {
      elements.push({
        data: {
          id: `${edge.source}-${edge.target}`,
          source: edge.source,
          target: edge.target,
          importance: edge.importance,
          type: edge.type,
        },
      })
    }

    // Update elements
    cy.value.json({ elements })
    cy.value.layout(getLayoutConfig()).run()
  }

  /**
   * Fit graph to container
   */
  function fitGraph(): void {
    if (!cy.value) return
    cy.value.fit(undefined, 50)
  }

  /**
   * Reset zoom and pan
   */
  function resetView(): void {
    if (!cy.value) return
    cy.value.fit(undefined, 50)
    cy.value.center()
  }

  /**
   * Highlight a specific node
   */
  function highlightNode(nodeId: string): void {
    if (!cy.value) return
    cy.value.nodes().removeClass('highlighted')
    cy.value.$(`#${nodeId}`).addClass('highlighted')
  }

  /**
   * Clear all highlights
   */
  function clearHighlights(): void {
    if (!cy.value) return
    cy.value.nodes().removeClass('highlighted')
    cy.value.edges().removeClass('highlighted')
  }

  /**
   * Export graph as PNG
   */
  function exportPng(): string | null {
    if (!cy.value) return null
    return cy.value.png({ output: 'base64', bg: 'white', full: true })
  }

  /**
   * Get node by ID
   */
  function getNodeById(nodeId: string): CircuitNode | null {
    return circuit.value?.nodes.find(n => n.id === nodeId) ?? null
  }

  /**
   * Get edges connected to a node
   */
  function getNodeEdges(nodeId: string): { incoming: CircuitEdge[]; outgoing: CircuitEdge[] } {
    if (!circuit.value) return { incoming: [], outgoing: [] }
    return {
      incoming: circuit.value.edges.filter(e => e.target === nodeId),
      outgoing: circuit.value.edges.filter(e => e.source === nodeId),
    }
  }

  // Watch for circuit changes
  watch(circuit, (newCircuit) => {
    if (newCircuit && container.value) {
      if (isInitialized.value) {
        updateGraph()
      } else {
        initGraph()
      }
    }
  })

  // Watch for threshold changes
  watch(threshold, () => {
    if (isInitialized.value) {
      updateGraph()
    }
  })

  // Watch for layout changes
  watch(() => layout?.value, () => {
    if (isInitialized.value && cy.value) {
      cy.value.layout(getLayoutConfig()).run()
    }
  })

  // Initialize when container is ready
  onMounted(() => {
    if (container.value && circuit.value) {
      initGraph()
    }
  })

  // Cleanup on unmount
  onUnmounted(() => {
    if (cy.value) {
      cy.value.destroy()
      cy.value = null
    }
  })

  return {
    // State
    cy,
    selectedNode,
    selectedEdge,
    hoveredNode,
    hoveredEdge,
    isInitialized,

    // Computed
    filteredEdges,
    stats,
    dotGraph,

    // Methods
    initGraph,
    updateGraph,
    fitGraph,
    resetView,
    highlightNode,
    clearHighlights,
    exportPng,
    getNodeById,
    getNodeEdges,
  }
}
