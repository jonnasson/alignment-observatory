/**
 * IOI Detection Composable
 *
 * Provides utilities for Indirect Object Identification circuit analysis.
 */

import { ref, computed, type Ref } from 'vue'
import type {
  IOICircuit,
  IOIHead,
  IOISentence,
  IOIComponentType,
  IOIValidationResult,
  IOIViewOptions,
  KnownIOIHeads,
} from '@/types'

/** Color mapping for IOI component types */
export const ioiComponentColors: Record<IOIComponentType, string> = {
  name_mover: '#ef4444', // red
  s_inhibition: '#3b82f6', // blue
  duplicate_token: '#22c55e', // green
  previous_token: '#a855f7', // purple
  backup_name_mover: '#f97316', // orange
}

/** Display names for IOI component types */
export const ioiComponentLabels: Record<IOIComponentType, string> = {
  name_mover: 'Name Mover',
  s_inhibition: 'S-Inhibition',
  duplicate_token: 'Duplicate Token',
  previous_token: 'Previous Token',
  backup_name_mover: 'Backup Name Mover',
}

/** Known IOI heads for GPT-2 Small (from Wang et al. 2022) */
export const knownGPT2Heads: KnownIOIHeads = {
  modelType: 'gpt2-small',
  nameMoverHeads: [
    [9, 9],
    [10, 0],
    [9, 6],
  ],
  sInhibitionHeads: [
    [7, 3],
    [7, 9],
    [8, 6],
    [8, 10],
  ],
  duplicateTokenHeads: [
    [0, 1],
    [0, 10],
    [3, 0],
  ],
  previousTokenHeads: [
    [2, 2],
    [4, 11],
  ],
  backupNameMoverHeads: [
    [9, 0],
    [9, 7],
    [10, 1],
    [10, 2],
    [10, 6],
    [10, 10],
    [11, 2],
    [11, 9],
  ],
}

/** Options for useIOI composable */
export interface UseIOIOptions {
  circuit: Ref<IOICircuit | null>
  viewOptions?: Ref<IOIViewOptions>
}

/**
 * Composable for IOI circuit analysis and visualization
 */
export function useIOI(options: UseIOIOptions) {
  const { circuit, viewOptions } = options

  // Default view options
  const defaultViewOptions: IOIViewOptions = {
    showScores: true,
    colorByType: true,
    highlightedHeads: [],
    showValidation: false,
  }

  const currentViewOptions = computed(() => viewOptions?.value ?? defaultViewOptions)

  // Selected head for detail view
  const selectedHead = ref<IOIHead | null>(null)
  const hoveredHead = ref<IOIHead | null>(null)

  // All heads flattened
  const allHeads = computed((): IOIHead[] => {
    if (!circuit.value) return []

    return [
      ...circuit.value.nameMoverHeads,
      ...circuit.value.sInhibitionHeads,
      ...circuit.value.duplicateTokenHeads,
      ...circuit.value.previousTokenHeads,
      ...circuit.value.backupNameMoverHeads,
    ]
  })

  // Heads organized by layer
  const headsByLayer = computed((): Map<number, IOIHead[]> => {
    const result = new Map<number, IOIHead[]>()

    for (const head of allHeads.value) {
      const existing = result.get(head.layer) ?? []
      existing.push(head)
      result.set(head.layer, existing)
    }

    return result
  })

  // Layer range
  const layerRange = computed((): [number, number] => {
    if (allHeads.value.length === 0) return [0, 0]

    const layers = allHeads.value.map((h) => h.layer)
    return [Math.min(...layers), Math.max(...layers)]
  })

  // Get heads of a specific type
  function getHeadsByType(type: IOIComponentType): IOIHead[] {
    if (!circuit.value) return []

    switch (type) {
      case 'name_mover':
        return circuit.value.nameMoverHeads
      case 's_inhibition':
        return circuit.value.sInhibitionHeads
      case 'duplicate_token':
        return circuit.value.duplicateTokenHeads
      case 'previous_token':
        return circuit.value.previousTokenHeads
      case 'backup_name_mover':
        return circuit.value.backupNameMoverHeads
    }
  }

  // Get color for a head
  function getHeadColor(head: IOIHead): string {
    return ioiComponentColors[head.componentType]
  }

  // Check if head is highlighted
  function isHeadHighlighted(head: IOIHead): boolean {
    return currentViewOptions.value.highlightedHeads.some(
      ([l, h]) => l === head.layer && h === head.head
    )
  }

  // Format head for display
  function formatHead(head: IOIHead): string {
    return `L${head.layer}H${head.head}`
  }

  // Validate circuit against known heads
  function validateCircuit(knownHeads: KnownIOIHeads = knownGPT2Heads): IOIValidationResult {
    if (!circuit.value) {
      return {
        precision: 0,
        recall: 0,
        f1Score: 0,
        perComponentMetrics: {
          name_mover: [0, 0, 0],
          s_inhibition: [0, 0, 0],
          duplicate_token: [0, 0, 0],
          previous_token: [0, 0, 0],
          backup_name_mover: [0, 0, 0],
        },
        falsePositives: [],
        falseNegatives: [],
      }
    }

    const componentTypes: IOIComponentType[] = [
      'name_mover',
      's_inhibition',
      'duplicate_token',
      'previous_token',
      'backup_name_mover',
    ]

    const perComponentMetrics: Record<IOIComponentType, [number, number, number]> = {
      name_mover: [0, 0, 0],
      s_inhibition: [0, 0, 0],
      duplicate_token: [0, 0, 0],
      previous_token: [0, 0, 0],
      backup_name_mover: [0, 0, 0],
    }

    const allFP: [number, number][] = []
    const allFN: [number, number][] = []
    let totalTP = 0
    let totalFP = 0
    let totalFN = 0

    for (const type of componentTypes) {
      const detected = getHeadsByType(type).map((h): [number, number] => [h.layer, h.head])
      const known = getKnownHeadsOfType(knownHeads, type)

      const detectedSet = new Set(detected.map(([l, h]) => `${l},${h}`))
      const knownSet = new Set(known.map(([l, h]) => `${l},${h}`))

      let tp = 0
      for (const key of detectedSet) {
        if (knownSet.has(key)) {
          tp++
        } else {
          const parts = key.split(',').map(Number)
          allFP.push([parts[0] ?? 0, parts[1] ?? 0])
        }
      }

      for (const key of knownSet) {
        if (!detectedSet.has(key)) {
          const parts = key.split(',').map(Number)
          allFN.push([parts[0] ?? 0, parts[1] ?? 0])
        }
      }

      const fp = detected.length - tp
      const fn = known.length - tp

      const precision = detected.length > 0 ? tp / detected.length : 0
      const recall = known.length > 0 ? tp / known.length : 0
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0

      perComponentMetrics[type] = [precision, recall, f1]

      totalTP += tp
      totalFP += fp
      totalFN += fn
    }

    const precision = totalTP + totalFP > 0 ? totalTP / (totalTP + totalFP) : 0
    const recall = totalTP + totalFN > 0 ? totalTP / (totalTP + totalFN) : 0
    const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0

    return {
      precision,
      recall,
      f1Score,
      perComponentMetrics,
      falsePositives: allFP,
      falseNegatives: allFN,
    }
  }

  return {
    // State
    selectedHead,
    hoveredHead,
    // Computed
    allHeads,
    headsByLayer,
    layerRange,
    currentViewOptions,
    // Methods
    getHeadsByType,
    getHeadColor,
    isHeadHighlighted,
    formatHead,
    validateCircuit,
  }
}

/** Helper to get known heads of a specific type */
function getKnownHeadsOfType(
  known: KnownIOIHeads,
  type: IOIComponentType
): [number, number][] {
  switch (type) {
    case 'name_mover':
      return known.nameMoverHeads
    case 's_inhibition':
      return known.sInhibitionHeads
    case 'duplicate_token':
      return known.duplicateTokenHeads
    case 'previous_token':
      return known.previousTokenHeads
    case 'backup_name_mover':
      return known.backupNameMoverHeads
  }
}

/**
 * Parse IOI sentence structure from text
 */
export function parseIOISentence(
  _text: string,
  subjectName: string,
  ioName: string,
  tokenStrings: string[]
): IOISentence | null {
  // Find positions of names in token strings
  const subjectPositions: number[] = []
  let ioPosition = -1
  let subject2Position = -1

  for (let i = 0; i < tokenStrings.length; i++) {
    const tokenStr = tokenStrings[i]
    if (!tokenStr) continue
    const token = tokenStr.trim()

    if (token === subjectName || token === ` ${subjectName}`) {
      if (subjectPositions.length === 0) {
        subjectPositions.push(i)
      } else if (subject2Position === -1) {
        subject2Position = i
      }
    }

    if (token === ioName || token === ` ${ioName}`) {
      if (ioPosition === -1) {
        ioPosition = i
      }
    }
  }

  if (subjectPositions.length === 0 || ioPosition === -1) {
    return null
  }

  return {
    tokens: [], // Would need tokenizer
    tokenStrings,
    subjectPositions,
    ioPosition,
    subject2Position: subject2Position === -1 ? (subjectPositions[0] ?? 0) : subject2Position,
    endPosition: tokenStrings.length - 1,
    correctAnswer: ioName,
    distractor: subjectName,
  }
}

/**
 * Generate demo IOI circuit for testing
 */
export function generateDemoIOICircuit(sentence: IOISentence): IOICircuit {
  return {
    nameMoverHeads: [
      {
        layer: 9,
        head: 9,
        componentType: 'name_mover',
        score: 0.92,
        metrics: { copyScore: 0.85, logitContribution: 2.3 },
      },
      {
        layer: 10,
        head: 0,
        componentType: 'name_mover',
        score: 0.88,
        metrics: { copyScore: 0.79, logitContribution: 1.9 },
      },
      {
        layer: 9,
        head: 6,
        componentType: 'name_mover',
        score: 0.71,
        metrics: { copyScore: 0.65, logitContribution: 1.2 },
      },
    ],
    sInhibitionHeads: [
      {
        layer: 7,
        head: 3,
        componentType: 's_inhibition',
        score: 0.85,
        metrics: { inhibitionScore: 0.78 },
      },
      {
        layer: 7,
        head: 9,
        componentType: 's_inhibition',
        score: 0.79,
        metrics: { inhibitionScore: 0.72 },
      },
      {
        layer: 8,
        head: 6,
        componentType: 's_inhibition',
        score: 0.74,
        metrics: { inhibitionScore: 0.68 },
      },
      {
        layer: 8,
        head: 10,
        componentType: 's_inhibition',
        score: 0.69,
        metrics: { inhibitionScore: 0.61 },
      },
    ],
    duplicateTokenHeads: [
      {
        layer: 0,
        head: 1,
        componentType: 'duplicate_token',
        score: 0.91,
        metrics: { duplicateAttention: 0.88 },
      },
      {
        layer: 0,
        head: 10,
        componentType: 'duplicate_token',
        score: 0.82,
        metrics: { duplicateAttention: 0.75 },
      },
      {
        layer: 3,
        head: 0,
        componentType: 'duplicate_token',
        score: 0.67,
        metrics: { duplicateAttention: 0.59 },
      },
    ],
    previousTokenHeads: [
      {
        layer: 2,
        head: 2,
        componentType: 'previous_token',
        score: 0.88,
        metrics: { prevTokenAttention: 0.92 },
      },
      {
        layer: 4,
        head: 11,
        componentType: 'previous_token',
        score: 0.76,
        metrics: { prevTokenAttention: 0.81 },
      },
    ],
    backupNameMoverHeads: [
      {
        layer: 9,
        head: 0,
        componentType: 'backup_name_mover',
        score: 0.58,
        metrics: { copyScore: 0.45 },
      },
      {
        layer: 9,
        head: 7,
        componentType: 'backup_name_mover',
        score: 0.52,
        metrics: { copyScore: 0.41 },
      },
    ],
    validityScore: 0.87,
    sentence,
  }
}
