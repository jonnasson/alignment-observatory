/**
 * Indirect Object Identification (IOI) circuit types
 * Based on Wang et al. 2022: "Interpretability in the Wild"
 */

/** Token role in IOI sentence structure */
export type IOITokenRole =
  | 'subject'
  | 'indirect_object'
  | 'subject_repeat'
  | 'end_position'
  | 'other'

/** An IOI-structured sentence with annotated positions */
export interface IOISentence {
  /** Token IDs */
  tokens: number[]
  /** Token strings */
  tokenStrings: string[]
  /** Positions of the subject (first occurrence) */
  subjectPositions: number[]
  /** Position of the indirect object */
  ioPosition: number
  /** Position of the subject repeat (second occurrence) */
  subject2Position: number
  /** Position where prediction is made */
  endPosition: number
  /** Expected correct answer (IO name) */
  correctAnswer: string
  /** Distractor (Subject name) */
  distractor: string
}

/** IOI head component types */
export type IOIComponentType =
  | 'name_mover'
  | 's_inhibition'
  | 'duplicate_token'
  | 'previous_token'
  | 'backup_name_mover'

/** A head identified as part of the IOI circuit */
export interface IOIHead {
  /** Layer index */
  layer: number
  /** Head index */
  head: number
  /** Component type in IOI circuit */
  componentType: IOIComponentType
  /** Detection score (higher = more confident) */
  score: number
  /** Component-specific metrics */
  metrics: Record<string, number>
}

/** Complete IOI circuit detection result */
export interface IOICircuit {
  /** Name mover heads (copy IO to output) */
  nameMoverHeads: IOIHead[]
  /** S-inhibition heads (suppress subject copying) */
  sInhibitionHeads: IOIHead[]
  /** Duplicate token heads (detect repeated tokens) */
  duplicateTokenHeads: IOIHead[]
  /** Previous token heads (track local context) */
  previousTokenHeads: IOIHead[]
  /** Backup name mover heads */
  backupNameMoverHeads: IOIHead[]
  /** Overall circuit validity score */
  validityScore: number
  /** The sentence used for detection */
  sentence: IOISentence
  /** DOT graph representation */
  dotGraph?: string
}

/** Result of validating against known IOI heads */
export interface IOIValidationResult {
  /** Precision (detected heads that are correct) */
  precision: number
  /** Recall (known heads that were detected) */
  recall: number
  /** F1 score */
  f1Score: number
  /** Per-component metrics: [precision, recall, f1] */
  perComponentMetrics: Record<IOIComponentType, [number, number, number]>
  /** False positive heads */
  falsePositives: [number, number][]
  /** False negative heads (missed) */
  falseNegatives: [number, number][]
}

/** Configuration for IOI detection */
export interface IOIDetectionConfig {
  /** Threshold for name mover detection */
  nameMoverThreshold: number
  /** Threshold for S-inhibition detection */
  sInhibitionThreshold: number
  /** Number of top heads to consider */
  topKHeads: number
  /** Layer ranges to search: [start, end] per component */
  layerRanges?: Record<IOIComponentType, [number, number]>
}

/** Request to parse an IOI sentence */
export interface ParseIOISentenceRequest {
  /** Raw text (e.g., "When Mary and John went to the store, John gave a drink to") */
  text: string
  /** Subject name */
  subjectName: string
  /** Indirect object name */
  ioName: string
}

/** Request to detect IOI circuit */
export interface DetectIOIRequest {
  /** Model ID */
  modelId: string
  /** Pre-parsed sentence */
  sentence: IOISentence
  /** Clean prompt */
  cleanPrompt: string
  /** Corrupt prompt (with swapped names) */
  corruptPrompt: string
  /** Detection configuration */
  config?: IOIDetectionConfig
}

/** Response from IOI detection */
export interface DetectIOIResponse {
  circuit: IOICircuit
  /** Logit difference (IO logit - Subject logit) */
  logitDiff: number
  /** Computation time in ms */
  computeTimeMs: number
}

/** Known IOI heads for validation */
export interface KnownIOIHeads {
  modelType: string
  nameMoverHeads: [number, number][]
  sInhibitionHeads: [number, number][]
  duplicateTokenHeads: [number, number][]
  previousTokenHeads: [number, number][]
  backupNameMoverHeads: [number, number][]
}

/** Options for IOI visualization */
export interface IOIViewOptions {
  /** Show head scores */
  showScores: boolean
  /** Color by component type */
  colorByType: boolean
  /** Highlight specific heads */
  highlightedHeads: [number, number][]
  /** Show validation comparison */
  showValidation: boolean
}
