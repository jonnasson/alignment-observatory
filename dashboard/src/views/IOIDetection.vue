<script setup lang="ts">
/**
 * IOI Detection View
 *
 * Detect and analyze Indirect Object Identification circuits.
 */

import { ref, computed } from 'vue'
import { BaseCard, BaseButton, LoadingSpinner } from '@/components/common'
import {
  IOISentenceForm,
  IOIHeadList,
  IOICircuitView,
  IOIValidation,
} from '@/components/ioi'
import { useIOI, generateDemoIOICircuit, knownGPT2Heads } from '@/composables/useIOI'
import type { IOICircuit, IOISentence, IOIHead, IOIValidationResult } from '@/types'

// Local state
const isLoading = ref(false)
const demoCircuit = ref<IOICircuit | null>(null)
const validationResult = ref<IOIValidationResult | null>(null)
const selectedHead = ref<IOIHead | null>(null)
const hoveredHead = ref<IOIHead | null>(null)

// Use IOI composable
const {
  getHeadColor,
  formatHead,
  validateCircuit,
} = useIOI({
  circuit: demoCircuit,
})

// Current highlighted heads for visualization
const highlightedHeads = computed((): [number, number][] => {
  if (selectedHead.value) {
    return [[selectedHead.value.layer, selectedHead.value.head]]
  }
  if (hoveredHead.value) {
    return [[hoveredHead.value.layer, hoveredHead.value.head]]
  }
  return []
})

// Handle detection
function handleDetect(text: string, subject: string, io: string): void {
  isLoading.value = true
  validationResult.value = null

  // Simulate API call with demo data
  setTimeout(() => {
    // Create a simple sentence structure for demo
    const sentence: IOISentence = {
      tokens: [],
      tokenStrings: text.split(/(\s+)/).filter(t => t.trim()),
      subjectPositions: [],
      ioPosition: -1,
      subject2Position: -1,
      endPosition: 0,
      correctAnswer: io,
      distractor: subject,
    }

    // Find positions
    for (let i = 0; i < sentence.tokenStrings.length; i++) {
      if (sentence.tokenStrings[i] === subject) {
        if (sentence.subjectPositions.length === 0) {
          sentence.subjectPositions.push(i)
        } else {
          sentence.subject2Position = i
        }
      }
      if (sentence.tokenStrings[i] === io && sentence.ioPosition === -1) {
        sentence.ioPosition = i
      }
    }
    sentence.endPosition = sentence.tokenStrings.length - 1

    // Generate demo circuit
    demoCircuit.value = generateDemoIOICircuit(sentence)
    isLoading.value = false
  }, 800)
}

// Handle validation
function handleValidate(): void {
  if (!demoCircuit.value) return

  validationResult.value = validateCircuit(knownGPT2Heads)
}

// Handle head selection
function handleHeadSelect(head: IOIHead): void {
  selectedHead.value = selectedHead.value === head ? null : head
}

function handleHeadHover(head: IOIHead | null): void {
  hoveredHead.value = head
}

// Clear everything
function clearAll(): void {
  demoCircuit.value = null
  validationResult.value = null
  selectedHead.value = null
  hoveredHead.value = null
}

// Check if we have a circuit
const hasCircuit = computed(() => demoCircuit.value !== null)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
        IOI Detection
      </h1>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Detect and analyze Indirect Object Identification circuits in transformer models.
      </p>
    </div>

    <!-- Sentence Form -->
    <BaseCard title="IOI Sentence" subtitle="Configure the IOI task" padding="md">
      <IOISentenceForm
        :is-loading="isLoading"
        @detect="handleDetect"
        @validate="handleValidate"
      />

      <!-- Clear button -->
      <div v-if="hasCircuit" class="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <BaseButton variant="ghost" @click="clearAll">
          <svg class="w-4 h-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
          Clear Results
        </BaseButton>
      </div>
    </BaseCard>

    <!-- Loading state -->
    <div v-if="isLoading" class="flex justify-center py-12">
      <LoadingSpinner size="lg" />
    </div>

    <!-- Circuit visualization -->
    <template v-else>
      <BaseCard title="IOI Circuit" subtitle="Detected circuit components" padding="md">
        <IOICircuitView
          :circuit="demoCircuit ?? undefined"
          :num-layers="12"
          :num-heads="12"
          @head-select="(head) => head && handleHeadSelect(head)"
          @head-hover="(head) => handleHeadHover(head)"
        />
      </BaseCard>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <!-- Detected Heads List -->
        <BaseCard title="Detected Heads" subtitle="IOI circuit components by type">
          <IOIHeadList
            v-if="hasCircuit"
            :name-mover-heads="demoCircuit!.nameMoverHeads"
            :s-inhibition-heads="demoCircuit!.sInhibitionHeads"
            :duplicate-token-heads="demoCircuit!.duplicateTokenHeads"
            :previous-token-heads="demoCircuit!.previousTokenHeads"
            :backup-name-mover-heads="demoCircuit!.backupNameMoverHeads"
            :highlighted="highlightedHeads"
            @head-select="handleHeadSelect"
            @head-hover="handleHeadHover"
          />
          <div v-else class="text-sm text-gray-500">
            Run detection to see IOI circuit heads
          </div>
        </BaseCard>

        <!-- Validation Results -->
        <BaseCard title="Validation Results" subtitle="Comparison with known GPT-2 heads">
          <IOIValidation
            :result="validationResult ?? undefined"
            model-type="GPT-2 Small"
          />
          <div v-if="hasCircuit && !validationResult" class="mt-3">
            <BaseButton variant="secondary" size="sm" @click="handleValidate">
              Run Validation
            </BaseButton>
          </div>
        </BaseCard>
      </div>

      <!-- Selected Head Details -->
      <BaseCard v-if="selectedHead" title="Selected Head Details" padding="md">
        <div class="space-y-3">
          <div class="flex items-center gap-4">
            <span
              class="text-lg font-mono font-bold px-2 py-1 rounded"
              :style="{
                backgroundColor: getHeadColor(selectedHead) + '20',
                color: getHeadColor(selectedHead),
              }"
            >
              {{ formatHead(selectedHead) }}
            </span>
            <BaseButton variant="ghost" size="sm" @click="selectedHead = null">
              Clear selection
            </BaseButton>
          </div>

          <div class="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span class="text-gray-500">Component Type:</span>
              <span class="ml-2 font-medium">{{ selectedHead.componentType.replace(/_/g, ' ') }}</span>
            </div>
            <div>
              <span class="text-gray-500">Detection Score:</span>
              <span class="ml-2 font-medium">{{ (selectedHead.score * 100).toFixed(1) }}%</span>
            </div>
          </div>

          <!-- Metrics -->
          <div v-if="Object.keys(selectedHead.metrics).length > 0">
            <div class="text-sm text-gray-500 mb-1">Metrics:</div>
            <div class="grid grid-cols-2 gap-2">
              <div
                v-for="(value, key) in selectedHead.metrics"
                :key="key"
                class="bg-gray-50 dark:bg-gray-800/50 rounded px-2 py-1 text-sm"
              >
                <span class="text-gray-500">{{ key }}:</span>
                <span class="ml-1 font-mono">{{ typeof value === 'number' ? value.toFixed(3) : value }}</span>
              </div>
            </div>
          </div>
        </div>
      </BaseCard>
    </template>

    <!-- Info card -->
    <BaseCard title="About IOI" padding="md" class="bg-blue-50 dark:bg-blue-900/20">
      <div class="text-sm text-gray-600 dark:text-gray-400 space-y-2">
        <p>
          <strong>Indirect Object Identification (IOI)</strong> is a task where the model must
          identify the correct recipient in sentences like "When Mary and John went to the store,
          John gave a drink to ___".
        </p>
        <p>
          The IOI circuit consists of several component types working together:
        </p>
        <ul class="list-disc list-inside space-y-1 ml-2">
          <li><strong>Name Movers:</strong> Copy the IO name to the output</li>
          <li><strong>S-Inhibition:</strong> Suppress copying of the subject name</li>
          <li><strong>Duplicate Token:</strong> Detect the repeated subject name</li>
          <li><strong>Previous Token:</strong> Track local context for positional info</li>
        </ul>
        <p class="text-xs text-gray-500 mt-3">
          Based on "Interpretability in the Wild" (Wang et al. 2022)
        </p>
      </div>
    </BaseCard>
  </div>
</template>
