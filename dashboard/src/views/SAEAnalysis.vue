<script setup lang="ts">
/**
 * SAE Analysis View
 *
 * Analyze sparse autoencoder features and their activations.
 */

import { ref, computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useTraceStore } from '@/stores/trace.store'
import { BaseCard, BaseButton, BaseSelect, LoadingSpinner } from '@/components/common'
import {
  FeatureActivationGrid,
  TopKFeatures,
  FeatureStats,
  FeatureCoactivation,
} from '@/components/sae'
import { useSAE } from '@/composables/useSAE'
import type {
  SAEConfig,
  SAEFeatures,
  PositionFeatures,
  FeatureActivation,
} from '@/types'

const traceStore = useTraceStore()
const { traceList } = storeToRefs(traceStore)

// Local state
const selectedSAE = ref<string | null>(null)
const selectedTraceId = ref<string | null>(null)
const topK = ref(10)
const isLoading = ref(false)
const showCoactivation = ref(false)

// Demo data for testing
const demoFeatures = ref<SAEFeatures | null>(null)
const demoTopKPerPosition = ref<PositionFeatures[] | null>(null)
const demoConfig = ref<SAEConfig | null>(null)

// SAE options
const saeOptions = [
  { value: 'gpt2-small-resid-0', label: 'GPT-2 Small Layer 0 Residual' },
  { value: 'gpt2-small-resid-5', label: 'GPT-2 Small Layer 5 Residual' },
  { value: 'gpt2-small-mlp-3', label: 'GPT-2 Small Layer 3 MLP' },
  { value: 'gpt2-small-attn-7', label: 'GPT-2 Small Layer 7 Attention' },
]

// Trace options
const traceOptions = computed(() => {
  return traceList.value.map((trace) => ({
    value: trace.traceId,
    label: trace.inputText.slice(0, 40) + (trace.inputText.length > 40 ? '...' : ''),
  }))
})

// Use SAE composable
const {
  selectedFeature,
  hoveredFeature,
  stats,
  topKGlobal,
  coactivations,
  getFeaturePositions,
} = useSAE({
  features: demoFeatures,
  topKPerPosition: demoTopKPerPosition,
  viewOptions: computed(() => ({
    topK: topK.value,
    activationThreshold: 0.01,
    showCoactivation: showCoactivation.value,
    highlightedFeatures: selectedFeature.value !== null ? [selectedFeature.value] : [],
  })),
})

// Demo tokens
const demoTokens = ref<string[]>([])

// Generate demo data
function generateDemoData(): void {
  isLoading.value = true

  // Simulate API call
  setTimeout(() => {
    // Demo config
    demoConfig.value = {
      dIn: 768,
      dSae: 24576,
      activation: 'relu',
      hookPoint: 'blocks.5.hook_resid_post',
      layer: 5,
    }

    // Demo tokens
    demoTokens.value = [
      'When', ' Mary', ' and', ' John', ' went', ' to', ' the', ' store',
      ',', ' John', ' gave', ' a', ' drink', ' to',
    ]

    // Generate demo top-K features per position
    const positions: PositionFeatures[] = []
    for (let pos = 0; pos < demoTokens.value.length; pos++) {
      const features: FeatureActivation[] = []
      const numActive = Math.floor(Math.random() * 15) + 5

      for (let i = 0; i < numActive; i++) {
        features.push({
          featureIdx: Math.floor(Math.random() * 24576),
          activation: Math.random() * 8 + 0.5,
        })
      }

      // Sort by activation
      features.sort((a, b) => b.activation - a.activation)

      positions.push({
        position: pos,
        token: demoTokens.value[pos],
        topK: features.slice(0, topK.value),
      })
    }

    demoTopKPerPosition.value = positions

    // Generate demo SAE features
    const activationData = new Float32Array(demoTokens.value.length * 100)
    let nonZero = 0
    for (let i = 0; i < activationData.length; i++) {
      if (Math.random() < 0.05) {
        activationData[i] = Math.random() * 5
        nonZero++
      }
    }

    demoFeatures.value = {
      activations: {
        data: Array.from(activationData),
        shape: [demoTokens.value.length, 100],
        dtype: 'float32',
      },
      sparsity: 1 - nonZero / activationData.length,
      meanActiveFeatures: nonZero / demoTokens.value.length,
      config: demoConfig.value,
    }

    isLoading.value = false
  }, 500)
}

// Handle feature selection
function handleFeatureSelect(feature: FeatureActivation): void {
  selectedFeature.value = feature.featureIdx
}

function handleFeatureHover(feature: FeatureActivation | null): void {
  hoveredFeature.value = feature?.featureIdx ?? null
}

function handleGlobalFeatureSelect(featureIdx: number): void {
  selectedFeature.value = selectedFeature.value === featureIdx ? null : featureIdx
}

// Clear data
function clearData(): void {
  demoFeatures.value = null
  demoTopKPerPosition.value = null
  demoConfig.value = null
  demoTokens.value = []
  selectedFeature.value = null
}

// Check if we have data
const hasData = computed(() => demoFeatures.value !== null)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
        SAE Analysis
      </h1>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Analyze sparse autoencoder features and their activations across tokens.
      </p>
    </div>

    <!-- Configuration -->
    <BaseCard title="SAE Configuration" padding="md">
      <div class="flex flex-wrap gap-4 items-end">
        <div class="w-64">
          <BaseSelect
            v-model="selectedSAE"
            :options="saeOptions"
            label="Sparse Autoencoder"
            placeholder="Select SAE"
          />
        </div>

        <div v-if="traceOptions.length > 0" class="w-64">
          <BaseSelect
            v-model="selectedTraceId"
            :options="traceOptions"
            label="Activation Trace"
            placeholder="Select trace"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Top-K Features: {{ topK }}
          </label>
          <input
            v-model="topK"
            type="range"
            min="5"
            max="50"
            step="5"
            class="w-40 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>

        <div class="flex items-center gap-2">
          <input
            id="show-coactivation"
            v-model="showCoactivation"
            type="checkbox"
            class="rounded border-gray-300 text-blue-500 focus:ring-blue-500"
          />
          <label for="show-coactivation" class="text-sm text-gray-700 dark:text-gray-300">
            Show co-activation
          </label>
        </div>

        <div class="flex gap-2">
          <BaseButton
            variant="primary"
            :loading="isLoading"
            @click="generateDemoData"
          >
            <svg class="w-4 h-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            {{ hasData ? 'Regenerate Demo' : 'Load Demo Data' }}
          </BaseButton>

          <BaseButton
            v-if="hasData"
            variant="ghost"
            @click="clearData"
          >
            Clear
          </BaseButton>
        </div>
      </div>
    </BaseCard>

    <!-- Loading state -->
    <div v-if="isLoading" class="flex justify-center py-12">
      <LoadingSpinner size="lg" />
    </div>

    <!-- Main content -->
    <template v-else-if="hasData">
      <!-- Feature Activation Grid -->
      <BaseCard title="Feature Activations" subtitle="Top features per token position" padding="md">
        <FeatureActivationGrid
          :features="demoTopKPerPosition ?? []"
          :top-k="topK"
          :max-activation="stats?.maxActivation ?? 10"
          :tokens="demoTokens"
          @feature-select="handleFeatureSelect"
          @feature-hover="handleFeatureHover"
        />
      </BaseCard>

      <!-- Stats and Top Features -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <!-- Feature Statistics -->
        <BaseCard title="Feature Statistics" subtitle="Sparsity and activation metrics">
          <FeatureStats
            :config="demoConfig ?? undefined"
            :stats="stats ?? undefined"
            :features="demoFeatures ?? undefined"
          />
        </BaseCard>

        <!-- Top Global Features -->
        <BaseCard title="Most Frequent Features" subtitle="Features active across most positions">
          <TopKFeatures
            :features="topKGlobal"
            :max-value="1"
            mode="frequency"
            title=""
            :highlighted="selectedFeature !== null ? [selectedFeature] : []"
            @feature-select="handleGlobalFeatureSelect"
            @feature-hover="(idx) => hoveredFeature = idx"
          />
        </BaseCard>

        <!-- Co-activation -->
        <BaseCard
          title="Feature Co-activation"
          subtitle="Correlated feature pairs"
          :class="{ 'opacity-50': !showCoactivation }"
        >
          <div v-if="!showCoactivation" class="text-sm text-gray-500">
            Enable "Show co-activation" to view feature correlations
          </div>
          <FeatureCoactivation
            v-else
            :coactivations="coactivations"
            :highlighted-feature="selectedFeature ?? undefined"
            @pair-select="(a, b) => console.log('Selected pair:', a, b)"
          />
        </BaseCard>
      </div>

      <!-- Selected Feature Details -->
      <BaseCard v-if="selectedFeature !== null" title="Selected Feature Details" padding="md">
        <div class="space-y-3">
          <div class="flex items-center gap-4">
            <span class="text-lg font-mono font-bold text-blue-600 dark:text-blue-400">
              #{{ selectedFeature.toLocaleString() }}
            </span>
            <BaseButton variant="ghost" size="sm" @click="selectedFeature = null">
              Clear selection
            </BaseButton>
          </div>

          <div class="text-sm text-gray-600 dark:text-gray-400">
            <strong>Active at positions:</strong>
            {{ getFeaturePositions(selectedFeature).join(', ') || 'None in current view' }}
          </div>

          <div class="text-xs text-gray-500">
            Feature interpretation would require SAE feature dashboard integration
          </div>
        </div>
      </BaseCard>
    </template>

    <!-- Empty state -->
    <template v-else>
      <BaseCard title="Feature Activations" subtitle="Top features per token position" padding="none">
        <div class="h-96 flex items-center justify-center text-gray-400 dark:text-gray-500">
          <div class="text-center">
            <svg class="w-12 h-12 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
            <p class="text-lg font-medium">Feature Activation Grid</p>
            <p class="text-sm mt-2">Load an SAE and encode activations to visualize features</p>
            <BaseButton variant="primary" class="mt-4" @click="generateDemoData">
              Load Demo Data
            </BaseButton>
          </div>
        </div>
      </BaseCard>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <BaseCard title="Feature Statistics" subtitle="Sparsity and activation metrics">
          <div class="text-sm text-gray-500 dark:text-gray-400">
            No SAE loaded
          </div>
        </BaseCard>

        <BaseCard title="Feature Co-activation" subtitle="Correlated feature pairs">
          <div class="text-sm text-gray-500 dark:text-gray-400">
            No data available
          </div>
        </BaseCard>
      </div>
    </template>
  </div>
</template>
