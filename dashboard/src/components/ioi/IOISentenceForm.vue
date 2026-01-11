<script setup lang="ts">
/**
 * IOISentenceForm Component
 *
 * Form for configuring IOI sentence structure and names.
 */

import { ref, computed } from 'vue'
import { BaseButton } from '@/components/common'

const props = defineProps({
  /** Initial input text */
  initialText: {
    type: String,
    default: 'When Mary and John went to the store, John gave a drink to',
  },
  /** Initial subject name */
  initialSubject: {
    type: String,
    default: 'John',
  },
  /** Initial IO name */
  initialIO: {
    type: String,
    default: 'Mary',
  },
  /** Loading state */
  isLoading: {
    type: Boolean,
    default: false,
  },
  /** Disable form */
  disabled: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits<{
  (e: 'detect', text: string, subject: string, io: string): void
  (e: 'validate'): void
}>()

// Form state
const inputText = ref(props.initialText)
const subjectName = ref(props.initialSubject)
const ioName = ref(props.initialIO)

// Validation
const isValid = computed(() => {
  return (
    inputText.value.trim().length > 0 &&
    subjectName.value.trim().length > 0 &&
    ioName.value.trim().length > 0 &&
    inputText.value.includes(subjectName.value) &&
    inputText.value.includes(ioName.value)
  )
})

// Validation message
const validationMessage = computed(() => {
  if (!inputText.value.trim()) return 'Enter input text'
  if (!subjectName.value.trim()) return 'Enter subject name'
  if (!ioName.value.trim()) return 'Enter IO name'
  if (!inputText.value.includes(subjectName.value)) {
    return `Subject "${subjectName.value}" not found in text`
  }
  if (!inputText.value.includes(ioName.value)) {
    return `IO "${ioName.value}" not found in text`
  }
  return ''
})

// Highlight names in text
const highlightedText = computed(() => {
  let text = inputText.value
  if (subjectName.value) {
    text = text.replace(
      new RegExp(`(${subjectName.value})`, 'g'),
      '<span class="bg-blue-200 dark:bg-blue-800 px-0.5 rounded">$1</span>'
    )
  }
  if (ioName.value) {
    text = text.replace(
      new RegExp(`(${ioName.value})`, 'g'),
      '<span class="bg-green-200 dark:bg-green-800 px-0.5 rounded">$1</span>'
    )
  }
  return text
})

// Handle detect
function handleDetect(): void {
  if (isValid.value && !props.isLoading) {
    emit('detect', inputText.value, subjectName.value, ioName.value)
  }
}

// Example sentences
const examples = [
  {
    text: 'When Mary and John went to the store, John gave a drink to',
    subject: 'John',
    io: 'Mary',
  },
  {
    text: 'After Alice met Bob at the park, Bob handed the book to',
    subject: 'Bob',
    io: 'Alice',
  },
  {
    text: 'Since Sarah and Tom arrived early, Tom passed the keys to',
    subject: 'Tom',
    io: 'Sarah',
  },
]

function loadExample(idx: number): void {
  const ex = examples[idx]
  if (ex) {
    inputText.value = ex.text
    subjectName.value = ex.subject
    ioName.value = ex.io
  }
}
</script>

<template>
  <div class="ioi-sentence-form space-y-4">
    <!-- Input text -->
    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
        Input Text
      </label>
      <textarea
        v-model="inputText"
        rows="2"
        class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        placeholder="When Mary and John went to the store, John gave a drink to"
        :disabled="disabled"
      />
      <!-- Preview with highlights -->
      <div
        v-if="inputText"
        class="mt-1 text-sm text-gray-600 dark:text-gray-400"
        v-html="highlightedText"
      />
    </div>

    <!-- Name inputs -->
    <div class="grid grid-cols-2 gap-4">
      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          <span class="inline-block w-2 h-2 rounded bg-blue-500 mr-1" />
          Subject Name (repeated)
        </label>
        <input
          v-model="subjectName"
          type="text"
          class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="John"
          :disabled="disabled"
        />
      </div>
      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          <span class="inline-block w-2 h-2 rounded bg-green-500 mr-1" />
          Indirect Object Name
        </label>
        <input
          v-model="ioName"
          type="text"
          class="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="Mary"
          :disabled="disabled"
        />
      </div>
    </div>

    <!-- Validation message -->
    <div v-if="validationMessage" class="text-sm text-amber-600 dark:text-amber-400">
      {{ validationMessage }}
    </div>

    <!-- Example sentences -->
    <div class="flex flex-wrap gap-2">
      <span class="text-xs text-gray-500">Examples:</span>
      <button
        v-for="(ex, idx) in examples"
        :key="idx"
        class="text-xs text-blue-600 dark:text-blue-400 hover:underline"
        :disabled="disabled"
        @click="loadExample(idx)"
      >
        {{ ex.subject }}/{{ ex.io }}
      </button>
    </div>

    <!-- Action buttons -->
    <div class="flex gap-3 pt-2">
      <BaseButton
        variant="primary"
        :loading="isLoading"
        :disabled="!isValid || disabled"
        @click="handleDetect"
      >
        <svg class="w-4 h-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        Detect IOI Circuit
      </BaseButton>
      <BaseButton
        variant="secondary"
        :disabled="disabled"
        @click="emit('validate')"
      >
        <svg class="w-4 h-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Validate Against Known Heads
      </BaseButton>
    </div>

    <!-- Explanation -->
    <div class="text-xs text-gray-500 border-t border-gray-200 dark:border-gray-700 pt-3 mt-2">
      <strong>IOI Task:</strong> The model should predict the indirect object (IO) name at the end.
      The subject name appears twice, creating ambiguity that the model must resolve using
      positional and structural cues.
    </div>
  </div>
</template>
