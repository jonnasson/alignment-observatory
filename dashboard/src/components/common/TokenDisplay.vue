<script setup lang="ts">
  import { computed } from 'vue'

  interface Props {
    tokens: string[]
    selectedIndex?: number | null
    highlightedIndices?: number[]
    showIndices?: boolean
    interactive?: boolean
    maxVisible?: number
  }

  const props = withDefaults(defineProps<Props>(), {
    selectedIndex: null,
    highlightedIndices: () => [],
    showIndices: false,
    interactive: true,
    maxVisible: undefined,
  })

  defineEmits<{
    tokenClick: [index: number]
    tokenHover: [index: number | null]
  }>()

  const displayTokens = computed(() => {
    if (props.maxVisible && props.tokens.length > props.maxVisible) {
      return [
        ...props.tokens.slice(0, Math.floor(props.maxVisible / 2)),
        '...',
        ...props.tokens.slice(-Math.floor(props.maxVisible / 2)),
      ]
    }
    return props.tokens
  })

  const getTokenIndex = (displayIndex: number): number | null => {
    if (!props.maxVisible || props.tokens.length <= props.maxVisible) {
      return displayIndex
    }
    const half = Math.floor(props.maxVisible / 2)
    if (displayIndex < half) {
      return displayIndex
    }
    if (displayIndex === half) {
      return null // Ellipsis
    }
    return props.tokens.length - (props.maxVisible - displayIndex)
  }

  const isHighlighted = (index: number | null) => {
    if (index === null) return false
    return props.highlightedIndices.includes(index)
  }

  const isSelected = (index: number | null) => {
    if (index === null) return false
    return props.selectedIndex === index
  }

  // Clean up token display (handle special characters)
  const formatToken = (token: string) => {
    // Replace common special tokens
    return token
      .replace(/^Ġ/, ' ') // GPT-2 space prefix
      .replace(/^▁/, ' ') // Llama space prefix
      .replace(/Ċ/g, '\\n') // Newline
  }
</script>

<template>
  <div class="flex flex-wrap gap-1 font-mono text-sm">
    <template v-for="(token, displayIndex) in displayTokens" :key="displayIndex">
      <span
        v-if="token === '...'"
        class="px-1.5 py-0.5 text-gray-400 dark:text-gray-500"
      >
        ...
      </span>
      <span
        v-else
        :class="[
          'px-1.5 py-0.5 rounded transition-colors',
          interactive ? 'cursor-pointer' : '',
          isSelected(getTokenIndex(displayIndex))
            ? 'bg-primary-500 text-white'
            : isHighlighted(getTokenIndex(displayIndex))
              ? 'bg-yellow-200 dark:bg-yellow-900/50 text-yellow-900 dark:text-yellow-100'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700',
        ]"
        :title="showIndices ? `Index: ${getTokenIndex(displayIndex)}` : undefined"
        @click="interactive && getTokenIndex(displayIndex) !== null && $emit('tokenClick', getTokenIndex(displayIndex)!)"
        @mouseenter="interactive && $emit('tokenHover', getTokenIndex(displayIndex))"
        @mouseleave="interactive && $emit('tokenHover', null)"
      >
        <span v-if="showIndices" class="text-xs text-gray-500 dark:text-gray-400 mr-1">
          {{ getTokenIndex(displayIndex) }}:
        </span>
        {{ formatToken(token) }}
      </span>
    </template>
  </div>
</template>
