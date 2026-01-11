<script setup lang="ts">
  interface Props {
    size?: 'sm' | 'md' | 'lg' | 'xl'
    color?: 'primary' | 'gray' | 'white'
    label?: string
  }

  withDefaults(defineProps<Props>(), {
    size: 'md',
    color: 'primary',
    label: 'Loading...',
  })
</script>

<template>
  <div class="flex flex-col items-center justify-center gap-2" role="status" aria-live="polite">
    <svg
      :class="[
        'animate-spin',
        {
          'h-4 w-4': size === 'sm',
          'h-6 w-6': size === 'md',
          'h-8 w-8': size === 'lg',
          'h-12 w-12': size === 'xl',
        },
        {
          'text-primary-600 dark:text-primary-400': color === 'primary',
          'text-gray-600 dark:text-gray-400': color === 'gray',
          'text-white': color === 'white',
        },
      ]"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        class="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        stroke-width="4"
      />
      <path
        class="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
    <span
      v-if="label"
      :class="[
        'text-sm',
        {
          'text-primary-600 dark:text-primary-400': color === 'primary',
          'text-gray-600 dark:text-gray-400': color === 'gray',
          'text-white': color === 'white',
        },
      ]"
    >
      {{ label }}
    </span>
    <span class="sr-only">{{ label }}</span>
  </div>
</template>
