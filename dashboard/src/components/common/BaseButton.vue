<script setup lang="ts">
  interface Props {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
    size?: 'sm' | 'md' | 'lg'
    disabled?: boolean
    loading?: boolean
    type?: 'button' | 'submit' | 'reset'
  }

  withDefaults(defineProps<Props>(), {
    variant: 'primary',
    size: 'md',
    disabled: false,
    loading: false,
    type: 'button',
  })

  defineEmits<{
    click: [event: MouseEvent]
  }>()
</script>

<template>
  <button
    :type="type"
    :disabled="disabled || loading"
    :class="[
      // Base styles
      'inline-flex items-center justify-center font-medium rounded-lg',
      'transition-colors duration-150 ease-in-out',
      'focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
      'disabled:opacity-50 disabled:cursor-not-allowed',

      // Size variants
      {
        'px-2.5 py-1.5 text-xs gap-1.5': size === 'sm',
        'px-4 py-2 text-sm gap-2': size === 'md',
        'px-6 py-3 text-base gap-2.5': size === 'lg',
      },

      // Color variants
      {
        'bg-primary-600 text-white hover:bg-primary-700 focus-visible:ring-primary-500':
          variant === 'primary',
        'bg-gray-100 text-gray-900 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-100 dark:hover:bg-gray-700 focus-visible:ring-gray-500':
          variant === 'secondary',
        'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800 focus-visible:ring-gray-500':
          variant === 'ghost',
        'bg-red-600 text-white hover:bg-red-700 focus-visible:ring-red-500':
          variant === 'danger',
      },
    ]"
    @click="$emit('click', $event)"
  >
    <!-- Loading spinner -->
    <svg
      v-if="loading"
      class="animate-spin"
      :class="{ 'h-3 w-3': size === 'sm', 'h-4 w-4': size === 'md', 'h-5 w-5': size === 'lg' }"
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

    <!-- Slot content -->
    <slot />
  </button>
</template>
