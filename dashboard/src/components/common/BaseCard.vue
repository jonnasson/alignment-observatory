<script setup lang="ts">
  interface Props {
    title?: string
    subtitle?: string
    padding?: 'none' | 'sm' | 'md' | 'lg'
    hoverable?: boolean
  }

  withDefaults(defineProps<Props>(), {
    title: undefined,
    subtitle: undefined,
    padding: 'md',
    hoverable: false,
  })
</script>

<template>
  <div
    :class="[
      'bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800',
      'shadow-sm',
      { 'transition-shadow hover:shadow-md cursor-pointer': hoverable },
    ]"
  >
    <!-- Header -->
    <div
      v-if="title || $slots.header"
      class="px-4 py-3 border-b border-gray-200 dark:border-gray-800"
    >
      <slot name="header">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-sm font-semibold text-gray-900 dark:text-gray-100">
              {{ title }}
            </h3>
            <p v-if="subtitle" class="mt-0.5 text-xs text-gray-500 dark:text-gray-400">
              {{ subtitle }}
            </p>
          </div>
          <slot name="header-actions" />
        </div>
      </slot>
    </div>

    <!-- Content -->
    <div
      :class="[
        {
          'p-0': padding === 'none',
          'p-3': padding === 'sm',
          'p-4': padding === 'md',
          'p-6': padding === 'lg',
        },
      ]"
    >
      <slot />
    </div>

    <!-- Footer -->
    <div
      v-if="$slots.footer"
      class="px-4 py-3 border-t border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50 rounded-b-xl"
    >
      <slot name="footer" />
    </div>
  </div>
</template>
