<script setup lang="ts">
  import { ExclamationTriangleIcon, XMarkIcon } from '@heroicons/vue/24/outline'

  interface Props {
    title?: string
    message: string
    dismissible?: boolean
    variant?: 'error' | 'warning' | 'info'
  }

  withDefaults(defineProps<Props>(), {
    title: undefined,
    dismissible: false,
    variant: 'error',
  })

  defineEmits<{
    dismiss: []
  }>()
</script>

<template>
  <div
    :class="[
      'rounded-lg p-4',
      {
        'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800':
          variant === 'error',
        'bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800':
          variant === 'warning',
        'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800':
          variant === 'info',
      },
    ]"
    role="alert"
  >
    <div class="flex">
      <div class="flex-shrink-0">
        <ExclamationTriangleIcon
          :class="[
            'h-5 w-5',
            {
              'text-red-400 dark:text-red-500': variant === 'error',
              'text-yellow-400 dark:text-yellow-500': variant === 'warning',
              'text-blue-400 dark:text-blue-500': variant === 'info',
            },
          ]"
          aria-hidden="true"
        />
      </div>
      <div class="ml-3 flex-1">
        <h3
          v-if="title"
          :class="[
            'text-sm font-medium',
            {
              'text-red-800 dark:text-red-200': variant === 'error',
              'text-yellow-800 dark:text-yellow-200': variant === 'warning',
              'text-blue-800 dark:text-blue-200': variant === 'info',
            },
          ]"
        >
          {{ title }}
        </h3>
        <p
          :class="[
            'text-sm',
            title ? 'mt-1' : '',
            {
              'text-red-700 dark:text-red-300': variant === 'error',
              'text-yellow-700 dark:text-yellow-300': variant === 'warning',
              'text-blue-700 dark:text-blue-300': variant === 'info',
            },
          ]"
        >
          {{ message }}
        </p>
        <slot name="actions" />
      </div>
      <div v-if="dismissible" class="ml-auto pl-3">
        <div class="-mx-1.5 -my-1.5">
          <button
            type="button"
            :class="[
              'inline-flex rounded-md p-1.5 focus:outline-none focus:ring-2 focus:ring-offset-2',
              {
                'text-red-500 hover:bg-red-100 dark:hover:bg-red-900/30 focus:ring-red-600':
                  variant === 'error',
                'text-yellow-500 hover:bg-yellow-100 dark:hover:bg-yellow-900/30 focus:ring-yellow-600':
                  variant === 'warning',
                'text-blue-500 hover:bg-blue-100 dark:hover:bg-blue-900/30 focus:ring-blue-600':
                  variant === 'info',
              },
            ]"
            @click="$emit('dismiss')"
          >
            <span class="sr-only">Dismiss</span>
            <XMarkIcon class="h-5 w-5" aria-hidden="true" />
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
