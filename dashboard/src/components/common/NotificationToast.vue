<script setup lang="ts">
/**
 * NotificationToast Component
 *
 * Displays notifications from the UI store as toast messages.
 * Positioned at bottom-right of the screen.
 */

import { storeToRefs } from 'pinia'
import { useUIStore } from '@/stores/ui.store'
import {
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XCircleIcon,
} from '@heroicons/vue/24/outline'

const uiStore = useUIStore()
const { notifications } = storeToRefs(uiStore)

const iconMap = {
  success: CheckCircleIcon,
  error: XCircleIcon,
  warning: ExclamationTriangleIcon,
  info: InformationCircleIcon,
}

const colorMap = {
  success:
    'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800 text-green-800 dark:text-green-200',
  error:
    'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800 text-red-800 dark:text-red-200',
  warning:
    'bg-yellow-50 dark:bg-yellow-900/30 border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-200',
  info: 'bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-200',
}
</script>

<template>
  <div
    aria-live="polite"
    aria-atomic="true"
    class="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm"
  >
    <TransitionGroup
      enter-active-class="transition ease-out duration-300"
      enter-from-class="opacity-0 translate-y-2"
      enter-to-class="opacity-100 translate-y-0"
      leave-active-class="transition ease-in duration-200"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 translate-y-2"
    >
      <div
        v-for="notification in notifications"
        :key="notification.id"
        :class="['rounded-lg border p-4 shadow-lg', colorMap[notification.type]]"
        role="alert"
      >
        <div class="flex items-start gap-3">
          <component
            :is="iconMap[notification.type]"
            class="h-5 w-5 flex-shrink-0"
            aria-hidden="true"
          />
          <div class="flex-1 min-w-0">
            <p class="text-sm font-medium">{{ notification.title }}</p>
            <p v-if="notification.message" class="mt-1 text-sm opacity-90">
              {{ notification.message }}
            </p>
          </div>
          <button
            v-if="notification.dismissible"
            type="button"
            class="flex-shrink-0 rounded-md p-1 hover:opacity-75 focus:outline-none focus:ring-2 focus:ring-offset-2"
            aria-label="Dismiss notification"
            @click="uiStore.dismissNotification(notification.id)"
          >
            <XMarkIcon class="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      </div>
    </TransitionGroup>
  </div>
</template>
