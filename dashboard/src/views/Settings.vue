<script setup lang="ts">
  import { ref } from 'vue'
  import { BaseCard, BaseButton } from '@/components/common'
  import { useTheme } from '@/composables/useTheme'

  const { isDark, toggleDark } = useTheme()

  const apiUrl = ref('http://localhost:8000')
</script>

<template>
  <div class="space-y-6">
    <div>
      <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
        Settings
      </h1>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Configure dashboard preferences and connections.
      </p>
    </div>

    <BaseCard title="Appearance" subtitle="Customize the dashboard look and feel" padding="md">
      <div class="flex items-center justify-between">
        <div>
          <p class="text-sm font-medium text-gray-900 dark:text-gray-100">Dark Mode</p>
          <p class="text-sm text-gray-500 dark:text-gray-400">Toggle dark/light theme</p>
        </div>
        <button
          :class="[
            'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
            isDark ? 'bg-primary-600' : 'bg-gray-200',
          ]"
          role="switch"
          :aria-checked="isDark"
          @click="toggleDark()"
        >
          <span
            :class="[
              'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
              isDark ? 'translate-x-5' : 'translate-x-0',
            ]"
          />
        </button>
      </div>
    </BaseCard>

    <BaseCard title="API Connection" subtitle="Backend server configuration" padding="md">
      <div class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            API URL
          </label>
          <input
            v-model="apiUrl"
            type="text"
            class="w-full max-w-md rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm"
          />
        </div>
        <div class="flex gap-3">
          <BaseButton variant="secondary" size="sm">
            Test Connection
          </BaseButton>
          <BaseButton variant="primary" size="sm">
            Save
          </BaseButton>
        </div>
      </div>
    </BaseCard>

    <BaseCard title="About" subtitle="Application information" padding="md">
      <div class="space-y-2 text-sm text-gray-600 dark:text-gray-400">
        <p><strong>Alignment Observatory Dashboard</strong></p>
        <p>Version: 0.1.0</p>
        <p>A visualization toolkit for AI interpretability research.</p>
        <div class="mt-4">
          <a
            href="https://github.com/alignment-observatory"
            target="_blank"
            class="text-primary-600 dark:text-primary-400 hover:underline"
          >
            View on GitHub
          </a>
        </div>
      </div>
    </BaseCard>
  </div>
</template>
