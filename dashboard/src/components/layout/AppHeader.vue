<script setup lang="ts">
  import { Bars3Icon, SunIcon, MoonIcon } from '@heroicons/vue/24/outline'
  import { useTheme } from '@/composables/useTheme'

  defineEmits<{
    toggleSidebar: []
  }>()

  const { isDark, toggleDark } = useTheme()
</script>

<template>
  <header
    class="sticky top-0 z-40 flex h-14 items-center gap-4 border-b border-gray-200 dark:border-gray-800 bg-white/95 dark:bg-gray-950/95 backdrop-blur supports-[backdrop-filter]:bg-white/60 dark:supports-[backdrop-filter]:bg-gray-950/60 px-4 lg:px-6"
  >
    <!-- Mobile menu button -->
    <button
      class="lg:hidden p-2 -ml-2 rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
      @click="$emit('toggleSidebar')"
    >
      <Bars3Icon class="h-5 w-5" />
      <span class="sr-only">Toggle sidebar</span>
    </button>

    <!-- Logo / Title -->
    <div class="flex items-center gap-2">
      <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600">
        <svg class="h-5 w-5 text-white" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
        </svg>
      </div>
      <span class="font-semibold text-gray-900 dark:text-gray-100 hidden sm:inline">
        Alignment Observatory
      </span>
    </div>

    <!-- Spacer -->
    <div class="flex-1" />

    <!-- Status indicator -->
    <div class="hidden md:flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
      <div class="flex items-center gap-1.5">
        <span class="relative flex h-2 w-2">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
          <span class="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
        </span>
        <span>API Connected</span>
      </div>
    </div>

    <!-- Actions -->
    <div class="flex items-center gap-2">
      <!-- Dark mode toggle -->
      <button
        class="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
        @click="toggleDark()"
      >
        <SunIcon v-if="isDark" class="h-5 w-5" />
        <MoonIcon v-else class="h-5 w-5" />
        <span class="sr-only">Toggle dark mode</span>
      </button>
    </div>
  </header>
</template>
