<script setup lang="ts">
  import { BaseButton } from '@/components/common'
  import { HomeIcon } from '@heroicons/vue/24/outline'
</script>

<template>
  <div class="min-h-[60vh] flex flex-col items-center justify-center text-center px-4">
    <div class="text-8xl font-bold text-gray-200 dark:text-gray-800">404</div>
    <h1 class="mt-4 text-2xl font-bold text-gray-900 dark:text-gray-100">
      Page Not Found
    </h1>
    <p class="mt-2 text-gray-500 dark:text-gray-400 max-w-md">
      The page you're looking for doesn't exist or has been moved.
    </p>
    <div class="mt-8">
      <router-link to="/">
        <BaseButton variant="primary">
          <HomeIcon class="h-4 w-4" />
          Back to Dashboard
        </BaseButton>
      </router-link>
    </div>
  </div>
</template>
