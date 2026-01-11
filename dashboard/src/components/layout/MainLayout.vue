<script setup lang="ts">
  import { ref } from 'vue'
  import AppHeader from './AppHeader.vue'
  import AppSidebar from './AppSidebar.vue'
  import { NotificationToast } from '@/components/common'

  const sidebarOpen = ref(false)
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-950">
    <!-- Sidebar -->
    <AppSidebar
      :open="sidebarOpen"
      @close="sidebarOpen = false"
    />

    <!-- Main content area -->
    <div class="lg:pl-64 flex flex-col min-h-screen">
      <!-- Header -->
      <AppHeader @toggle-sidebar="sidebarOpen = !sidebarOpen" />

      <!-- Page content -->
      <main class="flex-1 p-4 lg:p-6">
        <router-view />
      </main>
    </div>

    <!-- Global notifications -->
    <NotificationToast />
  </div>
</template>
