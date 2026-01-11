<script setup lang="ts">
  import { useRoute } from 'vue-router'
  import {
    HomeIcon,
    EyeIcon,
    CpuChipIcon,
    ShareIcon,
    SparklesIcon,
    BeakerIcon,
    Cog6ToothIcon,
    XMarkIcon,
  } from '@heroicons/vue/24/outline'

  interface Props {
    open: boolean
  }

  defineProps<Props>()

  defineEmits<{
    close: []
  }>()

  const route = useRoute()

  interface NavItem {
    name: string
    href: string
    icon: typeof HomeIcon
    description: string
  }

  const navigation: NavItem[] = [
    { name: 'Dashboard', href: '/', icon: HomeIcon, description: 'Overview and quick actions' },
    { name: 'Attention', href: '/attention', icon: EyeIcon, description: 'Explore attention patterns' },
    { name: 'Activations', href: '/activations', icon: CpuChipIcon, description: 'Browse layer activations' },
    { name: 'Circuits', href: '/circuits', icon: ShareIcon, description: 'Discover computational circuits' },
    { name: 'SAE Analysis', href: '/sae', icon: SparklesIcon, description: 'Sparse autoencoder features' },
    { name: 'IOI Detection', href: '/ioi', icon: BeakerIcon, description: 'Indirect object identification' },
  ]

  const secondaryNavigation: NavItem[] = [
    { name: 'Settings', href: '/settings', icon: Cog6ToothIcon, description: 'Configure dashboard' },
  ]

  const isActive = (href: string) => {
    if (href === '/') {
      return route.path === '/'
    }
    return route.path.startsWith(href)
  }
</script>

<template>
  <!-- Mobile overlay -->
  <div
    v-if="open"
    class="fixed inset-0 z-40 bg-gray-900/50 lg:hidden"
    @click="$emit('close')"
  />

  <!-- Sidebar -->
  <aside
    :class="[
      'fixed inset-y-0 left-0 z-50 w-64 flex flex-col bg-white dark:bg-gray-950 border-r border-gray-200 dark:border-gray-800',
      'transform transition-transform duration-200 ease-in-out lg:translate-x-0',
      open ? 'translate-x-0' : '-translate-x-full',
    ]"
  >
    <!-- Mobile close button -->
    <div class="flex items-center justify-between h-14 px-4 border-b border-gray-200 dark:border-gray-800 lg:hidden">
      <span class="font-semibold text-gray-900 dark:text-gray-100">Menu</span>
      <button
        class="p-2 -mr-2 rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
        @click="$emit('close')"
      >
        <XMarkIcon class="h-5 w-5" />
        <span class="sr-only">Close menu</span>
      </button>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 overflow-y-auto py-4 px-3">
      <ul class="space-y-1">
        <li v-for="item in navigation" :key="item.name">
          <router-link
            :to="item.href"
            :class="[
              'group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
              isActive(item.href)
                ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800',
            ]"
            @click="$emit('close')"
          >
            <component
              :is="item.icon"
              :class="[
                'h-5 w-5 flex-shrink-0',
                isActive(item.href)
                  ? 'text-primary-600 dark:text-primary-400'
                  : 'text-gray-400 dark:text-gray-500 group-hover:text-gray-600 dark:group-hover:text-gray-400',
              ]"
            />
            <span>{{ item.name }}</span>
          </router-link>
        </li>
      </ul>

      <!-- Secondary navigation -->
      <div class="mt-8">
        <h3 class="px-3 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Settings
        </h3>
        <ul class="mt-2 space-y-1">
          <li v-for="item in secondaryNavigation" :key="item.name">
            <router-link
              :to="item.href"
              :class="[
                'group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive(item.href)
                  ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800',
              ]"
              @click="$emit('close')"
            >
              <component
                :is="item.icon"
                :class="[
                  'h-5 w-5 flex-shrink-0',
                  isActive(item.href)
                    ? 'text-primary-600 dark:text-primary-400'
                    : 'text-gray-400 dark:text-gray-500 group-hover:text-gray-600 dark:group-hover:text-gray-400',
                ]"
              />
              <span>{{ item.name }}</span>
            </router-link>
          </li>
        </ul>
      </div>
    </nav>

    <!-- Footer -->
    <div class="border-t border-gray-200 dark:border-gray-800 p-4">
      <div class="text-xs text-gray-500 dark:text-gray-400">
        <p>Alignment Observatory</p>
        <p class="mt-1">v0.1.0</p>
      </div>
    </div>
  </aside>
</template>
