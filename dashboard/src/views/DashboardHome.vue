<script setup lang="ts">
  import { ref, onMounted, computed } from 'vue'
  import { useRouter } from 'vue-router'
  import { Dialog, DialogPanel, DialogTitle, TransitionRoot, TransitionChild } from '@headlessui/vue'
  import { BaseCard, BaseButton, LoadingSpinner } from '@/components/common'
  import {
    EyeIcon,
    CpuChipIcon,
    ShareIcon,
    SparklesIcon,
    BeakerIcon,
    PlayIcon,
    XMarkIcon,
  } from '@heroicons/vue/24/outline'
  import { useTraceStore } from '@/stores/trace.store'
  import { useModelStore } from '@/stores/model.store'
  import { useUIStore } from '@/stores/ui.store'

  const router = useRouter()
  const traceStore = useTraceStore()
  const modelStore = useModelStore()
  const uiStore = useUIStore()

  // Modal states
  const showNewTraceModal = ref(false)
  const showLoadTraceModal = ref(false)
  const showLoadModelModal = ref(false)

  // Form states
  const newTraceText = ref('')
  const isCreatingTrace = ref(false)
  const isLoadingExample = ref(false)
  const selectedModelName = ref('gpt2')
  const errorMessage = ref<string | null>(null)

  // Example traces
  const exampleTexts = [
    'The quick brown fox jumps over the lazy dog.',
    'When Mary and John went to the store, John gave a drink to',
    'The capital of France is',
  ]

  // Computed
  const recentTraces = computed(() => traceStore.traceList.slice(0, 5))

  // Lifecycle
  onMounted(async () => {
    const results = await Promise.allSettled([
      traceStore.fetchTraces(),
      modelStore.fetchModels(),
    ])
    // Notify user of any failures
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        const operation = index === 0 ? 'traces' : 'models'
        uiStore.notifyError(`Failed to load ${operation}`, result.reason?.message || 'Please try refreshing the page.')
      }
    })
  })

  // Actions
  function handleApiError(err: unknown): void {
    const error = err as { code?: string; message?: string }
    if (error.code === 'MODEL_NOT_LOADED' || error.message?.includes('model')) {
      errorMessage.value = 'Please load a model first before creating a trace.'
      showLoadModelModal.value = true
    } else {
      errorMessage.value = error.message || 'An error occurred. Please try again.'
    }
    // Clear error after 5 seconds
    setTimeout(() => {
      errorMessage.value = null
    }, 5000)
  }

  async function handleCreateTrace() {
    if (!newTraceText.value.trim()) return

    isCreatingTrace.value = true
    errorMessage.value = null
    try {
      await traceStore.createTrace(newTraceText.value)
      showNewTraceModal.value = false
      newTraceText.value = ''
      router.push('/attention')
    } catch (err) {
      console.error('Failed to create trace:', err)
      handleApiError(err)
    } finally {
      isCreatingTrace.value = false
    }
  }

  async function handleLoadExample() {
    isLoadingExample.value = true
    errorMessage.value = null
    try {
      // Use the IOI example sentence
      await traceStore.createTrace('When Mary and John went to the store, John gave a drink to')
      router.push('/attention')
    } catch (err) {
      console.error('Failed to load example:', err)
      handleApiError(err)
    } finally {
      isLoadingExample.value = false
    }
  }

  async function handleSelectTrace(traceId: string) {
    traceStore.setActiveTrace(traceId)
    showLoadTraceModal.value = false
    router.push('/attention')
  }

  async function handleLoadModel() {
    try {
      await modelStore.loadModel(selectedModelName.value)
      showLoadModelModal.value = false
      uiStore.notifySuccess('Model Loaded', `${selectedModelName.value} is now ready.`)
    } catch (err) {
      console.error('Failed to load model:', err)
      handleApiError(err)
    }
  }

  interface QuickAction {
    name: string
    description: string
    href: string
    icon: typeof EyeIcon
    color: string
  }

  const quickActions: QuickAction[] = [
    {
      name: 'Attention Explorer',
      description: 'Visualize attention patterns',
      href: '/attention',
      icon: EyeIcon,
      color: 'text-attention',
    },
    {
      name: 'Activation Browser',
      description: 'Browse layer activations',
      href: '/activations',
      icon: CpuChipIcon,
      color: 'text-activation',
    },
    {
      name: 'Circuit Discovery',
      description: 'Find computational circuits',
      href: '/circuits',
      icon: ShareIcon,
      color: 'text-circuit',
    },
    {
      name: 'SAE Analysis',
      description: 'Explore learned features',
      href: '/sae',
      icon: SparklesIcon,
      color: 'text-sae',
    },
    {
      name: 'IOI Detection',
      description: 'Detect IOI circuits',
      href: '/ioi',
      icon: BeakerIcon,
      color: 'text-primary-500',
    },
  ]
</script>

<template>
  <div class="space-y-6">
    <!-- Page header -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
        Dashboard
      </h1>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Welcome to the Alignment Observatory. Explore transformer internals and discover interpretable circuits.
      </p>
    </div>

    <!-- Error banner -->
    <div
      v-if="errorMessage"
      class="rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-4"
    >
      <p class="text-sm text-red-700 dark:text-red-300">{{ errorMessage }}</p>
    </div>

    <!-- Quick start card -->
    <BaseCard title="Quick Start" subtitle="Create a new trace or load an existing one">
      <div class="flex flex-wrap gap-3">
        <BaseButton variant="primary" @click="showNewTraceModal = true">
          <PlayIcon class="h-4 w-4" />
          New Trace
        </BaseButton>
        <BaseButton variant="secondary" @click="showLoadTraceModal = true">
          Load Trace
        </BaseButton>
        <BaseButton variant="ghost" :disabled="isLoadingExample" @click="handleLoadExample">
          <LoadingSpinner v-if="isLoadingExample" class="h-4 w-4" />
          Load Example
        </BaseButton>
      </div>
    </BaseCard>

    <!-- Quick actions grid -->
    <div>
      <h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
        Analysis Tools
      </h2>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <router-link
          v-for="action in quickActions"
          :key="action.name"
          :to="action.href"
          class="group"
        >
          <BaseCard hoverable padding="md">
            <div class="flex items-start gap-4">
              <div
                :class="[
                  'flex h-10 w-10 items-center justify-center rounded-lg',
                  'bg-gray-100 dark:bg-gray-800 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30',
                  'transition-colors',
                ]"
              >
                <component
                  :is="action.icon"
                  :class="['h-5 w-5', action.color, 'group-hover:text-primary-600 dark:group-hover:text-primary-400']"
                />
              </div>
              <div>
                <h3 class="font-medium text-gray-900 dark:text-gray-100 group-hover:text-primary-600 dark:group-hover:text-primary-400">
                  {{ action.name }}
                </h3>
                <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  {{ action.description }}
                </p>
              </div>
            </div>
          </BaseCard>
        </router-link>
      </div>
    </div>

    <!-- Status overview -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <BaseCard title="Model Status" subtitle="Currently loaded model">
        <div class="flex items-center justify-between">
          <div>
            <p v-if="modelStore.currentModel" class="text-sm font-medium text-gray-900 dark:text-gray-100">
              {{ modelStore.currentModel.name }}
            </p>
            <p v-else class="text-sm text-gray-500 dark:text-gray-400">No model loaded</p>
            <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">
              <template v-if="modelStore.currentModel">
                {{ modelStore.currentModel.numLayers }} layers, {{ modelStore.currentModel.numHeads }} heads
              </template>
              <template v-else>
                Load a model to start tracing
              </template>
            </p>
          </div>
          <BaseButton size="sm" variant="secondary" :disabled="modelStore.isLoadingModel" @click="showLoadModelModal = true">
            <LoadingSpinner v-if="modelStore.isLoadingModel" class="h-4 w-4" />
            {{ modelStore.currentModel ? 'Change Model' : 'Load Model' }}
          </BaseButton>
        </div>
      </BaseCard>

      <BaseCard title="Recent Traces" subtitle="Your recent analysis sessions">
        <div v-if="recentTraces.length > 0" class="space-y-2">
          <button
            v-for="trace in recentTraces"
            :key="trace.traceId"
            type="button"
            :aria-label="`Load trace: ${trace.tokens.join('').slice(0, 30)}`"
            class="w-full text-left px-3 py-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500"
            @click="handleSelectTrace(trace.traceId)"
          >
            <p class="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
              {{ trace.tokens.join('').slice(0, 50) }}{{ trace.tokens.join('').length > 50 ? '...' : '' }}
            </p>
            <p class="text-xs text-gray-400 dark:text-gray-500">
              {{ trace.metadata.numLayers }} layers · {{ trace.tokens.length }} tokens
            </p>
          </button>
        </div>
        <div v-else class="text-sm text-gray-500 dark:text-gray-400">
          <p>No recent traces</p>
          <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">
            Create a trace to get started
          </p>
        </div>
      </BaseCard>
    </div>

    <!-- New Trace Modal -->
    <TransitionRoot :show="showNewTraceModal" as="template">
      <Dialog class="relative z-50" @close="showNewTraceModal = false">
        <TransitionChild
          enter="ease-out duration-300"
          enter-from="opacity-0"
          enter-to="opacity-100"
          leave="ease-in duration-200"
          leave-from="opacity-100"
          leave-to="opacity-0"
        >
          <div class="fixed inset-0 bg-black/30 dark:bg-black/50" />
        </TransitionChild>

        <div class="fixed inset-0 overflow-y-auto">
          <div class="flex min-h-full items-center justify-center p-4">
            <TransitionChild
              enter="ease-out duration-300"
              enter-from="opacity-0 scale-95"
              enter-to="opacity-100 scale-100"
              leave="ease-in duration-200"
              leave-from="opacity-100 scale-100"
              leave-to="opacity-0 scale-95"
            >
              <DialogPanel class="w-full max-w-lg rounded-xl bg-white dark:bg-gray-900 p-6 shadow-xl">
                <div class="flex items-center justify-between mb-4">
                  <DialogTitle class="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    Create New Trace
                  </DialogTitle>
                  <button
                    type="button"
                    aria-label="Close dialog"
                    class="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-md"
                    @click="showNewTraceModal = false"
                  >
                    <XMarkIcon class="h-5 w-5" aria-hidden="true" />
                  </button>
                </div>

                <div class="space-y-4">
                  <div>
                    <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Input Text
                    </label>
                    <textarea
                      v-model="newTraceText"
                      rows="4"
                      class="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
                      placeholder="Enter text to trace through the model..."
                    />
                  </div>

                  <div>
                    <p class="text-sm text-gray-500 dark:text-gray-400 mb-2">Or try an example:</p>
                    <div class="flex flex-wrap gap-2">
                      <button
                        v-for="(example, idx) in exampleTexts"
                        :key="idx"
                        type="button"
                        :aria-label="`Use example: ${example.slice(0, 20)}...`"
                        class="text-xs px-2 py-1 rounded bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
                        @click="newTraceText = example"
                      >
                        {{ example.slice(0, 30) }}...
                      </button>
                    </div>
                  </div>

                  <div class="flex justify-end gap-3 pt-4">
                    <BaseButton variant="ghost" @click="showNewTraceModal = false">
                      Cancel
                    </BaseButton>
                    <BaseButton
                      variant="primary"
                      :disabled="!newTraceText.trim() || isCreatingTrace"
                      @click="handleCreateTrace"
                    >
                      <LoadingSpinner v-if="isCreatingTrace" class="h-4 w-4" />
                      Create Trace
                    </BaseButton>
                  </div>
                </div>
              </DialogPanel>
            </TransitionChild>
          </div>
        </div>
      </Dialog>
    </TransitionRoot>

    <!-- Load Trace Modal -->
    <TransitionRoot :show="showLoadTraceModal" as="template">
      <Dialog class="relative z-50" @close="showLoadTraceModal = false">
        <TransitionChild
          enter="ease-out duration-300"
          enter-from="opacity-0"
          enter-to="opacity-100"
          leave="ease-in duration-200"
          leave-from="opacity-100"
          leave-to="opacity-0"
        >
          <div class="fixed inset-0 bg-black/30 dark:bg-black/50" />
        </TransitionChild>

        <div class="fixed inset-0 overflow-y-auto">
          <div class="flex min-h-full items-center justify-center p-4">
            <TransitionChild
              enter="ease-out duration-300"
              enter-from="opacity-0 scale-95"
              enter-to="opacity-100 scale-100"
              leave="ease-in duration-200"
              leave-from="opacity-100 scale-100"
              leave-to="opacity-0 scale-95"
            >
              <DialogPanel class="w-full max-w-lg rounded-xl bg-white dark:bg-gray-900 p-6 shadow-xl">
                <div class="flex items-center justify-between mb-4">
                  <DialogTitle class="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    Load Existing Trace
                  </DialogTitle>
                  <button
                    type="button"
                    aria-label="Close dialog"
                    class="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-md"
                    @click="showLoadTraceModal = false"
                  >
                    <XMarkIcon class="h-5 w-5" aria-hidden="true" />
                  </button>
                </div>

                <div v-if="traceStore.traceList.length > 0" class="space-y-2 max-h-80 overflow-y-auto">
                  <button
                    v-for="trace in traceStore.traceList"
                    :key="trace.traceId"
                    type="button"
                    :aria-label="`Load trace: ${trace.tokens.join('').slice(0, 30)}`"
                    class="w-full text-left px-4 py-3 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-primary-500 dark:hover:border-primary-500 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500"
                    @click="handleSelectTrace(trace.traceId)"
                  >
                    <p class="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                      {{ trace.tokens.join('').slice(0, 60) }}{{ trace.tokens.join('').length > 60 ? '...' : '' }}
                    </p>
                    <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Model: {{ trace.metadata.modelName }} · {{ trace.metadata.numLayers }} layers · {{ trace.tokens.length }} tokens
                    </p>
                  </button>
                </div>
                <div v-else class="text-center py-8">
                  <p class="text-gray-500 dark:text-gray-400">No traces available</p>
                  <p class="text-sm text-gray-400 dark:text-gray-500 mt-1">Create a new trace to get started</p>
                </div>

                <div class="flex justify-end gap-3 pt-4 mt-4 border-t border-gray-200 dark:border-gray-700">
                  <BaseButton variant="ghost" @click="showLoadTraceModal = false">
                    Cancel
                  </BaseButton>
                </div>
              </DialogPanel>
            </TransitionChild>
          </div>
        </div>
      </Dialog>
    </TransitionRoot>

    <!-- Load Model Modal -->
    <TransitionRoot :show="showLoadModelModal" as="template">
      <Dialog class="relative z-50" @close="showLoadModelModal = false">
        <TransitionChild
          enter="ease-out duration-300"
          enter-from="opacity-0"
          enter-to="opacity-100"
          leave="ease-in duration-200"
          leave-from="opacity-100"
          leave-to="opacity-0"
        >
          <div class="fixed inset-0 bg-black/30 dark:bg-black/50" />
        </TransitionChild>

        <div class="fixed inset-0 overflow-y-auto">
          <div class="flex min-h-full items-center justify-center p-4">
            <TransitionChild
              enter="ease-out duration-300"
              enter-from="opacity-0 scale-95"
              enter-to="opacity-100 scale-100"
              leave="ease-in duration-200"
              leave-from="opacity-100 scale-100"
              leave-to="opacity-0 scale-95"
            >
              <DialogPanel class="w-full max-w-lg rounded-xl bg-white dark:bg-gray-900 p-6 shadow-xl">
                <div class="flex items-center justify-between mb-4">
                  <DialogTitle class="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    Load Model
                  </DialogTitle>
                  <button
                    type="button"
                    aria-label="Close dialog"
                    class="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-md"
                    @click="showLoadModelModal = false"
                  >
                    <XMarkIcon class="h-5 w-5" aria-hidden="true" />
                  </button>
                </div>

                <div class="space-y-4">
                  <div>
                    <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Select Model
                    </label>
                    <select
                      v-model="selectedModelName"
                      class="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
                    >
                      <option value="gpt2">GPT-2 (124M parameters)</option>
                      <option value="gpt2-medium">GPT-2 Medium (355M)</option>
                      <option value="gpt2-large">GPT-2 Large (774M)</option>
                      <option value="gpt2-xl">GPT-2 XL (1.5B)</option>
                    </select>
                  </div>

                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    The model will be downloaded and loaded into memory. This may take a moment for larger models.
                  </p>

                  <div class="flex justify-end gap-3 pt-4">
                    <BaseButton variant="ghost" @click="showLoadModelModal = false">
                      Cancel
                    </BaseButton>
                    <BaseButton
                      variant="primary"
                      :disabled="modelStore.isLoadingModel"
                      @click="handleLoadModel"
                    >
                      <LoadingSpinner v-if="modelStore.isLoadingModel" class="h-4 w-4" />
                      Load Model
                    </BaseButton>
                  </div>
                </div>
              </DialogPanel>
            </TransitionChild>
          </div>
        </div>
      </Dialog>
    </TransitionRoot>
  </div>
</template>
