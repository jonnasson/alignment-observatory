<script setup lang="ts">
  import { computed } from 'vue'
  import {
    Listbox,
    ListboxButton,
    ListboxOptions,
    ListboxOption,
  } from '@headlessui/vue'
  import { ChevronUpDownIcon, CheckIcon } from '@heroicons/vue/20/solid'

  interface Option {
    value: string | number
    label: string
    disabled?: boolean
  }

  interface Props {
    modelValue: string | number | null
    options: Option[]
    placeholder?: string
    disabled?: boolean
    label?: string
  }

  const props = withDefaults(defineProps<Props>(), {
    placeholder: 'Select...',
    disabled: false,
    label: undefined,
  })

  defineEmits<{
    'update:modelValue': [value: string | number | null]
  }>()

  const selectedOption = computed(() =>
    props.options.find((opt) => opt.value === props.modelValue)
  )
</script>

<template>
  <div class="w-full">
    <label v-if="label" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
      {{ label }}
    </label>
    <Listbox
      :model-value="modelValue"
      :disabled="disabled"
      @update:model-value="$emit('update:modelValue', $event)"
    >
      <div class="relative">
        <ListboxButton
          class="relative w-full cursor-default rounded-lg bg-white dark:bg-gray-800 py-2 pl-3 pr-10 text-left border border-gray-300 dark:border-gray-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:border-primary-500 sm:text-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <span
            :class="[
              'block truncate',
              selectedOption ? 'text-gray-900 dark:text-gray-100' : 'text-gray-500',
            ]"
          >
            {{ selectedOption?.label ?? placeholder }}
          </span>
          <span class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
            <ChevronUpDownIcon class="h-5 w-5 text-gray-400" aria-hidden="true" />
          </span>
        </ListboxButton>

        <transition
          leave-active-class="transition duration-100 ease-in"
          leave-from-class="opacity-100"
          leave-to-class="opacity-0"
        >
          <ListboxOptions
            class="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-lg bg-white dark:bg-gray-800 py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm"
          >
            <ListboxOption
              v-for="option in options"
              :key="option.value"
              v-slot="{ active, selected }"
              :value="option.value"
              :disabled="option.disabled"
              as="template"
            >
              <li
                :class="[
                  'relative cursor-default select-none py-2 pl-10 pr-4',
                  active
                    ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-900 dark:text-primary-100'
                    : 'text-gray-900 dark:text-gray-100',
                  option.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
                ]"
              >
                <span :class="['block truncate', selected ? 'font-medium' : 'font-normal']">
                  {{ option.label }}
                </span>
                <span
                  v-if="selected"
                  class="absolute inset-y-0 left-0 flex items-center pl-3 text-primary-600 dark:text-primary-400"
                >
                  <CheckIcon class="h-5 w-5" aria-hidden="true" />
                </span>
              </li>
            </ListboxOption>
          </ListboxOptions>
        </transition>
      </div>
    </Listbox>
  </div>
</template>
