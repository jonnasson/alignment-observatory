/**
 * Unified Theme Composable
 *
 * Single source of truth for dark mode across the application.
 * Wraps VueUse's useDark() to ensure all components share the same reactive state.
 */

import { useDark, useToggle, usePreferredDark } from '@vueuse/core'
import { computed } from 'vue'

// Single shared instance of useDark for the entire app
// By creating this at module level, all components share the same state
const isDark = useDark({
  storageKey: 'theme-preference',
  valueDark: 'dark',
  valueLight: 'light',
})

const toggleDark = useToggle(isDark)
const prefersDark = usePreferredDark()

/**
 * Use the unified theme state
 *
 * @example
 * ```typescript
 * const { isDark, toggleDark, theme } = useTheme()
 *
 * // Check if dark mode
 * if (isDark.value) { ... }
 *
 * // Toggle dark mode
 * toggleDark()
 *
 * // Set dark mode explicitly
 * setDark(true)
 * ```
 */
export function useTheme() {
  const theme = computed(() => (isDark.value ? 'dark' : 'light'))

  return {
    /** Reactive boolean indicating if dark mode is active */
    isDark,
    /** Computed theme name: 'light' or 'dark' */
    theme,
    /** Toggle between light and dark mode */
    toggleDark,
    /** Reactive boolean indicating system preference */
    prefersDark,
    /** Set dark mode explicitly */
    setDark: (value: boolean) => {
      isDark.value = value
    },
  }
}
