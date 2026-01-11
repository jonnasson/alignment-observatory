/**
 * Keyboard Shortcuts Composable
 *
 * Provides keyboard shortcut handling with support for modifiers.
 */

import { onMounted, onUnmounted, ref } from 'vue'

export interface Shortcut {
  key: string
  ctrl?: boolean
  shift?: boolean
  alt?: boolean
  meta?: boolean
  action: () => void
  description?: string
  preventDefault?: boolean
}

export interface ShortcutGroup {
  name: string
  shortcuts: Shortcut[]
}

/**
 * Normalize key string for comparison
 */
function normalizeKey(key: string): string {
  return key.toLowerCase()
}

/**
 * Check if event matches shortcut
 */
function matchesShortcut(event: KeyboardEvent, shortcut: Shortcut): boolean {
  const key = normalizeKey(event.key)
  const shortcutKey = normalizeKey(shortcut.key)

  if (key !== shortcutKey) return false

  const ctrlMatch = (shortcut.ctrl ?? false) === (event.ctrlKey || event.metaKey)
  const shiftMatch = (shortcut.shift ?? false) === event.shiftKey
  const altMatch = (shortcut.alt ?? false) === event.altKey

  return ctrlMatch && shiftMatch && altMatch
}

/**
 * Format shortcut for display
 */
export function formatShortcut(shortcut: Shortcut): string {
  const parts: string[] = []

  if (shortcut.ctrl || shortcut.meta) {
    parts.push(navigator.platform.includes('Mac') ? '⌘' : 'Ctrl')
  }
  if (shortcut.alt) {
    parts.push(navigator.platform.includes('Mac') ? '⌥' : 'Alt')
  }
  if (shortcut.shift) {
    parts.push('⇧')
  }

  // Format key
  const keyDisplay = shortcut.key.length === 1 ? shortcut.key.toUpperCase() : shortcut.key

  parts.push(keyDisplay)

  return parts.join(navigator.platform.includes('Mac') ? '' : '+')
}

export function useKeyboardShortcuts(shortcuts: Shortcut[] = []) {
  const registeredShortcuts = ref<Shortcut[]>([...shortcuts])
  const isEnabled = ref(true)

  function handleKeyDown(event: KeyboardEvent): void {
    if (!isEnabled.value) return

    // Skip if typing in input/textarea
    const target = event.target as HTMLElement
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
      return
    }

    for (const shortcut of registeredShortcuts.value) {
      if (matchesShortcut(event, shortcut)) {
        if (shortcut.preventDefault !== false) {
          event.preventDefault()
        }
        shortcut.action()
        return
      }
    }
  }

  /**
   * Register a new shortcut
   */
  function register(shortcut: Shortcut): () => void {
    registeredShortcuts.value.push(shortcut)

    // Return unregister function
    return () => {
      const idx = registeredShortcuts.value.indexOf(shortcut)
      if (idx >= 0) {
        registeredShortcuts.value.splice(idx, 1)
      }
    }
  }

  /**
   * Register multiple shortcuts
   */
  function registerAll(shortcuts: Shortcut[]): () => void {
    const unregisters = shortcuts.map((s) => register(s))
    return () => unregisters.forEach((fn) => fn())
  }

  /**
   * Unregister all shortcuts
   */
  function clear(): void {
    registeredShortcuts.value = []
  }

  /**
   * Enable shortcuts
   */
  function enable(): void {
    isEnabled.value = true
  }

  /**
   * Disable shortcuts
   */
  function disable(): void {
    isEnabled.value = false
  }

  /**
   * Get all shortcuts grouped by description
   */
  function getShortcutGroups(): ShortcutGroup[] {
    const groups: Map<string, Shortcut[]> = new Map()

    for (const shortcut of registeredShortcuts.value) {
      if (!shortcut.description) continue

      // Extract group from description (format: "Group: Description")
      let group = 'General'
      let desc = shortcut.description

      if (shortcut.description.includes(':')) {
        const parts = shortcut.description.split(':')
        group = parts[0]?.trim() ?? 'General'
        desc = parts[1]?.trim() ?? shortcut.description
      }

      if (!groups.has(group)) {
        groups.set(group, [])
      }
      groups.get(group)!.push({ ...shortcut, description: desc })
    }

    return Array.from(groups.entries()).map(([name, shortcuts]) => ({
      name,
      shortcuts,
    }))
  }

  // Setup and teardown
  onMounted(() => {
    window.addEventListener('keydown', handleKeyDown)
  })

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeyDown)
  })

  return {
    shortcuts: registeredShortcuts,
    isEnabled,
    register,
    registerAll,
    clear,
    enable,
    disable,
    getShortcutGroups,
    formatShortcut,
  }
}

/**
 * Common shortcuts preset for visualization
 */
export function getVisualizationShortcuts(handlers: {
  zoomIn?: () => void
  zoomOut?: () => void
  resetView?: () => void
  toggleLabels?: () => void
  nextLayer?: () => void
  prevLayer?: () => void
  nextHead?: () => void
  prevHead?: () => void
  exportImage?: () => void
}): Shortcut[] {
  const shortcuts: Shortcut[] = []

  if (handlers.zoomIn) {
    shortcuts.push({
      key: '=',
      ctrl: true,
      action: handlers.zoomIn,
      description: 'View: Zoom in',
    })
  }

  if (handlers.zoomOut) {
    shortcuts.push({
      key: '-',
      ctrl: true,
      action: handlers.zoomOut,
      description: 'View: Zoom out',
    })
  }

  if (handlers.resetView) {
    shortcuts.push({
      key: '0',
      ctrl: true,
      action: handlers.resetView,
      description: 'View: Reset view',
    })
  }

  if (handlers.toggleLabels) {
    shortcuts.push({
      key: 'l',
      action: handlers.toggleLabels,
      description: 'View: Toggle labels',
    })
  }

  if (handlers.nextLayer) {
    shortcuts.push({
      key: 'ArrowDown',
      action: handlers.nextLayer,
      description: 'Navigation: Next layer',
    })
  }

  if (handlers.prevLayer) {
    shortcuts.push({
      key: 'ArrowUp',
      action: handlers.prevLayer,
      description: 'Navigation: Previous layer',
    })
  }

  if (handlers.nextHead) {
    shortcuts.push({
      key: 'ArrowRight',
      action: handlers.nextHead,
      description: 'Navigation: Next head',
    })
  }

  if (handlers.prevHead) {
    shortcuts.push({
      key: 'ArrowLeft',
      action: handlers.prevHead,
      description: 'Navigation: Previous head',
    })
  }

  if (handlers.exportImage) {
    shortcuts.push({
      key: 's',
      ctrl: true,
      shift: true,
      action: handlers.exportImage,
      description: 'Export: Save as image',
    })
  }

  return shortcuts
}
