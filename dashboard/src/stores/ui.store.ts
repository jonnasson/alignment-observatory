/**
 * UI Store
 *
 * Manages global UI state including layout and preferences.
 * Note: Theme is managed by useTheme composable for consistency.
 */

import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { useTheme } from '@/composables/useTheme'

export type Theme = 'light' | 'dark' | 'system'
export type ColorScale = 'viridis' | 'inferno' | 'coolwarm' | 'blues'

export interface PanelSizes {
  sidebarWidth: number
  detailsPanelWidth: number
  bottomPanelHeight: number
}

export interface NotificationItem {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  title: string
  message?: string
  duration?: number
  dismissible?: boolean
}

export const useUIStore = defineStore('ui', () => {
  // ============================================================================
  // State
  // ============================================================================

  // Theme - managed by useTheme composable
  const { isDark } = useTheme()

  // Sidebar
  const sidebarCollapsed = ref(false)
  const sidebarWidth = ref(256)

  // Panel sizes
  const panelSizes = ref<PanelSizes>({
    sidebarWidth: 256,
    detailsPanelWidth: 320,
    bottomPanelHeight: 200,
  })

  // Visualization preferences
  const colorScale = ref<ColorScale>('viridis')
  const showTokenLabels = ref(true)
  const showLayerLabels = ref(true)
  const heatmapInterpolation = ref(true)
  const animationsEnabled = ref(true)

  // Notifications
  const notifications = ref<NotificationItem[]>([])
  const maxNotifications = ref(5)

  // Modal state
  const activeModal = ref<string | null>(null)
  const modalData = ref<Record<string, unknown> | null>(null)

  // Loading overlay
  const globalLoading = ref(false)
  const globalLoadingMessage = ref<string | null>(null)

  // ============================================================================
  // Getters
  // ============================================================================

  const isDarkMode = computed(() => isDark.value)

  const effectiveSidebarWidth = computed(() => (sidebarCollapsed.value ? 64 : panelSizes.value.sidebarWidth))

  // ============================================================================
  // Sidebar
  // ============================================================================

  /**
   * Toggle sidebar collapsed state
   */
  function toggleSidebar(): void {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  /**
   * Set sidebar collapsed state
   */
  function setSidebarCollapsed(collapsed: boolean): void {
    sidebarCollapsed.value = collapsed
  }

  // ============================================================================
  // Panel Sizes
  // ============================================================================

  /**
   * Update panel sizes
   */
  function setPanelSizes(sizes: Partial<PanelSizes>): void {
    panelSizes.value = { ...panelSizes.value, ...sizes }
  }

  // ============================================================================
  // Visualization Preferences
  // ============================================================================

  /**
   * Set color scale
   */
  function setColorScale(scale: ColorScale): void {
    colorScale.value = scale
  }

  /**
   * Toggle token labels
   */
  function toggleTokenLabels(): void {
    showTokenLabels.value = !showTokenLabels.value
  }

  /**
   * Toggle layer labels
   */
  function toggleLayerLabels(): void {
    showLayerLabels.value = !showLayerLabels.value
  }

  /**
   * Toggle animations
   */
  function toggleAnimations(): void {
    animationsEnabled.value = !animationsEnabled.value
  }

  // ============================================================================
  // Notifications
  // ============================================================================

  /**
   * Add a notification
   */
  function notify(notification: Omit<NotificationItem, 'id'>): string {
    const id = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

    const item: NotificationItem = {
      id,
      dismissible: true,
      duration: 5000,
      ...notification,
    }

    notifications.value.push(item)

    // Limit notifications
    while (notifications.value.length > maxNotifications.value) {
      notifications.value.shift()
    }

    // Auto-dismiss
    if (item.duration && item.duration > 0) {
      setTimeout(() => {
        dismissNotification(id)
      }, item.duration)
    }

    return id
  }

  /**
   * Dismiss a notification
   */
  function dismissNotification(id: string): void {
    const idx = notifications.value.findIndex((n) => n.id === id)
    if (idx >= 0) {
      notifications.value.splice(idx, 1)
    }
  }

  /**
   * Clear all notifications
   */
  function clearNotifications(): void {
    notifications.value = []
  }

  /**
   * Shorthand notification methods
   */
  function notifySuccess(title: string, message?: string): string {
    return notify({ type: 'success', title, message })
  }

  function notifyError(title: string, message?: string): string {
    return notify({ type: 'error', title, message, duration: 10000 })
  }

  function notifyWarning(title: string, message?: string): string {
    return notify({ type: 'warning', title, message })
  }

  function notifyInfo(title: string, message?: string): string {
    return notify({ type: 'info', title, message })
  }

  // ============================================================================
  // Modal
  // ============================================================================

  /**
   * Open a modal
   */
  function openModal(modalId: string, data?: Record<string, unknown>): void {
    activeModal.value = modalId
    modalData.value = data ?? null
  }

  /**
   * Close modal
   */
  function closeModal(): void {
    activeModal.value = null
    modalData.value = null
  }

  // ============================================================================
  // Global Loading
  // ============================================================================

  /**
   * Show global loading overlay
   */
  function showLoading(message?: string): void {
    globalLoading.value = true
    globalLoadingMessage.value = message ?? null
  }

  /**
   * Hide global loading overlay
   */
  function hideLoading(): void {
    globalLoading.value = false
    globalLoadingMessage.value = null
  }

  // ============================================================================
  // Persistence
  // ============================================================================

  /**
   * Save preferences to localStorage
   * Note: Theme is saved separately by useTheme composable
   */
  function savePreferences(): void {
    const prefs = {
      sidebarCollapsed: sidebarCollapsed.value,
      panelSizes: panelSizes.value,
      colorScale: colorScale.value,
      showTokenLabels: showTokenLabels.value,
      showLayerLabels: showLayerLabels.value,
      animationsEnabled: animationsEnabled.value,
    }
    localStorage.setItem('ui-preferences', JSON.stringify(prefs))
  }

  /**
   * Load preferences from localStorage
   * Note: Theme is loaded separately by useTheme composable
   */
  function loadPreferences(): void {
    const saved = localStorage.getItem('ui-preferences')
    if (saved) {
      try {
        const prefs = JSON.parse(saved)
        if (prefs.sidebarCollapsed !== undefined) sidebarCollapsed.value = prefs.sidebarCollapsed
        if (prefs.panelSizes) panelSizes.value = { ...panelSizes.value, ...prefs.panelSizes }
        if (prefs.colorScale) colorScale.value = prefs.colorScale
        if (prefs.showTokenLabels !== undefined) showTokenLabels.value = prefs.showTokenLabels
        if (prefs.showLayerLabels !== undefined) showLayerLabels.value = prefs.showLayerLabels
        if (prefs.animationsEnabled !== undefined) animationsEnabled.value = prefs.animationsEnabled
      } catch {
        // Ignore parse errors
      }
    }
  }

  // Watch for preference changes and save
  watch(
    [sidebarCollapsed, panelSizes, colorScale, showTokenLabels, showLayerLabels, animationsEnabled],
    () => {
      savePreferences()
    },
    { deep: true }
  )

  return {
    // State
    sidebarCollapsed,
    sidebarWidth,
    panelSizes,
    colorScale,
    showTokenLabels,
    showLayerLabels,
    heatmapInterpolation,
    animationsEnabled,
    notifications,
    activeModal,
    modalData,
    globalLoading,
    globalLoadingMessage,

    // Getters
    isDarkMode,
    effectiveSidebarWidth,

    // Actions
    toggleSidebar,
    setSidebarCollapsed,
    setPanelSizes,
    setColorScale,
    toggleTokenLabels,
    toggleLayerLabels,
    toggleAnimations,
    notify,
    dismissNotification,
    clearNotifications,
    notifySuccess,
    notifyError,
    notifyWarning,
    notifyInfo,
    openModal,
    closeModal,
    showLoading,
    hideLoading,
    loadPreferences,
    savePreferences,
  }
})
