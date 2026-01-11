import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
import './style.css'

const app = createApp(App)

// State management
const pinia = createPinia()
app.use(pinia)

// Routing
app.use(router)

// Mount app
app.mount('#app')

// Initialize UI store after mount
// Note: Theme is automatically initialized by useTheme composable
import { useUIStore } from './stores'
const uiStore = useUIStore()
uiStore.loadPreferences()
