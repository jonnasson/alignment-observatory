/**
 * Vue Router configuration
 */

import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: () => import('@/components/layout/MainLayout.vue'),
    children: [
      {
        path: '',
        name: 'home',
        component: () => import('@/views/DashboardHome.vue'),
        meta: {
          title: 'Dashboard',
          description: 'Overview and quick actions',
        },
      },
      {
        path: 'attention',
        name: 'attention',
        component: () => import('@/views/AttentionExplorer.vue'),
        meta: {
          title: 'Attention Explorer',
          description: 'Explore attention patterns across layers and heads',
        },
      },
      {
        path: 'activations',
        name: 'activations',
        component: () => import('@/views/ActivationBrowser.vue'),
        meta: {
          title: 'Activation Browser',
          description: 'Browse layer activations and residual streams',
        },
      },
      {
        path: 'circuits',
        name: 'circuits',
        component: () => import('@/views/CircuitDiscovery.vue'),
        meta: {
          title: 'Circuit Discovery',
          description: 'Discover and visualize computational circuits',
        },
      },
      {
        path: 'sae',
        name: 'sae',
        component: () => import('@/views/SAEAnalysis.vue'),
        meta: {
          title: 'SAE Analysis',
          description: 'Analyze sparse autoencoder features',
        },
      },
      {
        path: 'ioi',
        name: 'ioi',
        component: () => import('@/views/IOIDetection.vue'),
        meta: {
          title: 'IOI Detection',
          description: 'Detect indirect object identification circuits',
        },
      },
      {
        path: 'settings',
        name: 'settings',
        component: () => import('@/views/Settings.vue'),
        meta: {
          title: 'Settings',
          description: 'Configure dashboard settings',
        },
      },
    ],
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/views/NotFound.vue'),
    meta: {
      title: 'Page Not Found',
    },
  },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(_to, _from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    }
    return { top: 0 }
  },
})

// Update document title on navigation
router.beforeEach((to, _from, next) => {
  const title = to.meta.title as string | undefined
  document.title = title ? `${title} | Alignment Observatory` : 'Alignment Observatory'
  next()
})

export default router
