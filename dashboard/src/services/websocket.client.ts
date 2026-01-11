/**
 * WebSocket Client
 *
 * Handles real-time communication with the FastAPI backend for:
 * - Live trace streaming during inference
 * - Progress updates for long-running operations
 * - Layer-by-layer activation data
 */

import { ref, shallowRef, type Ref, type ShallowRef } from 'vue'

// WebSocket Configuration
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

// Message Types
export const WSMessageType = {
  // Client -> Server
  SUBSCRIBE: 'subscribe',
  UNSUBSCRIBE: 'unsubscribe',
  CREATE_TRACE: 'create_trace',
  CANCEL: 'cancel',

  // Server -> Client
  PROGRESS: 'progress',
  ACTIVATION_DATA: 'activation_data',
  ATTENTION_DATA: 'attention_data',
  COMPLETE: 'complete',
  ERROR: 'error',
  HEARTBEAT: 'heartbeat',
} as const

export type WSMessageType = (typeof WSMessageType)[keyof typeof WSMessageType]

// Message Interfaces
export interface WSProgressData {
  operation: string
  current: number
  total: number
  message?: string
  percent: number
}

export interface WSActivationData {
  layer: number
  component: string
  shape: number[]
  data?: number[]
}

export interface WSAttentionData {
  layer: number
  head?: number
  shape: number[]
  data?: number[]
}

export interface WSErrorData {
  code: string
  message: string
  details?: Record<string, unknown>
}

export interface WSServerMessage {
  type: WSMessageType
  traceId?: string
  data?: WSProgressData | WSActivationData | WSAttentionData | Record<string, unknown>
  error?: WSErrorData
}

export interface WSClientMessage {
  type: WSMessageType
  traceId?: string
  text?: string
  options?: Record<string, unknown>
}

// Connection State
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting'

// Event Callbacks
export interface WSEventHandlers {
  onProgress?: (traceId: string, data: WSProgressData) => void
  onActivationData?: (traceId: string, data: WSActivationData) => void
  onAttentionData?: (traceId: string, data: WSAttentionData) => void
  onComplete?: (traceId: string, data?: Record<string, unknown>) => void
  onError?: (error: WSErrorData, traceId?: string) => void
  onConnectionChange?: (state: ConnectionState) => void
}

/**
 * WebSocket Client Class
 */
export class WebSocketClient {
  private ws: WebSocket | null = null
  private connectionId: string
  private handlers: WSEventHandlers = {}
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private heartbeatTimeout: ReturnType<typeof setTimeout> | null = null
  private subscriptions = new Set<string>()

  public state: Ref<ConnectionState> = ref('disconnected')
  public lastError: ShallowRef<WSErrorData | null> = shallowRef(null)

  constructor(connectionId?: string) {
    this.connectionId = connectionId || this.generateId()
  }

  private generateId(): string {
    return `conn_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
  }

  /**
   * Connect to WebSocket server
   */
  connect(handlers?: WSEventHandlers): Promise<void> {
    if (handlers) {
      this.handlers = handlers
    }

    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve()
        return
      }

      this.state.value = 'connecting'
      this.handlers.onConnectionChange?.('connecting')

      const url = `${WS_BASE_URL}/ws/traces/${this.connectionId}`
      this.ws = new WebSocket(url)

      this.ws.onopen = () => {
        this.state.value = 'connected'
        this.reconnectAttempts = 0
        this.handlers.onConnectionChange?.('connected')
        this.startHeartbeatCheck()

        // Re-subscribe to any active subscriptions
        this.subscriptions.forEach((traceId) => {
          this.send({ type: WSMessageType.SUBSCRIBE, traceId })
        })

        resolve()
      }

      this.ws.onclose = () => {
        this.stopHeartbeatCheck()
        if (this.state.value !== 'disconnected') {
          this.attemptReconnect()
        }
      }

      this.ws.onerror = (_event) => {
        const error: WSErrorData = {
          code: 'WEBSOCKET_ERROR',
          message: 'WebSocket connection error',
        }
        this.lastError.value = error
        this.handlers.onError?.(error)

        if (this.state.value === 'connecting') {
          reject(new Error('Failed to connect'))
        }
      }

      this.ws.onmessage = (event) => {
        this.handleMessage(event)
      }
    })
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.state.value = 'disconnected'
    this.handlers.onConnectionChange?.('disconnected')
    this.stopHeartbeatCheck()
    this.subscriptions.clear()

    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  /**
   * Subscribe to trace updates
   */
  subscribe(traceId: string): void {
    this.subscriptions.add(traceId)
    this.send({ type: WSMessageType.SUBSCRIBE, traceId })
  }

  /**
   * Unsubscribe from trace updates
   */
  unsubscribe(traceId: string): void {
    this.subscriptions.delete(traceId)
    this.send({ type: WSMessageType.UNSUBSCRIBE, traceId })
  }

  /**
   * Request trace creation with streaming updates
   */
  createTrace(text: string, options?: Record<string, unknown>): void {
    this.send({
      type: WSMessageType.CREATE_TRACE,
      text,
      options,
    })
  }

  /**
   * Cancel an ongoing operation
   */
  cancel(traceId: string): void {
    this.send({ type: WSMessageType.CANCEL, traceId })
  }

  /**
   * Send a message to the server
   */
  private send(message: WSClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    }
  }

  /**
   * Handle incoming messages
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WSServerMessage = JSON.parse(event.data)

      switch (message.type) {
        case WSMessageType.PROGRESS:
          if (message.traceId && message.data) {
            this.handlers.onProgress?.(message.traceId, message.data as WSProgressData)
          }
          break

        case WSMessageType.ACTIVATION_DATA:
          if (message.traceId && message.data) {
            this.handlers.onActivationData?.(message.traceId, message.data as WSActivationData)
          }
          break

        case WSMessageType.ATTENTION_DATA:
          if (message.traceId && message.data) {
            this.handlers.onAttentionData?.(message.traceId, message.data as WSAttentionData)
          }
          break

        case WSMessageType.COMPLETE:
          if (message.traceId) {
            this.handlers.onComplete?.(message.traceId, message.data as Record<string, unknown>)
          }
          break

        case WSMessageType.ERROR:
          if (message.error) {
            this.lastError.value = message.error
            this.handlers.onError?.(message.error, message.traceId)
          }
          break

        case WSMessageType.HEARTBEAT:
          this.resetHeartbeatCheck()
          break
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error)
    }
  }

  /**
   * Attempt to reconnect after connection loss
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.state.value = 'disconnected'
      this.handlers.onConnectionChange?.('disconnected')
      this.handlers.onError?.({
        code: 'RECONNECT_FAILED',
        message: 'Failed to reconnect after maximum attempts',
      })
      return
    }

    this.state.value = 'reconnecting'
    this.handlers.onConnectionChange?.('reconnecting')
    this.reconnectAttempts++

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)

    setTimeout(() => {
      if (this.state.value === 'reconnecting') {
        this.connect().catch(() => {
          // Will trigger another reconnect attempt
        })
      }
    }, delay)
  }

  /**
   * Start heartbeat monitoring
   */
  private startHeartbeatCheck(): void {
    this.resetHeartbeatCheck()
  }

  /**
   * Reset heartbeat timer
   */
  private resetHeartbeatCheck(): void {
    this.stopHeartbeatCheck()
    // Expect heartbeat every 30s, timeout after 45s
    this.heartbeatTimeout = setTimeout(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        // Connection may be stale, trigger reconnect
        this.ws.close()
      }
    }, 45000)
  }

  /**
   * Stop heartbeat monitoring
   */
  private stopHeartbeatCheck(): void {
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout)
      this.heartbeatTimeout = null
    }
  }

  /**
   * Check if connected
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

// Singleton instance
let wsClient: WebSocketClient | null = null

/**
 * Get or create WebSocket client instance
 */
export function useWebSocket(handlers?: WSEventHandlers): WebSocketClient {
  if (!wsClient) {
    wsClient = new WebSocketClient()
  }

  if (handlers) {
    wsClient.connect(handlers)
  }

  return wsClient
}

/**
 * Create a new WebSocket client instance
 */
export function createWebSocketClient(connectionId?: string): WebSocketClient {
  return new WebSocketClient(connectionId)
}

export default useWebSocket
