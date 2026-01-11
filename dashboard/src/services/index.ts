/**
 * Services barrel export
 */

export { api, traceApi, circuitApi, ioiApi, saeApi, modelApi, healthApi } from './api.client'
export type { ApiClientError } from './api.client'
export type {
  TraceInfo,
  CreateTraceParams,
  LoadTraceParams,
  CircuitDiscoveryParams,
  IOISentence,
  DetectIOIParams,
  LoadSAEParams,
  EncodeParams,
  LoadModelParams,
} from './api.client'

export {
  useWebSocket,
  createWebSocketClient,
  WebSocketClient,
  WSMessageType,
} from './websocket.client'
export type {
  WSProgressData,
  WSActivationData,
  WSAttentionData,
  WSErrorData,
  WSServerMessage,
  WSClientMessage,
  ConnectionState,
  WSEventHandlers,
} from './websocket.client'
