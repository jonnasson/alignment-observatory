"""
WebSocket endpoint for real-time trace streaming.

Allows clients to:
- Subscribe to trace updates during inference
- Receive layer-by-layer activation data as it's computed
- Get progress updates for long-running operations
"""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.schemas.common import ApiError, ProgressData, WSMessageType, WSServerMessage

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections and subscriptions."""

    def __init__(self) -> None:
        self._active_connections: dict[str, WebSocket] = {}
        self._subscriptions: dict[str, set[str]] = {}  # trace_id -> set of connection_ids

    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self._active_connections[connection_id] = websocket

    def disconnect(self, connection_id: str) -> None:
        """Remove a WebSocket connection."""
        if connection_id in self._active_connections:
            del self._active_connections[connection_id]

        # Remove from all subscriptions
        for trace_id in list(self._subscriptions.keys()):
            self._subscriptions[trace_id].discard(connection_id)
            if not self._subscriptions[trace_id]:
                del self._subscriptions[trace_id]

    def subscribe(self, connection_id: str, trace_id: str) -> None:
        """Subscribe a connection to trace updates."""
        if trace_id not in self._subscriptions:
            self._subscriptions[trace_id] = set()
        self._subscriptions[trace_id].add(connection_id)

    def unsubscribe(self, connection_id: str, trace_id: str) -> None:
        """Unsubscribe a connection from trace updates."""
        if trace_id in self._subscriptions:
            self._subscriptions[trace_id].discard(connection_id)

    async def send_to_connection(self, connection_id: str, message: WSServerMessage) -> None:
        """Send a message to a specific connection."""
        if connection_id in self._active_connections:
            websocket = self._active_connections[connection_id]
            await websocket.send_json(message.model_dump())

    async def broadcast_to_trace(self, trace_id: str, message: WSServerMessage) -> None:
        """Broadcast a message to all connections subscribed to a trace."""
        if trace_id in self._subscriptions:
            for connection_id in self._subscriptions[trace_id]:
                await self.send_to_connection(connection_id, message)

    async def send_progress(
        self,
        trace_id: str,
        operation: str,
        current: int,
        total: int,
        message: str | None = None,
    ) -> None:
        """Send a progress update to trace subscribers."""
        progress = ProgressData(
            operation=operation,
            current=current,
            total=total,
            message=message,
            percent=(current / total) * 100 if total > 0 else 0,
        )
        ws_message = WSServerMessage(
            type=WSMessageType.PROGRESS,
            trace_id=trace_id,
            data=progress.model_dump(),
        )
        await self.broadcast_to_trace(trace_id, ws_message)

    async def send_activation_data(
        self,
        trace_id: str,
        layer: int,
        data: dict[str, Any],
    ) -> None:
        """Send activation data for a layer."""
        ws_message = WSServerMessage(
            type=WSMessageType.ACTIVATION_DATA,
            trace_id=trace_id,
            data={"layer": layer, **data},
        )
        await self.broadcast_to_trace(trace_id, ws_message)

    async def send_error(
        self,
        connection_id: str,
        code: str,
        message: str,
        trace_id: str | None = None,
    ) -> None:
        """Send an error message."""
        ws_message = WSServerMessage(
            type=WSMessageType.ERROR,
            trace_id=trace_id,
            error=ApiError(code=code, message=message),
        )
        await self.send_to_connection(connection_id, ws_message)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/traces/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str) -> None:
    """
    WebSocket endpoint for trace streaming.

    Protocol:
    - Client sends: {"type": "subscribe", "trace_id": "..."}
    - Client sends: {"type": "unsubscribe", "trace_id": "..."}
    - Client sends: {"type": "create_trace", "text": "...", "options": {...}}
    - Server sends: {"type": "progress", "trace_id": "...", "data": {...}}
    - Server sends: {"type": "activation_data", "trace_id": "...", "data": {...}}
    - Server sends: {"type": "complete", "trace_id": "..."}
    - Server sends: {"type": "error", "error": {...}}
    """
    await manager.connect(websocket, connection_id)

    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat(websocket, connection_id))

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == WSMessageType.SUBSCRIBE:
                trace_id = message.get("trace_id")
                if trace_id:
                    manager.subscribe(connection_id, trace_id)

            elif msg_type == WSMessageType.UNSUBSCRIBE:
                trace_id = message.get("trace_id")
                if trace_id:
                    manager.unsubscribe(connection_id, trace_id)

            elif msg_type == WSMessageType.CREATE_TRACE:
                # Handle trace creation with streaming
                # This would integrate with TraceService
                await handle_create_trace(connection_id, message)

            elif msg_type == WSMessageType.CANCEL:
                # Cancel ongoing operation
                trace_id = message.get("trace_id")
                # TODO: Implement cancellation

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await manager.send_error(
            connection_id,
            "WEBSOCKET_ERROR",
            str(e),
        )
    finally:
        heartbeat_task.cancel()
        manager.disconnect(connection_id)


async def send_heartbeat(websocket: WebSocket, connection_id: str) -> None:
    """Send periodic heartbeat messages."""
    from api.config import get_settings

    settings = get_settings()

    while True:
        await asyncio.sleep(settings.ws_heartbeat_interval)
        try:
            message = WSServerMessage(type=WSMessageType.HEARTBEAT)
            await websocket.send_json(message.model_dump())
        except Exception:
            break


async def handle_create_trace(connection_id: str, message: dict[str, Any]) -> None:
    """
    Handle trace creation with streaming updates.

    Streams progress and layer-by-layer activations back to the client.
    """
    text = message.get("text", "")
    options = message.get("options", {})

    # TODO: Integrate with TraceService and MicroscopeService
    # For now, send mock progress updates

    trace_id = "mock_trace"
    manager.subscribe(connection_id, trace_id)

    try:
        # Simulate layer-by-layer processing
        num_layers = 12
        for layer in range(num_layers):
            await manager.send_progress(
                trace_id,
                "inference",
                layer + 1,
                num_layers,
                f"Processing layer {layer}",
            )
            await asyncio.sleep(0.1)  # Simulate computation

        # Send completion
        complete_message = WSServerMessage(
            type=WSMessageType.COMPLETE,
            trace_id=trace_id,
            data={"message": "Trace created successfully"},
        )
        await manager.send_to_connection(connection_id, complete_message)

    except Exception as e:
        await manager.send_error(
            connection_id,
            "TRACE_CREATION_FAILED",
            str(e),
            trace_id,
        )


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager."""
    return manager
