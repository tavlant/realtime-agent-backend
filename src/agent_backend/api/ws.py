"""WebSocket endpoint that streams assistant turns.

Protocol (JSON text frames)

client -> server
    {"type": "user_message", "content": "..."}
    {"type": "ping"}

server -> client
    {"type": "ready", "conversation_id": "...", "connection_id": "..."}
    {"type": "turn_started", "turn_id": "..."}
    {"type": "token", "turn_id": "...", "text": "..."}            (many)
    {"type": "tool_call", "turn_id": "...", "id": "...", "name": "...", "arguments": {...}}
    {"type": "tool_result", "turn_id": "...", "id": "...", "name": "...", "content": "...",
     "is_error": false}
    {"type": "turn_completed", "turn_id": "...", "content": "...", "usage": {...}}
    {"type": "error", "code": "...", "message": "...", "retryable": bool}
    {"type": "pong"}

Messages on one socket are processed in order. If the client disconnects mid-turn, the
turn still finishes and is persisted, so the answer is in the history on reconnect.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from agent_backend.api.deps import get_ws_container
from agent_backend.api.schemas import ClientPing, ClientUserMessage
from agent_backend.container import Container
from agent_backend.core.locks import LockLostError, LockNotAcquiredError
from agent_backend.core.retry import RetryableError
from agent_backend.db.repository import ConversationNotFoundError
from agent_backend.log import bind_context

logger = logging.getLogger(__name__)

router = APIRouter()

WS_CLOSE_NOT_FOUND = 4404


class SafeSender:
    """Sends events, and turns a dead socket into a no-op instead of failing the turn."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.closed = False

    async def __call__(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        try:
            await self.websocket.send_json(event)
        except Exception:  # any transport error means the client is gone
            self.closed = True
            logger.info("client went away mid-turn; finishing turn without streaming")


def _error(code: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {"type": "error", "code": code, "message": message, "retryable": retryable}


@router.websocket("/ws/conversations/{conversation_id}")
async def conversation_socket(
    websocket: WebSocket,
    conversation_id: str,
    container: Container = Depends(get_ws_container),
) -> None:
    connection_id = uuid.uuid4().hex[:12]
    with bind_context(conversation_id=conversation_id, connection_id=connection_id):
        await websocket.accept()
        if await container.repo.get(conversation_id) is None:
            await websocket.send_json(_error("not_found", "conversation not found"))
            await websocket.close(code=WS_CLOSE_NOT_FOUND)
            return

        send = SafeSender(websocket)
        await send(
            {"type": "ready", "conversation_id": conversation_id, "connection_id": connection_id}
        )
        logger.info("websocket connected")
        try:
            while not send.closed:
                raw = await websocket.receive_text()
                await _handle_frame(raw, conversation_id, container, send)
        except WebSocketDisconnect:
            pass
        finally:
            logger.info("websocket disconnected")


async def _handle_frame(
    raw: str, conversation_id: str, container: Container, send: SafeSender
) -> None:
    try:
        payload = json.loads(raw)
        frame_type = payload.get("type") if isinstance(payload, dict) else None
        if frame_type == "ping":
            ClientPing.model_validate(payload)
            await send({"type": "pong"})
            return
        message = ClientUserMessage.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        await send(_error("invalid_message", f"could not parse frame: {exc}".splitlines()[0]))
        return

    try:
        # Shielded: if this handler is cancelled the turn still completes and persists.
        turn = container.track(container.agent.run_turn(conversation_id, message.content, send))
        await asyncio.shield(turn)
    except LockNotAcquiredError:
        await send(
            _error("conversation_busy", "another turn is still running; retry shortly", True)
        )
    except (RetryableError, LockLostError) as exc:
        logger.warning("turn failed with a transient error", extra={"error": repr(exc)})
        await send(_error("upstream_unavailable", "the model is unavailable; retry", True))
    except ConversationNotFoundError:
        await send(_error("not_found", "conversation not found"))
    except Exception:
        logger.exception("turn failed")
        await send(_error("internal_error", "unexpected server error"))
