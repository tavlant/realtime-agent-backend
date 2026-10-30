"""FastAPI dependency helpers."""

from __future__ import annotations

from typing import cast

from fastapi import Request, WebSocket

from agent_backend.container import Container


def get_container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


def get_ws_container(websocket: WebSocket) -> Container:
    return cast(Container, websocket.app.state.container)
