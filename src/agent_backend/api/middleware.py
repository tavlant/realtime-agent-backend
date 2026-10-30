"""Pure-ASGI request-ID middleware (works for HTTP and leaves WebSockets untouched)."""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agent_backend.log import bind_context

logger = logging.getLogger("agent_backend.access")

HEADER = b"x-request-id"
_VALID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIdMiddleware:
    """Accepts an incoming X-Request-ID (or creates one), binds it to the log context and
    echoes it on the response, then logs one structured access line per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(HEADER, b"").decode("latin-1")
        request_id = incoming if _VALID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        if scope["type"] == "websocket":
            with bind_context(request_id=request_id):
                await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((HEADER, request_id.encode()))
                message["headers"] = headers
            await send(message)

        with bind_context(request_id=request_id):
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                logger.info(
                    "request",
                    extra={
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "status": status_code,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
