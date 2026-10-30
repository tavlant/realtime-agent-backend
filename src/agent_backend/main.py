"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent_backend import __version__
from agent_backend.api.middleware import RequestIdMiddleware
from agent_backend.api.routes import router as rest_router
from agent_backend.api.ws import router as ws_router
from agent_backend.config import Settings, get_settings
from agent_backend.container import Container
from agent_backend.log import configure_logging
from agent_backend.telemetry import setup_telemetry


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        app.state.container = container or Container.build(settings)
        await app.state.container.startup()
        try:
            yield
        finally:
            if owned:
                await app.state.container.shutdown()

    app = FastAPI(
        title="Realtime Agent Backend",
        version=__version__,
        description="Streaming LLM chat agents over WebSockets with tools, memory and RAG.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(rest_router)
    app.include_router(ws_router)
    setup_telemetry(app, settings)
    return app
