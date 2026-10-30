"""Shared fixtures. Everything runs offline: SQLite (aiosqlite), fakeredis and FakeLLM."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from agent_backend.config import Settings
from agent_backend.container import Container
from agent_backend.main import create_app


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "database_url": f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}",
        "redis_url": "memory://",
        "llm_provider": "fake",
        "log_json": False,
        "log_level": "WARNING",
        "vector_store": "memory",
        "lock_wait_timeout_s": 30.0,
        "retry_base_delay_s": 0.01,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings_factory(tmp_path: Path) -> Callable[..., Settings]:
    return lambda **overrides: make_settings(tmp_path, **overrides)


@pytest.fixture
def settings(settings_factory: Callable[..., Settings]) -> Settings:
    return settings_factory()


@pytest.fixture
async def container(settings: Settings) -> AsyncIterator[Container]:
    built = Container.build(settings, redis=FakeAsyncRedis(decode_responses=True))
    await built.startup()
    try:
        yield built
    finally:
        await built.shutdown()


@pytest.fixture
def client_factory(
    settings_factory: Callable[..., Settings],
) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(**overrides: Any) -> TestClient:
        client = TestClient(create_app(settings_factory(**overrides)))
        client.__enter__()  # run lifespan (startup)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(client_factory: Callable[..., TestClient]) -> TestClient:
    return client_factory()


def collect_turn(ws: Any) -> list[dict[str, Any]]:
    """Read WebSocket events until the current turn completes (or errors)."""
    events: list[dict[str, Any]] = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] in ("turn_completed", "error"):
            return events
