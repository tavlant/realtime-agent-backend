"""Structured logging with request / conversation / turn IDs carried in contextvars.

Every log line emitted while handling a request or a WebSocket turn automatically
includes the IDs bound for that task, so one turn can be traced across the agent loop,
tool calls and storage without threading IDs through every function signature.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

_context: ContextVar[dict[str, str] | None] = ContextVar("log_context", default=None)

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


def current_context() -> dict[str, str]:
    return dict(_context.get() or {})


@contextmanager
def bind_context(**ids: str | None) -> Iterator[None]:
    """Bind IDs (request_id, conversation_id, turn_id, ...) for the current task."""
    merged = current_context()
    merged.update({k: v for k, v in ids.items() if v is not None})
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(current_context())
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        ctx = current_context()
        extras = {
            k: v for k, v in vars(record).items() if k not in _RESERVED and not k.startswith("_")
        }
        fields = " ".join(f"{k}={v}" for k, v in {**ctx, **extras}.items())
        return f"{base} {fields}".rstrip()


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ConsoleFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # uvicorn installs its own handlers; route its logs through ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True
