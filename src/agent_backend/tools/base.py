"""Tool abstraction and registry.

Tools declare their arguments as a Pydantic model; the registry derives the JSON Schema
the model sees, validates every call before running it, enforces a timeout, and turns
any failure into an error result the model can read and recover from. A tool can never
crash a conversation turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from agent_backend.llm.base import ToolCall, ToolSpec
from agent_backend.telemetry import span

logger = logging.getLogger(__name__)

ArgsT = TypeVar("ArgsT", bound=BaseModel)

MAX_RESULT_CHARS = 4000


class ToolError(Exception):
    """A tool-level failure whose message is safe to show to the model."""


class Tool(ABC, Generic[ArgsT]):
    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]
    timeout_s: ClassVar[float | None] = None  # falls back to the registry default

    @abstractmethod
    async def run(self, args: ArgsT) -> str: ...

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, parameters=schema)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False
    duration_ms: float = 0.0


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]] = (), default_timeout_s: float = 10.0) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        self._specs: list[ToolSpec] | None = None
        self.default_timeout_s = default_timeout_s
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        self._specs = None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        # Deterministic order keeps prompts stable (and provider-side caches warm);
        # schemas are generated once, not on every turn.
        if self._specs is None:
            self._specs = [self._tools[name].spec() for name in self.names()]
        return list(self._specs)

    async def execute(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        tool = self._tools.get(call.name)
        if tool is None:
            return self._result(call, f"unknown tool {call.name!r}", True, started)
        try:
            args = tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors()
            )
            return self._result(call, f"invalid arguments: {errors}", True, started)

        timeout = tool.timeout_s or self.default_timeout_s
        try:
            with span("tool.execute", tool=call.name):
                content = await asyncio.wait_for(tool.run(args), timeout=timeout)
        except asyncio.TimeoutError:
            return self._result(call, f"timed out after {timeout:.1f}s", True, started)
        except ToolError as exc:
            return self._result(call, str(exc), True, started)
        except Exception:
            logger.exception("tool crashed", extra={"tool": call.name})
            return self._result(call, "internal tool error", True, started)
        return self._result(call, content, False, started)

    async def execute_many(self, calls: Iterable[ToolCall]) -> list[ToolResult]:
        """Run independent tool calls concurrently, preserving order."""
        return list(await asyncio.gather(*(self.execute(c) for c in calls)))

    @staticmethod
    def _result(call: ToolCall, content: str, is_error: bool, started: float) -> ToolResult:
        if len(content) > MAX_RESULT_CHARS:
            content = content[:MAX_RESULT_CHARS] + " ...[truncated]"
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "tool executed",
            extra={"tool": call.name, "is_error": is_error, "duration_ms": duration_ms},
        )
        return ToolResult(call.id, call.name, content, is_error, duration_ms)
