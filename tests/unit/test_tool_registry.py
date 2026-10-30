import asyncio
import time

import pytest
from pydantic import BaseModel, Field

from agent_backend.llm.base import ToolCall
from agent_backend.tools import build_default_registry
from agent_backend.tools.base import MAX_RESULT_CHARS, Tool, ToolError, ToolRegistry


class EchoArgs(BaseModel):
    text: str = Field(..., min_length=1)
    times: int = Field(default=1, ge=1, le=3)


class EchoTool(Tool[EchoArgs]):
    name = "echo"
    description = "Echo text."
    args_model = EchoArgs

    async def run(self, args: EchoArgs) -> str:
        return args.text * args.times


class SleepArgs(BaseModel):
    seconds: float


class SleepTool(Tool[SleepArgs]):
    name = "sleep"
    description = "Sleep."
    args_model = SleepArgs
    timeout_s = 0.2

    async def run(self, args: SleepArgs) -> str:
        await asyncio.sleep(args.seconds)
        return "slept"


class BrokenTool(Tool[EchoArgs]):
    name = "broken"
    description = "Always fails."
    args_model = EchoArgs

    async def run(self, args: EchoArgs) -> str:
        if args.text == "expected":
            raise ToolError("a readable failure")
        raise RuntimeError("secret internal detail")


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([EchoTool(), SleepTool(), BrokenTool()])


def test_specs_are_sorted_and_have_json_schema(registry: ToolRegistry) -> None:
    specs = registry.specs()
    assert [s.name for s in specs] == ["broken", "echo", "sleep"]
    echo = next(s for s in specs if s.name == "echo")
    assert echo.parameters["type"] == "object"
    assert echo.parameters["required"] == ["text"]
    assert "title" not in echo.parameters


def test_duplicate_registration_is_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())


async def test_executes_valid_call(registry: ToolRegistry) -> None:
    result = await registry.execute(ToolCall("c1", "echo", {"text": "hi", "times": 2}))
    assert (result.call_id, result.name, result.content, result.is_error) == (
        "c1",
        "echo",
        "hihi",
        False,
    )


async def test_unknown_tool(registry: ToolRegistry) -> None:
    result = await registry.execute(ToolCall("c1", "nope", {}))
    assert result.is_error
    assert "unknown tool" in result.content


async def test_invalid_arguments_are_reported_not_raised(registry: ToolRegistry) -> None:
    result = await registry.execute(ToolCall("c1", "echo", {"times": 9}))
    assert result.is_error
    assert "text" in result.content
    assert "times" in result.content


async def test_timeout(registry: ToolRegistry) -> None:
    result = await registry.execute(ToolCall("c1", "sleep", {"seconds": 5}))
    assert result.is_error
    assert "timed out" in result.content


async def test_tool_errors_are_readable_and_crashes_are_masked(registry: ToolRegistry) -> None:
    readable = await registry.execute(ToolCall("c1", "broken", {"text": "expected"}))
    assert readable.is_error
    assert readable.content == "a readable failure"
    crashed = await registry.execute(ToolCall("c2", "broken", {"text": "boom"}))
    assert crashed.is_error
    assert crashed.content == "internal tool error"
    assert "secret" not in crashed.content


async def test_long_results_are_truncated(registry: ToolRegistry) -> None:
    result = await registry.execute(ToolCall("c1", "echo", {"text": "x" * 3000, "times": 2}))
    assert len(result.content) < MAX_RESULT_CHARS + 50
    assert result.content.endswith("[truncated]")


async def test_execute_many_runs_concurrently_and_keeps_order(registry: ToolRegistry) -> None:
    calls = [
        ToolCall("a", "sleep", {"seconds": 0.1}),
        ToolCall("b", "echo", {"text": "fast"}),
        ToolCall("c", "sleep", {"seconds": 0.1}),
    ]
    started = time.perf_counter()
    results = await registry.execute_many(calls)
    elapsed = time.perf_counter() - started
    assert [r.call_id for r in results] == ["a", "b", "c"]
    assert elapsed < 0.19  # two 0.1s sleeps overlapped


async def test_default_registry_tools() -> None:
    registry = build_default_registry(knowledge_base=None)
    assert registry.names() == ["calculator", "current_time"]
    now = await registry.execute(ToolCall("t", "current_time", {"timezone": "Europe/Berlin"}))
    assert not now.is_error
    assert "Europe/Berlin" in now.content
    bad = await registry.execute(ToolCall("t", "current_time", {"timezone": "Mars/Olympus"}))
    assert bad.is_error
