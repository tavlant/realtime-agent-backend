"""Provider adapters: wire-format translation and streaming assembly (no network)."""

from types import SimpleNamespace as NS
from typing import Any

from agent_backend.config import Settings
from agent_backend.llm.anthropic_provider import to_anthropic_messages, to_anthropic_tools
from agent_backend.llm.base import (
    ChatMessage,
    StreamFinished,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
)
from agent_backend.llm.fake import FakeLLM
from agent_backend.llm.openai_provider import OpenAIProvider, to_openai_messages

CALL_A = ToolCall("call_a", "calculator", {"expression": "2+2"})
CALL_B = ToolCall("call_b", "current_time", {"timezone": "UTC"})
TOOLS = [ToolSpec("calculator", "math", {"type": "object", "properties": {}})]

CONVERSATION = [
    ChatMessage("user", "what is 2+2 and the time?"),
    ChatMessage("assistant", "Let me check.", tool_calls=(CALL_A, CALL_B)),
    ChatMessage("tool", "4", tool_call_id="call_a", name="calculator"),
    ChatMessage("tool", "boom", tool_call_id="call_b", name="current_time", is_error=True),
    ChatMessage("assistant", "4, and the clock failed."),
]


def test_openai_message_translation() -> None:
    out = to_openai_messages(CONVERSATION, system="be brief")
    assert out[0] == {"role": "system", "content": "be brief"}
    assistant = out[2]
    assert assistant["tool_calls"][0]["function"] == {
        "name": "calculator",
        "arguments": '{"expression": "2+2"}',
    }
    assert out[3] == {"role": "tool", "tool_call_id": "call_a", "content": "4"}
    assert out[-1] == {"role": "assistant", "content": "4, and the clock failed."}


def test_anthropic_message_translation_merges_parallel_tool_results() -> None:
    out = to_anthropic_messages(CONVERSATION)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
    assistant_blocks = out[1]["content"]
    assert assistant_blocks[0] == {"type": "text", "text": "Let me check."}
    assert assistant_blocks[1] == {
        "type": "tool_use",
        "id": "call_a",
        "name": "calculator",
        "input": {"expression": "2+2"},
    }
    results = out[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["call_a", "call_b"]
    assert results[1]["is_error"] is True
    assert "is_error" not in results[0]


def test_anthropic_tool_schema() -> None:
    [tool] = to_anthropic_tools(TOOLS)
    assert tool["name"] == "calculator"
    assert tool["input_schema"] == TOOLS[0].parameters


class FakeOpenAIStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> "FakeOpenAIStream":
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


def chunk(content: str | None = None, tool_calls: Any = None, finish: str | None = None) -> NS:
    return NS(
        usage=None,
        choices=[NS(delta=NS(content=content, tool_calls=tool_calls), finish_reason=finish)],
    )


def tc(index: int, id: str | None = None, name: str | None = None, args: str = "") -> NS:
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


async def test_openai_adapter_streams_text_and_assembles_tool_call_fragments() -> None:
    chunks = [
        chunk(content="Sure, "),
        chunk(content="calculating."),
        chunk(tool_calls=[tc(0, "call_1", "calculator", '{"expr')]),
        chunk(tool_calls=[tc(0, args='ession": "6*7"}')]),
        chunk(finish="tool_calls"),
        NS(usage=NS(prompt_tokens=11, completion_tokens=5), choices=[]),
    ]
    captured: dict[str, Any] = {}

    async def create(**kwargs: Any) -> FakeOpenAIStream:
        captured.update(kwargs)
        return FakeOpenAIStream(chunks)

    client = NS(chat=NS(completions=NS(create=create)))
    provider = OpenAIProvider(api_key=None, model="test-model", client=client)
    events = [e async for e in provider.stream(CONVERSATION[:1], system="sys", tools=TOOLS)]

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Sure, ", "calculating."]
    [call] = [e.call for e in events if isinstance(e, ToolCallRequested)]
    assert call == ToolCall("call_1", "calculator", {"expression": "6*7"})
    finish = events[-1]
    assert isinstance(finish, StreamFinished)
    assert finish.stop_reason == "tool_use"
    assert (finish.usage.input_tokens, finish.usage.output_tokens) == (11, 5)
    assert captured["stream"] is True
    assert captured["tools"][0]["function"]["name"] == "calculator"


async def test_fake_llm_rules() -> None:
    llm = FakeLLM()
    tools = [ToolSpec(n, "", {}) for n in ("calculator", "current_time", "search_knowledge_base")]

    async def first_call(text: str) -> ToolCall | None:
        events = [e async for e in llm.stream([ChatMessage("user", text)], tools=tools)]
        calls = [e.call for e in events if isinstance(e, ToolCallRequested)]
        return calls[0] if calls else None

    assert (await first_call("What is 12 * (3 + 4)?")).arguments == {"expression": "12 * (3 + 4)"}
    assert (await first_call("what time is it?")).name == "current_time"
    assert (await first_call("search the docs for redis locks")).arguments == {
        "query": "redis locks"
    }
    assert await first_call("hello there") is None
    assert await llm.complete([ChatMessage("user", "hello there")]) == "You said: hello there"


async def test_fake_llm_respects_tool_choice_none() -> None:
    llm = FakeLLM()
    events = [
        e async for e in llm.stream([ChatMessage("user", "1 + 1")], tools=TOOLS, tool_choice="none")
    ]
    assert not any(isinstance(e, ToolCallRequested) for e in events)


def test_provider_auto_selection() -> None:
    assert Settings(llm_provider="auto").resolved_llm_provider() in {"fake", "openai", "anthropic"}
    assert (
        Settings(
            llm_provider="auto", anthropic_api_key="k", openai_api_key=None
        ).resolved_llm_provider()
        == "anthropic"
    )
    assert (
        Settings(
            llm_provider="auto", anthropic_api_key=None, openai_api_key="k"
        ).resolved_llm_provider()
        == "openai"
    )
    assert (
        Settings(
            llm_provider="auto", anthropic_api_key=None, openai_api_key=None
        ).resolved_llm_provider()
        == "fake"
    )
