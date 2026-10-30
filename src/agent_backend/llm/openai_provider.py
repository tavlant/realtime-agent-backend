"""OpenAI Chat Completions adapter (streaming + function/tool calling)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agent_backend.core.retry import RetryableError
from agent_backend.llm.base import (
    ChatMessage,
    LLMEvent,
    LLMProvider,
    StreamFinished,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolChoice,
    ToolSpec,
    Usage,
)

logger = logging.getLogger(__name__)


def to_openai_messages(messages: Sequence[ChatMessage], system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def to_openai_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def _parse_arguments(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Surfaced to the model as a tool error by the registry's validation.
        return {"__invalid_json__": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        max_output_tokens: int = 4096,
        client: Any | None = None,
    ) -> None:
        if client is None:
            import openai

            client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
        self._client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
    ) -> AsyncIterator[LLMEvent]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages, system),
            "max_completion_tokens": self.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = tool_choice

        try:
            response = await self._client.chat.completions.create(**kwargs)
            pending: dict[int, dict[str, str]] = {}
            finish_reason = "stop"
            usage = Usage()
            async for chunk in response:
                if getattr(chunk, "usage", None):
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is not None and delta.content:
                    yield TextDelta(text=delta.content)
                for tc in (delta.tool_calls or []) if delta is not None else []:
                    slot = pending.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function is not None:
                        slot["name"] += tc.function.name or ""
                        slot["arguments"] += tc.function.arguments or ""
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        except Exception as exc:
            translated = _translate_error(exc)
            if translated is exc:
                raise
            raise translated from exc

        for index in sorted(pending):
            slot = pending[index]
            yield ToolCallRequested(
                call=ToolCall(slot["id"], slot["name"], _parse_arguments(slot["arguments"]))
            )
        stop = (
            "tool_use" if pending else ("max_tokens" if finish_reason == "length" else "end_turn")
        )
        yield StreamFinished(stop_reason=stop, usage=usage)

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()


def _translate_error(exc: Exception) -> Exception:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return exc
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return RetryableError(str(exc))
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return RetryableError(str(exc))
    return exc
