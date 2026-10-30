"""Anthropic Messages API adapter (streaming + tool use)."""

from __future__ import annotations

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

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def to_anthropic_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Translate neutral messages to Anthropic content blocks.

    Tool results become `tool_result` blocks inside a user turn, and consecutive results
    are merged into a single user message (parallel tool calls must be answered together).
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
                **({"is_error": True} if m.is_error else {}),
            }
            prev = out[-1] if out else None
            if (
                prev is not None
                and prev["role"] == "user"
                and isinstance(prev["content"], list)
                and all(b.get("type") == "tool_result" for b in prev["content"])
            ):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == "assistant" and m.tool_calls:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in m.tool_calls
            )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def to_anthropic_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
            # Tool inputs stream as generated; the registry validates them before running.
            "eager_input_streaming": True,
        }
        for t in tools
    ]


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        max_output_tokens: int = 16000,
        effort: str = "medium",
        refusal_fallbacks: bool = True,
        client: Any | None = None,
    ) -> None:
        if client is None:
            import anthropic

            # Retries are handled by our own policy so they can respect stream semantics.
            client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)
        self._client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.effort = effort
        self.refusal_fallbacks = refusal_fallbacks

    def _stream_manager(self, kwargs: dict[str, Any]) -> Any:
        if self.refusal_fallbacks:
            return self._client.beta.messages.stream(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs
            )
        return self._client.messages.stream(**kwargs)

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
            "max_tokens": self.max_output_tokens,
            "messages": to_anthropic_messages(messages),
            "output_config": {"effort": self.effort},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
            if tool_choice == "none":
                kwargs["tool_choice"] = {"type": "none"}

        try:
            async with self._stream_manager(kwargs) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield TextDelta(text=event.text)
                final = await stream.get_final_message()
        except Exception as exc:
            translated = _translate_error(exc)
            if translated is exc:
                raise
            raise translated from exc

        stop_reason = final.stop_reason or "end_turn"
        tool_uses = [b for b in final.content if b.type == "tool_use"]
        if stop_reason == "tool_use":
            for block in tool_uses:
                arguments = block.input if isinstance(block.input, dict) else {}
                yield ToolCallRequested(call=ToolCall(block.id, block.name, arguments))
        elif tool_uses:
            # max_tokens / refusal can cut a tool call off mid-input: never run it.
            logger.warning("dropping incomplete tool calls", extra={"stop_reason": stop_reason})
        usage = Usage(
            input_tokens=getattr(final.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(final.usage, "output_tokens", 0) or 0,
        )
        yield StreamFinished(stop_reason=stop_reason, usage=usage)

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()


def _translate_error(exc: Exception) -> Exception:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return exc
    if isinstance(
        exc, (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError)
    ):
        return RetryableError(str(exc))
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return RetryableError(str(exc))
    return exc
