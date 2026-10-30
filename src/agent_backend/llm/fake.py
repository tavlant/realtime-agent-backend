"""A deterministic, offline LLM.

FakeLLM makes the whole system runnable and testable without API keys or network access.
It follows a few transparent rules so that tool calling and streaming are exercised end
to end:

* a user message containing an arithmetic expression  -> calls `calculator`
* a user message mentioning "time" or "date"           -> calls `current_time`
* a user message starting with "search" / "docs:"      -> calls `search_knowledge_base`
* after tool results                                   -> answers using those results
* a summarisation request                              -> returns a compact digest
* anything else                                        -> "You said: <message>"
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Sequence

from agent_backend.core.tokens import estimate_tokens
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

_EXPRESSION = re.compile(
    r"(?:sqrt|abs|round|log|sin|cos|tan|exp|min|max|pi|[\d.]+)"
    r"[\d\s.+\-*/%()^,a-z]*[\d)]",
)
_OPERATOR = re.compile(r"[+\-*/%^]|\b(?:sqrt|log|sin|cos|tan|exp|abs|round|min|max)\(")
_SEARCH = re.compile(r"^\s*(?:search(?:\s+the\s+docs)?(?:\s+for)?|docs:)\s*(.+)$", re.I)
_TIME = re.compile(r"\b(time|date|today|clock)\b", re.I)
_TOKEN = re.compile(r"\S+\s*|\s+")

SUMMARY_HINT = "rolling summary"


def _find_expression(text: str) -> str | None:
    for match in _EXPRESSION.finditer(text.lower()):
        candidate = match.group(0).strip()
        if _OPERATOR.search(candidate) and any(ch.isdigit() for ch in candidate):
            return candidate.replace("^", "**")
    return None


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, token_delay_s: float = 0.0) -> None:
        self.token_delay_s = token_delay_s

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
    ) -> AsyncIterator[LLMEvent]:
        tool_names = {t.name for t in tools} if tool_choice == "auto" else set()
        input_tokens = estimate_tokens(system or "") + sum(
            estimate_tokens(m.content) for m in messages
        )

        if system and SUMMARY_HINT in system.lower():
            async for event in self._emit(self._summarise(messages), input_tokens):
                yield event
            return

        last = messages[-1] if messages else ChatMessage(role="user", content="")

        if last.role == "user":
            call = self._plan_tool_call(last.content, tool_names, call_index=len(messages))
            if call is not None:
                yield ToolCallRequested(call=call)
                yield StreamFinished(
                    stop_reason="tool_use", usage=Usage(input_tokens=input_tokens, output_tokens=8)
                )
                return
            reply = f"You said: {last.content}"
        elif last.role == "tool":
            reply = self._answer_from_tools(messages)
        else:
            reply = "How can I help?"

        async for event in self._emit(reply, input_tokens):
            yield event

    async def _emit(self, text: str, input_tokens: int) -> AsyncIterator[LLMEvent]:
        for token in _TOKEN.findall(text):
            if self.token_delay_s:
                await asyncio.sleep(self.token_delay_s)
            yield TextDelta(text=token)
        yield StreamFinished(
            stop_reason="end_turn",
            usage=Usage(input_tokens=input_tokens, output_tokens=estimate_tokens(text)),
        )

    @staticmethod
    def _plan_tool_call(text: str, tool_names: set[str], call_index: int) -> ToolCall | None:
        call_id = f"call_{call_index}"
        if "search_knowledge_base" in tool_names and (m := _SEARCH.match(text)):
            return ToolCall(call_id, "search_knowledge_base", {"query": m.group(1).strip()})
        if "calculator" in tool_names and (expr := _find_expression(text)):
            return ToolCall(call_id, "calculator", {"expression": expr})
        if "current_time" in tool_names and _TIME.search(text):
            return ToolCall(call_id, "current_time", {"timezone": "UTC"})
        return None

    @staticmethod
    def _answer_from_tools(messages: Sequence[ChatMessage]) -> str:
        results: list[ChatMessage] = []
        for message in reversed(messages):
            if message.role != "tool":
                break
            results.append(message)
        parts = []
        for result in reversed(results):
            if result.is_error:
                parts.append(f"The {result.name} tool failed: {result.content}")
            elif result.name == "calculator":
                parts.append(f"The answer is {result.content}.")
            elif result.name == "current_time":
                parts.append(f"The current time is {result.content}.")
            else:
                parts.append(f"From the knowledge base: {result.content}")
        return " ".join(parts) or "Done."

    @staticmethod
    def _summarise(messages: Sequence[ChatMessage]) -> str:
        """Deterministic digest: previous summary + one clipped line per new message."""
        prompt = messages[-1].content if messages else ""
        previous, body = "", prompt
        if "Existing summary:" in prompt and "New messages:" in prompt:
            head, body = prompt.split("New messages:", 1)
            previous = head.split("Existing summary:", 1)[1].strip()
            body = body.split("Return only", 1)[0]
        lines = [line.strip()[:100] for line in body.splitlines() if line.strip()]
        parts = [] if previous in ("", "(none yet)") else [previous]
        parts.append(" | ".join(lines))
        # Keep the most recent part if the digest grows too long.
        return " || ".join(parts)[-1200:]
