"""Rolling summarisation of older conversation turns."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from agent_backend.core.retry import RetryPolicy, retry_async
from agent_backend.llm.base import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = (
    "You maintain the rolling summary of a conversation between a user and an AI assistant. "
    "Merge the existing summary with the new messages into one updated summary. Keep facts "
    "about the user, names, numbers, decisions, tool results that matter later, and open "
    "questions. Drop pleasantries. Write plain prose, at most 200 words."
)


def render_transcript(messages: Sequence[ChatMessage], max_chars_per_message: int = 1500) -> str:
    lines = []
    for m in messages:
        if m.role == "tool":
            label = f"Tool result ({m.name}{', error' if m.is_error else ''})"
        elif m.role == "assistant" and m.tool_calls:
            calls = ", ".join(f"{c.name}({c.arguments})" for c in m.tool_calls)
            label = f"Assistant [called {calls}]"
        else:
            label = m.role.capitalize()
        content = (
            m.content
            if len(m.content) <= max_chars_per_message
            else (m.content[:max_chars_per_message] + " ...")
        )
        lines.append(f"{label}: {content}".rstrip())
    return "\n".join(lines)


class Summarizer:
    def __init__(
        self, llm: LLMProvider, retry_policy: RetryPolicy | None = None, max_chars: int = 2400
    ) -> None:
        self.llm = llm
        self.retry_policy = retry_policy or RetryPolicy()
        self.max_chars = max_chars

    async def summarize(self, previous: str | None, messages: Sequence[ChatMessage]) -> str:
        prompt = (
            f"Existing summary:\n{previous or '(none yet)'}\n\n"
            f"New messages:\n{render_transcript(messages)}\n\n"
            "Return only the updated summary."
        )

        async def call() -> str:
            return await self.llm.complete(
                [ChatMessage(role="user", content=prompt)], system=SUMMARY_SYSTEM_PROMPT
            )

        summary = (await retry_async(call, self.retry_policy, op="summarize")).strip()
        if len(summary) > self.max_chars:
            summary = summary[: self.max_chars].rsplit(" ", 1)[0] + " ..."
        return summary
