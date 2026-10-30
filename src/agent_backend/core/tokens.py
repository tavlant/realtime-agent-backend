"""Cheap, provider-independent token estimation.

Exact counts differ per tokenizer; for context budgeting a conservative estimate is enough
and avoids a network round-trip or a heavy tokenizer dependency on every turn.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_backend.llm.base import ChatMessage

_PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # ~4 characters per token for English, rounded up.
    return (len(text) + 3) // 4


def estimate_message_tokens(message: ChatMessage) -> int:
    total = _PER_MESSAGE_OVERHEAD + estimate_tokens(message.content)
    for call in message.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(str(call.arguments))
    return total


def estimate_messages_tokens(messages: Iterable[ChatMessage]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)
