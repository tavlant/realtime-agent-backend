"""Provider-neutral chat types and the LLMProvider interface.

The rest of the application only speaks these types. Each adapter translates them to and
from its vendor's wire format, so switching providers is a config change, not a refactor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]
ToolChoice = Literal["auto", "none"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None  # tool name, for role == "tool"
    is_error: bool = False  # tool result that represents a failure

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.is_error:
            data["is_error"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        return cls(
            role=data["role"],
            content=data.get("content") or "",
            tool_calls=tuple(
                ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments") or {})
                for c in data.get("tool_calls") or ()
            ),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            is_error=bool(data.get("is_error", False)),
        )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object


# --- streaming events ------------------------------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ToolCallRequested:
    call: ToolCall
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class StreamFinished:
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    type: Literal["finish"] = "finish"


LLMEvent = TextDelta | ToolCallRequested | StreamFinished


class LLMProvider(ABC):
    """A chat model that streams text and may request tool calls."""

    name: str = "base"

    @abstractmethod
    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
    ) -> AsyncIterator[LLMEvent]:
        """Stream a single assistant turn.

        `tool_choice="none"` keeps the tool definitions visible (history may contain tool
        calls) but asks the model to answer in text.

        Yields TextDelta events as tokens arrive, a ToolCallRequested per tool call the
        model wants to make, and finally exactly one StreamFinished.
        Transient failures must be raised as `RetryableError`.
        """

    async def complete(self, messages: Sequence[ChatMessage], *, system: str | None = None) -> str:
        """Non-streaming convenience used for background work such as summarisation."""
        parts: list[str] = []
        async for event in self.stream(messages, system=system):
            if isinstance(event, TextDelta):
                parts.append(event.text)
        return "".join(parts)

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""
