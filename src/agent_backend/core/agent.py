"""The agent turn loop: stream tokens, execute tool calls, feed results back, persist.

One turn, end to end:

1. take the per-conversation Redis lock (turns never interleave);
2. load context (rolling summary + recent messages) from Redis, falling back to Postgres;
3. stream the model's answer to the client token by token;
4. if the model requested tools, run them concurrently, append the results and loop
   (bounded by `max_tool_rounds`; the final round offers no tools, forcing an answer);
5. commit the whole turn (user, tool and assistant messages) atomically to Postgres,
   then to Redis - history only ever contains complete turns;
6. tell the client the turn is complete, then summarise old turns if the context is
   over budget (still inside the lock, but after the user already has the answer).
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agent_backend.core.locks import ConversationLocks
from agent_backend.core.memory import ContextState, ConversationMemory
from agent_backend.core.retry import RetryPolicy, stream_with_retry
from agent_backend.db.repository import StoredMessage
from agent_backend.llm.base import (
    ChatMessage,
    LLMProvider,
    StreamFinished,
    TextDelta,
    ToolCall,
    ToolCallRequested,
)
from agent_backend.log import bind_context
from agent_backend.telemetry import span
from agent_backend.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

EventSink = Callable[[dict[str, Any]], Awaitable[None]]

TOOL_LIMIT_NOTICE = "I stopped after reaching the tool-call limit for this turn."


async def _discard(_event: dict[str, Any]) -> None:
    return None


@dataclass
class TurnResult:
    turn_id: str
    reply: str
    messages: list[ChatMessage]
    seqs: list[int]
    input_tokens: int = 0
    output_tokens: int = 0
    tool_rounds: int = 0
    summarized: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    system_prompt: str
    max_tool_rounds: int = 5
    first_token_timeout_s: float = 30.0
    idle_timeout_s: float = 30.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


class AgentRunner:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        memory: ConversationMemory,
        locks: ConversationLocks,
        config: AgentConfig,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.locks = locks
        self.config = config

    def _system_prompt(self, summary: str | None) -> str:
        if not summary:
            return self.config.system_prompt
        return (
            f"{self.config.system_prompt}\n\n"
            f"Summary of the earlier part of this conversation:\n{summary}"
        )

    async def run_turn(
        self,
        conversation_id: str,
        user_text: str,
        emit: EventSink = _discard,
        *,
        turn_id: str | None = None,
    ) -> TurnResult:
        turn_id = turn_id or uuid.uuid4().hex
        with (
            bind_context(conversation_id=conversation_id, turn_id=turn_id),
            span("agent.turn", conversation_id=conversation_id, turn_id=turn_id),
        ):
            started = time.perf_counter()
            async with self.locks.for_conversation(conversation_id) as lock:
                lock_wait_ms = (time.perf_counter() - started) * 1000
                await emit({"type": "turn_started", "turn_id": turn_id})
                result, state = await self._run_locked(conversation_id, user_text, turn_id, emit)
                result.timings_ms["lock_wait"] = round(lock_wait_ms, 2)

                lock.ensure_held()
                result.seqs = await self.memory.commit_turn(
                    conversation_id, turn_id, result.messages
                )
                result.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
                await emit(
                    {
                        "type": "turn_completed",
                        "turn_id": turn_id,
                        "content": result.reply,
                        "tool_rounds": result.tool_rounds,
                        "usage": {
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                        },
                    }
                )
                logger.info(
                    "turn completed",
                    extra={
                        "tool_rounds": result.tool_rounds,
                        "messages": len(result.messages),
                        **{f"{k}_ms": v for k, v in result.timings_ms.items()},
                    },
                )
                after_turn = ContextState(
                    state.summary,
                    state.summarized_upto_seq,
                    [
                        *state.messages,
                        *map(StoredMessage, result.seqs, result.messages),
                    ],
                )
                result.summarized = await self.memory.maybe_summarize(conversation_id, after_turn)
            return result

    async def _run_locked(
        self, conversation_id: str, user_text: str, turn_id: str, emit: EventSink
    ) -> tuple[TurnResult, ContextState]:
        state = await self.memory.load(conversation_id)
        system = self._system_prompt(state.summary)
        history = state.chat_messages
        new_messages: list[ChatMessage] = [ChatMessage(role="user", content=user_text)]
        result = TurnResult(turn_id=turn_id, reply="", messages=new_messages, seqs=[])
        specs = self.tools.specs()
        first_token_at: float | None = None
        t0 = time.perf_counter()

        for round_index in range(self.config.max_tool_rounds + 1):
            # On the last round, tools stay defined but are disabled: answer in text.
            final_round = round_index == self.config.max_tool_rounds
            factory = functools.partial(
                self.llm.stream,
                [*history, *new_messages],
                system=system,
                tools=specs,
                tool_choice="none" if final_round else "auto",
            )
            text_parts: list[str] = []
            calls: list[ToolCall] = []
            with span("llm.stream", provider=self.llm.name, round=round_index):
                async for event in stream_with_retry(
                    factory,
                    self.config.retry_policy,
                    first_timeout_s=self.config.first_token_timeout_s,
                    idle_timeout_s=self.config.idle_timeout_s,
                ):
                    if isinstance(event, TextDelta):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        text_parts.append(event.text)
                        await emit({"type": "token", "turn_id": turn_id, "text": event.text})
                    elif isinstance(event, ToolCallRequested):
                        calls.append(event.call)
                    elif isinstance(event, StreamFinished):
                        result.input_tokens += event.usage.input_tokens
                        result.output_tokens += event.usage.output_tokens

            text = "".join(text_parts)
            if calls and final_round:
                logger.warning("ignoring tool calls requested after the tool-round limit")
                calls = []
                text = text or TOOL_LIMIT_NOTICE
            if not calls:
                new_messages.append(ChatMessage(role="assistant", content=text))
                result.reply = text
                break

            result.tool_rounds += 1
            new_messages.append(
                ChatMessage(role="assistant", content=text, tool_calls=tuple(calls))
            )
            for call in calls:
                await emit(
                    {
                        "type": "tool_call",
                        "turn_id": turn_id,
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                )
            for tool_result in await self.tools.execute_many(calls):
                await emit(
                    {
                        "type": "tool_result",
                        "turn_id": turn_id,
                        "id": tool_result.call_id,
                        "name": tool_result.name,
                        "content": tool_result.content,
                        "is_error": tool_result.is_error,
                    }
                )
                new_messages.append(
                    ChatMessage(
                        role="tool",
                        content=tool_result.content,
                        tool_call_id=tool_result.call_id,
                        name=tool_result.name,
                        is_error=tool_result.is_error,
                    )
                )
        else:  # pragma: no cover - the tool-less final round always breaks
            new_messages.append(ChatMessage(role="assistant", content=TOOL_LIMIT_NOTICE))
            result.reply = TOOL_LIMIT_NOTICE

        if first_token_at is not None:
            result.timings_ms["first_token"] = round((first_token_at - t0) * 1000, 2)
        return result, state
