"""Concurrency: many simultaneous conversations, and contention on one conversation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from agent_backend.config import Settings
from agent_backend.container import Container
from agent_backend.llm.base import ChatMessage, LLMEvent, ToolChoice, ToolSpec
from agent_backend.llm.fake import FakeLLM
from tests.conftest import collect_turn

CONVERSATIONS = 50
TURNS = 3


def test_fifty_simultaneous_websocket_conversations(
    client_factory: Callable[..., TestClient],
) -> None:
    """50 sockets stream concurrently; every history is complete, ordered and unmixed."""
    client = client_factory(fake_llm_token_delay_ms=2)
    ids = [client.post("/conversations").json()["id"] for _ in range(CONVERSATIONS)]

    def run(index: int, conversation_id: str) -> list[str]:
        replies = []
        with client.websocket_connect(f"/ws/conversations/{conversation_id}") as ws:
            ws.receive_json()
            for turn in range(TURNS):
                ws.send_json({"type": "user_message", "content": f"conv {index} turn {turn}"})
                events = collect_turn(ws)
                assert events[-1]["type"] == "turn_completed", events[-1]
                streamed = "".join(e["text"] for e in events if e["type"] == "token")
                assert streamed == events[-1]["content"]
                replies.append(streamed)
        return replies

    with ThreadPoolExecutor(max_workers=CONVERSATIONS) as pool:
        results = list(pool.map(run, range(CONVERSATIONS), ids))

    for index, (conversation_id, replies) in enumerate(zip(ids, results, strict=True)):
        assert replies == [f"You said: conv {index} turn {t}" for t in range(TURNS)]
        items = client.get(f"/conversations/{conversation_id}/messages").json()["items"]
        expected = []
        for t in range(TURNS):
            expected += [f"conv {index} turn {t}", f"You said: conv {index} turn {t}"]
        assert [m["content"] for m in items] == expected
        assert [m["seq"] for m in items] == list(range(1, 2 * TURNS + 1))


class ConcurrencyProbeLLM(FakeLLM):
    """FakeLLM that records how many streams run at once for each conversation."""

    def __init__(self) -> None:
        super().__init__(token_delay_s=0.001)
        self.active = 0
        self.max_active = 0

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
    ) -> AsyncIterator[LLMEvent]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            async for event in super().stream(
                messages, system=system, tools=tools, tool_choice=tool_choice
            ):
                yield event
        finally:
            self.active -= 1


async def test_concurrent_turns_on_one_conversation_are_serialised(settings: Settings) -> None:
    from fakeredis import FakeAsyncRedis

    probe = ConcurrencyProbeLLM()
    container = Container.build(settings, redis=FakeAsyncRedis(decode_responses=True), llm=probe)
    await container.startup()
    try:
        conversation = await container.repo.create()
        texts = [f"message {i}" for i in range(15)]
        await asyncio.gather(*(container.agent.run_turn(conversation.id, t) for t in texts))

        assert probe.max_active == 1, "turns of one conversation must never overlap"
        rows = await container.repo.history(conversation.id, limit=100)
        assert [r.seq for r in rows] == list(range(1, 31))
        # Strict user/assistant alternation, and each reply answers its own question.
        for user, assistant in zip(rows[::2], rows[1::2], strict=True):
            assert user.role == "user"
            assert assistant.role == "assistant"
            assert assistant.content == f"You said: {user.content}"
            assert user.turn_id == assistant.turn_id
        assert sorted(r.content for r in rows[::2]) == sorted(texts)
    finally:
        await container.shutdown()


async def test_parallel_conversations_at_service_level(container: Container) -> None:
    """Different conversations proceed in parallel without blocking or mixing."""
    conversations = [await container.repo.create() for _ in range(CONVERSATIONS)]

    async def drive(i: int, conversation_id: str) -> None:
        for t in range(TURNS):
            result = await container.agent.run_turn(conversation_id, f"c{i} t{t}")
            assert result.reply == f"You said: c{i} t{t}"

    await asyncio.gather(*(drive(i, c.id) for i, c in enumerate(conversations)))
    for i, conversation in enumerate(conversations):
        state = await container.memory.load(conversation.id)
        assert [m.message.content for m in state.messages] == [
            text for t in range(TURNS) for text in (f"c{i} t{t}", f"You said: c{i} t{t}")
        ]
