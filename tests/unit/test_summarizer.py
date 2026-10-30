from collections.abc import AsyncIterator, Sequence

import pytest

from agent_backend.container import Container
from agent_backend.core.memory import ConversationMemory
from agent_backend.core.retry import RetryPolicy
from agent_backend.core.summarizer import Summarizer, render_transcript
from agent_backend.db.repository import StoredMessage
from agent_backend.llm.base import ChatMessage, LLMEvent, LLMProvider, ToolCall
from agent_backend.llm.fake import FakeLLM


def turn(i: int, with_tool: bool = False) -> list[ChatMessage]:
    messages = [ChatMessage("user", f"question {i} " + "lorem ipsum " * 20)]
    if with_tool:
        call = ToolCall(f"c{i}", "calculator", {"expression": "1+1"})
        messages += [
            ChatMessage("assistant", "", tool_calls=(call,)),
            ChatMessage("tool", "2", tool_call_id=call.id, name="calculator"),
        ]
    messages.append(ChatMessage("assistant", f"answer {i} " + "dolor sit " * 20))
    return messages


def stored(messages: Sequence[ChatMessage]) -> list[StoredMessage]:
    return [StoredMessage(seq, m) for seq, m in enumerate(messages, start=1)]


def test_render_transcript_labels_roles_and_tools() -> None:
    text = render_transcript(turn(1, with_tool=True))
    assert "User: question 1" in text
    assert "Assistant [called calculator({'expression': '1+1'})]" in text
    assert "Tool result (calculator): 2" in text


async def test_summarizer_merges_previous_summary_and_new_messages() -> None:
    summarizer = Summarizer(FakeLLM())
    summary = await summarizer.summarize("User is called Ada.", turn(7))
    assert "User is called Ada." in summary
    assert "question 7" in summary
    assert "answer 7" in summary


async def test_summarizer_caps_length() -> None:
    summarizer = Summarizer(FakeLLM(), max_chars=100)
    summary = await summarizer.summarize(None, turn(1) + turn(2))
    assert len(summary) <= 104


class FailingLLM(LLMProvider):
    name = "failing"

    async def stream(
        self, messages, *, system=None, tools=(), tool_choice="auto"
    ) -> AsyncIterator[LLMEvent]:
        raise ValueError("model exploded")
        yield  # pragma: no cover


def make_memory(budget: int = 300, min_recent: int = 2) -> ConversationMemory:
    return ConversationMemory(
        redis=None,  # type: ignore[arg-type]  # choose_cut does not touch storage
        repo=None,  # type: ignore[arg-type]
        summarizer=Summarizer(FakeLLM()),
        token_budget=budget,
        summary_target_ratio=0.5,
        min_recent_messages=min_recent,
    )


def test_cut_points_are_only_before_user_messages() -> None:
    messages = stored([m for i in range(6) for m in turn(i, with_tool=i % 2 == 0)])
    memory = make_memory()
    cut = memory.choose_cut(messages)
    assert cut > 0
    assert messages[cut].message.role == "user"
    # A tool call is never separated from its result.
    kept_roles = [m.message.role for m in messages[cut:]]
    assert kept_roles[0] == "user"


def test_cut_keeps_min_recent_messages() -> None:
    messages = stored([m for i in range(3) for m in turn(i)])
    memory = make_memory(budget=10, min_recent=4)
    cut = memory.choose_cut(messages)
    assert len(messages) - cut >= 4


def test_no_cut_when_only_one_turn() -> None:
    memory = make_memory(budget=10, min_recent=1)
    assert memory.choose_cut(stored(turn(0))) == 0


async def test_maybe_summarize_folds_old_turns(container: Container) -> None:
    container.memory.token_budget = 200
    container.memory.min_recent_messages = 2
    conversation = await container.repo.create()
    for i in range(5):
        await container.memory.commit_turn(conversation.id, f"t{i}", turn(i))

    assert await container.memory.maybe_summarize(conversation.id)

    refreshed = await container.repo.require(conversation.id)
    assert refreshed.summary
    assert "question 0" in refreshed.summary
    assert refreshed.summarized_upto_seq > 0
    state = await container.memory.load(conversation.id)
    assert state.from_cache
    assert state.summary == refreshed.summary
    assert state.messages[0].seq == refreshed.summarized_upto_seq + 1
    assert state.messages[0].message.role == "user"
    # Postgres still holds every message: summarisation never deletes history.
    assert len(await container.repo.history(conversation.id, limit=500)) == 10


async def test_summarisation_failure_keeps_full_context(container: Container) -> None:
    container.memory.token_budget = 100
    container.memory.summarizer = Summarizer(FailingLLM(), RetryPolicy(max_attempts=1))
    conversation = await container.repo.create()
    for i in range(4):
        await container.memory.commit_turn(conversation.id, f"t{i}", turn(i))
    assert not await container.memory.maybe_summarize(conversation.id)
    state = await container.memory.load(conversation.id)
    assert state.summary is None
    assert len(state.messages) == 8


@pytest.mark.parametrize("cached", [True, False])
async def test_load_rebuilds_from_database_on_cache_miss(
    container: Container, cached: bool
) -> None:
    conversation = await container.repo.create()
    await container.memory.load(conversation.id)  # warm the cache
    await container.memory.commit_turn(conversation.id, "t1", turn(1))
    if not cached:
        await container.memory.invalidate(conversation.id)
    state = await container.memory.load(conversation.id)
    assert state.from_cache is cached
    assert [m.seq for m in state.messages] == [1, 2]
    assert state.messages[0].message.content.startswith("question 1")
