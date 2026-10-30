"""Data access for conversations and messages (the durable source of truth)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent_backend.db.models import Conversation, Message, utcnow
from agent_backend.llm.base import ChatMessage


class ConversationNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class StoredMessage:
    seq: int
    message: ChatMessage


def _to_row(conversation_id: str, turn_id: str, seq: int, m: ChatMessage) -> Message:
    data = m.to_dict()
    return Message(
        conversation_id=conversation_id,
        seq=seq,
        turn_id=turn_id,
        role=m.role,
        content=m.content,
        tool_calls=data.get("tool_calls"),
        tool_call_id=m.tool_call_id,
        name=m.name,
        is_error=m.is_error,
    )


def row_to_chat(row: Message) -> ChatMessage:
    return ChatMessage.from_dict(
        {
            "role": row.role,
            "content": row.content,
            "tool_calls": row.tool_calls,
            "tool_call_id": row.tool_call_id,
            "name": row.name,
            "is_error": row.is_error,
        }
    )


class ConversationRepository:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def create(self, title: str | None = None) -> Conversation:
        async with self._sessionmaker() as session:
            conversation = Conversation(title=title)
            session.add(conversation)
            await session.commit()
            return conversation

    async def get(self, conversation_id: str) -> Conversation | None:
        async with self._sessionmaker() as session:
            return await session.get(Conversation, conversation_id)

    async def require(self, conversation_id: str) -> Conversation:
        conversation = await self.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    async def list(self, limit: int = 20, offset: int = 0) -> tuple[list[Conversation], int]:
        async with self._sessionmaker() as session:
            total = await session.scalar(select(func.count()).select_from(Conversation))
            rows = await session.scalars(
                select(Conversation)
                .order_by(Conversation.updated_at.desc(), Conversation.id)
                .limit(limit)
                .offset(offset)
            )
            return list(rows), int(total or 0)

    async def append_turn(
        self, conversation_id: str, turn_id: str, messages: Sequence[ChatMessage]
    ) -> list[int]:
        """Persist all messages of a turn atomically and return their sequence numbers.

        The counter is bumped with a single UPDATE ... RETURNING *before* any read, so the
        transaction takes its write lock first (this also avoids SQLite lock-upgrade
        deadlocks). The unique (conversation_id, seq) constraint is a second safety net
        behind the per-conversation Redis lock.
        """
        if not messages:
            return []
        async with self._sessionmaker() as session, session.begin():
            end_seq = await session.scalar(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(
                    message_count=Conversation.message_count + len(messages),
                    updated_at=utcnow(),
                )
                .returning(Conversation.message_count)
            )
            if end_seq is None:
                raise ConversationNotFoundError(conversation_id)
            start = end_seq - len(messages) + 1
            seqs = list(range(start, end_seq + 1))
            session.add_all(
                _to_row(conversation_id, turn_id, seq, m)
                for seq, m in zip(seqs, messages, strict=True)
            )
        return seqs

    async def messages_after(self, conversation_id: str, after_seq: int) -> list[StoredMessage]:
        async with self._sessionmaker() as session:
            rows = await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id, Message.seq > after_seq)
                .order_by(Message.seq)
            )
            return [StoredMessage(r.seq, row_to_chat(r)) for r in rows]

    async def history(
        self, conversation_id: str, limit: int = 100, before_seq: int | None = None
    ) -> list[Message]:
        """Newest-first page of raw message rows, returned in chronological order."""
        async with self._sessionmaker() as session:
            query = select(Message).where(Message.conversation_id == conversation_id)
            if before_seq is not None:
                query = query.where(Message.seq < before_seq)
            rows = await session.scalars(query.order_by(Message.seq.desc()).limit(limit))
            return list(reversed(list(rows)))

    async def set_summary(self, conversation_id: str, summary: str, upto_seq: int) -> None:
        async with self._sessionmaker() as session, session.begin():
            await session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(summary=summary, summarized_upto_seq=upto_seq)
            )
