"""Conversation memory: hot state in Redis, durable history in Postgres.

Read path:  Redis (summary + unsummarised messages)  --miss-->  rebuild from Postgres.
Write path: Postgres first (source of truth, one transaction per turn), then Redis.
            If the Redis write fails the hot keys are dropped and rebuilt on next read,
            so the cache can be stale-free without ever being authoritative.

When the unsummarised context exceeds the token budget, the oldest turns are folded into
a rolling summary. Cut points are only placed before a user message, so an assistant
tool call is never separated from its tool results.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from redis.asyncio import Redis
from redis.exceptions import RedisError

from agent_backend.core.summarizer import Summarizer
from agent_backend.core.tokens import estimate_messages_tokens, estimate_tokens
from agent_backend.db.repository import ConversationRepository, StoredMessage
from agent_backend.llm.base import ChatMessage
from agent_backend.telemetry import span

logger = logging.getLogger(__name__)


@dataclass
class ContextState:
    summary: str | None = None
    summarized_upto_seq: int = 0
    messages: list[StoredMessage] = field(default_factory=list)
    from_cache: bool = False

    @property
    def chat_messages(self) -> list[ChatMessage]:
        return [m.message for m in self.messages]

    def token_estimate(self) -> int:
        return estimate_tokens(self.summary or "") + estimate_messages_tokens(self.chat_messages)


class ConversationMemory:
    def __init__(
        self,
        redis: Redis,
        repo: ConversationRepository,
        summarizer: Summarizer,
        *,
        ttl_s: int = 3600,
        token_budget: int = 3000,
        summary_target_ratio: float = 0.5,
        min_recent_messages: int = 4,
    ) -> None:
        self.redis = redis
        self.repo = repo
        self.summarizer = summarizer
        self.ttl_s = ttl_s
        self.token_budget = token_budget
        self.summary_target_ratio = summary_target_ratio
        self.min_recent_messages = min_recent_messages

    @staticmethod
    def _keys(conversation_id: str) -> tuple[str, str]:
        return f"conv:{conversation_id}:meta", f"conv:{conversation_id}:msgs"

    @staticmethod
    def _encode(item: StoredMessage) -> str:
        return json.dumps({"seq": item.seq, "m": item.message.to_dict()}, separators=(",", ":"))

    @staticmethod
    def _decode(raw: str | bytes) -> StoredMessage:
        data = json.loads(raw)
        return StoredMessage(seq=data["seq"], message=ChatMessage.from_dict(data["m"]))

    # --- read ------------------------------------------------------------------------

    async def load(self, conversation_id: str) -> ContextState:
        meta_key, msgs_key = self._keys(conversation_id)
        with span("memory.load", conversation_id=conversation_id):
            try:
                async with self.redis.pipeline(transaction=False) as pipe:
                    pipe.hgetall(meta_key)
                    pipe.lrange(msgs_key, 0, -1)
                    pipe.expire(meta_key, self.ttl_s)
                    pipe.expire(msgs_key, self.ttl_s)
                    meta, raw_messages, *_ = await pipe.execute()
                if meta:
                    summary = _str(meta.get(b"summary", meta.get("summary"))) or None
                    upto = int(_str(meta.get(b"upto", meta.get("upto"))) or 0)
                    return ContextState(
                        summary, upto, [self._decode(r) for r in raw_messages], from_cache=True
                    )
            except RedisError:
                logger.warning("redis read failed; falling back to database", exc_info=True)
                return await self._load_from_db(conversation_id)

            state = await self._load_from_db(conversation_id)
            await self._write_cache(conversation_id, state)
            return state

    async def _load_from_db(self, conversation_id: str) -> ContextState:
        conversation = await self.repo.require(conversation_id)
        messages = await self.repo.messages_after(conversation_id, conversation.summarized_upto_seq)
        return ContextState(conversation.summary, conversation.summarized_upto_seq, messages)

    async def _write_cache(self, conversation_id: str, state: ContextState) -> None:
        meta_key, msgs_key = self._keys(conversation_id)
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.delete(meta_key, msgs_key)
                pipe.hset(
                    meta_key,
                    mapping={"summary": state.summary or "", "upto": state.summarized_upto_seq},
                )
                if state.messages:
                    pipe.rpush(msgs_key, *(self._encode(m) for m in state.messages))
                pipe.expire(meta_key, self.ttl_s)
                pipe.expire(msgs_key, self.ttl_s)
                await pipe.execute()
        except RedisError:
            logger.warning("redis cache write failed", exc_info=True)

    async def invalidate(self, conversation_id: str) -> None:
        try:
            await self.redis.delete(*self._keys(conversation_id))
        except RedisError:
            logger.warning("redis invalidate failed", exc_info=True)

    # --- write -----------------------------------------------------------------------

    async def commit_turn(
        self, conversation_id: str, turn_id: str, messages: Sequence[ChatMessage]
    ) -> list[int]:
        """Durably store a complete turn, then append it to the hot state."""
        with span("memory.commit_turn", conversation_id=conversation_id):
            seqs = await self.repo.append_turn(conversation_id, turn_id, messages)
            meta_key, msgs_key = self._keys(conversation_id)
            try:
                if await self.redis.exists(meta_key):
                    encoded = [
                        self._encode(StoredMessage(seq, m))
                        for seq, m in zip(seqs, messages, strict=True)
                    ]
                    async with self.redis.pipeline(transaction=True) as pipe:
                        pipe.rpush(msgs_key, *encoded)
                        pipe.expire(meta_key, self.ttl_s)
                        pipe.expire(msgs_key, self.ttl_s)
                        await pipe.execute()
            except RedisError:
                logger.warning("redis append failed; invalidating hot state", exc_info=True)
                await self.invalidate(conversation_id)
            return seqs

    # --- summarisation ---------------------------------------------------------------

    def choose_cut(self, messages: Sequence[StoredMessage]) -> int:
        """Index of the first message to keep verbatim (0 = nothing to summarise).

        Candidates are positions of user messages that leave at least
        `min_recent_messages` behind. Prefer the earliest cut that brings the kept tail
        under the target size; otherwise take the most aggressive valid cut.
        """
        target = int(self.token_budget * self.summary_target_ratio)
        last_allowed = len(messages) - self.min_recent_messages
        candidates = [i for i in range(1, last_allowed + 1) if messages[i].message.role == "user"]
        if not candidates:
            return 0
        for i in candidates:
            if estimate_messages_tokens(m.message for m in messages[i:]) <= target:
                return i
        return candidates[-1]

    async def maybe_summarize(
        self, conversation_id: str, state: ContextState | None = None
    ) -> bool:
        """Fold old turns into the summary if over budget.

        `state` lets the caller pass the context it already holds (saving a round trip).
        """
        state = state if state is not None else await self.load(conversation_id)
        if state.token_estimate() <= self.token_budget:
            return False
        cut = self.choose_cut(state.messages)
        if cut == 0:
            return False
        to_fold = state.messages[:cut]
        with span("memory.summarize", conversation_id=conversation_id, messages=len(to_fold)):
            try:
                summary = await self.summarizer.summarize(
                    state.summary, [m.message for m in to_fold]
                )
            except Exception:
                # Summarisation is an optimisation: never fail a turn because of it.
                logger.warning("summarisation failed; keeping full context", exc_info=True)
                return False
            upto = to_fold[-1].seq
            await self.repo.set_summary(conversation_id, summary, upto)
            new_state = ContextState(summary, upto, state.messages[cut:])
            await self._write_cache(conversation_id, new_state)
        logger.info(
            "conversation summarised",
            extra={"folded_messages": len(to_fold), "summarized_upto_seq": upto},
        )
        return True


def _str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode()
    return str(value)
