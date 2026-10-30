"""Composition root: builds and owns every long-lived dependency."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from redis.asyncio import Redis

from agent_backend.config import Settings
from agent_backend.core.agent import AgentConfig, AgentRunner
from agent_backend.core.locks import ConversationLocks
from agent_backend.core.memory import ConversationMemory
from agent_backend.core.retry import RetryPolicy
from agent_backend.core.summarizer import Summarizer
from agent_backend.db.repository import ConversationRepository
from agent_backend.db.session import Database
from agent_backend.llm.base import LLMProvider
from agent_backend.llm.factory import build_llm
from agent_backend.rag.embeddings import build_embedder
from agent_backend.rag.service import KnowledgeBase
from agent_backend.rag.store import InMemoryVectorStore, PgVectorStore, VectorStore
from agent_backend.tools import ToolRegistry, build_default_registry

logger = logging.getLogger(__name__)

T = TypeVar("T")


def build_redis(url: str) -> Redis:
    if url.startswith("memory://"):
        try:
            from fakeredis import FakeAsyncRedis
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("REDIS_URL=memory:// requires the 'dev' extra (fakeredis)") from exc
        return FakeAsyncRedis(decode_responses=True)
    return Redis.from_url(url, decode_responses=True, health_check_interval=30)


@dataclass
class Container:
    settings: Settings
    db: Database
    redis: Redis
    llm: LLMProvider
    repo: ConversationRepository
    knowledge_base: KnowledgeBase
    tools: ToolRegistry
    memory: ConversationMemory
    locks: ConversationLocks
    agent: AgentRunner
    inflight: set[asyncio.Task[Any]] = field(default_factory=set)

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        redis: Redis | None = None,
        llm: LLMProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> Container:
        db = Database(settings.database_url, echo=settings.db_echo)
        redis = redis if redis is not None else build_redis(settings.redis_url)
        llm = llm if llm is not None else build_llm(settings)
        repo = ConversationRepository(db.sessionmaker)

        if vector_store is None:
            if settings.resolved_vector_store() == "pgvector":
                vector_store = PgVectorStore(db.sessionmaker)
            else:
                vector_store = InMemoryVectorStore()
        knowledge_base = KnowledgeBase(
            build_embedder(settings),
            vector_store,
            chunk_max_chars=settings.chunk_max_chars,
            chunk_overlap_chars=settings.chunk_overlap_chars,
        )
        tools = build_default_registry(knowledge_base, default_timeout_s=settings.tool_timeout_s)

        retry_policy = RetryPolicy(
            max_attempts=settings.llm_max_attempts,
            base_delay_s=settings.retry_base_delay_s,
            max_delay_s=settings.retry_max_delay_s,
        )
        memory = ConversationMemory(
            redis,
            repo,
            Summarizer(llm, retry_policy),
            ttl_s=settings.hot_state_ttl_s,
            token_budget=settings.context_token_budget,
            summary_target_ratio=settings.summary_target_ratio,
            min_recent_messages=settings.min_recent_messages,
        )
        locks = ConversationLocks(redis, settings.lock_ttl_s, settings.lock_wait_timeout_s)
        agent = AgentRunner(
            llm,
            tools,
            memory,
            locks,
            AgentConfig(
                system_prompt=settings.system_prompt,
                max_tool_rounds=settings.max_tool_rounds,
                first_token_timeout_s=settings.llm_first_token_timeout_s,
                idle_timeout_s=settings.llm_idle_timeout_s,
                retry_policy=retry_policy,
            ),
        )
        return cls(settings, db, redis, llm, repo, knowledge_base, tools, memory, locks, agent)

    async def startup(self) -> None:
        if self.settings.db_auto_create:
            await self.db.init_schema()

    def track(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Run a turn as a tracked task.

        The task is independent of the request handler, so a dropped connection (or a
        cancelled handler) cannot abort a turn half-way, and shutdown can drain it.
        """
        task = asyncio.create_task(coro)
        self.inflight.add(task)
        task.add_done_callback(self._turn_done)
        return task

    def _turn_done(self, task: asyncio.Task[Any]) -> None:
        self.inflight.discard(task)
        if not task.cancelled() and task.exception() is not None:
            # Retrieved here so a turn whose client went away still leaves a trace.
            logger.info("tracked turn ended with error", extra={"error": repr(task.exception())})

    async def shutdown(self) -> None:
        if self.inflight:
            logger.info("draining in-flight turns", extra={"count": len(self.inflight)})
            _, pending = await asyncio.wait(
                self.inflight, timeout=self.settings.shutdown_drain_timeout_s
            )
            for task in pending:
                task.cancel()
        await self.llm.aclose()
        await self.redis.aclose()
        await self.db.dispose()
