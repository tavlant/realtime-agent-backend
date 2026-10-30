"""Vector stores: pgvector in Postgres, plus an in-memory store for dev and tests."""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent_backend.db.models import DocumentChunk


@dataclass(frozen=True)
class StoredChunk:
    document_id: str
    source: str
    chunk_index: int
    content: str
    embedding: list[float]
    heading: str | None = None


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    source: str
    chunk_index: int
    content: str
    score: float  # cosine similarity in [-1, 1]
    heading: str | None = None


class VectorStore(ABC):
    @abstractmethod
    async def add(self, chunks: Sequence[StoredChunk]) -> None: ...

    @abstractmethod
    async def search(self, embedding: Sequence[float], k: int = 4) -> list[SearchHit]: ...

    @abstractmethod
    async def count(self) -> int: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine search. Fine for thousands of chunks; not for millions."""

    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []
        self._lock = asyncio.Lock()

    async def add(self, chunks: Sequence[StoredChunk]) -> None:
        async with self._lock:
            self._chunks.extend(chunks)

    async def search(self, embedding: Sequence[float], k: int = 4) -> list[SearchHit]:
        scored = sorted(
            ((cosine(embedding, c.embedding), c) for c in self._chunks),
            key=lambda pair: pair[0],
            reverse=True,
        )[:k]
        return [
            SearchHit(c.document_id, c.source, c.chunk_index, c.content, round(s, 4), c.heading)
            for s, c in scored
        ]

    async def count(self) -> int:
        return len(self._chunks)


class PgVectorStore(VectorStore):
    """Approximate nearest-neighbour search with pgvector's HNSW index (cosine distance)."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def add(self, chunks: Sequence[StoredChunk]) -> None:
        async with self._sessionmaker() as session, session.begin():
            session.add_all(
                DocumentChunk(
                    document_id=c.document_id,
                    source=c.source,
                    chunk_index=c.chunk_index,
                    heading=c.heading,
                    content=c.content,
                    embedding=list(c.embedding),
                )
                for c in chunks
            )

    async def search(self, embedding: Sequence[float], k: int = 4) -> list[SearchHit]:
        distance = DocumentChunk.embedding.cosine_distance(list(embedding)).label("distance")
        async with self._sessionmaker() as session:
            rows = await session.execute(
                select(DocumentChunk, distance).order_by(distance).limit(k)
            )
            return [
                SearchHit(
                    chunk.document_id,
                    chunk.source,
                    chunk.chunk_index,
                    chunk.content,
                    round(1.0 - float(dist), 4),
                    chunk.heading,
                )
                for chunk, dist in rows
            ]

    async def count(self) -> int:
        async with self._sessionmaker() as session:
            return int(await session.scalar(select(func.count()).select_from(DocumentChunk)) or 0)
