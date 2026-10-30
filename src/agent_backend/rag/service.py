"""Knowledge base: ingest documents (chunk -> embed -> store) and semantic search."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from agent_backend.rag.chunking import chunk_text
from agent_backend.rag.embeddings import Embedder
from agent_backend.rag.store import SearchHit, StoredChunk, VectorStore
from agent_backend.telemetry import span

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    source: str
    chunks: int


class KnowledgeBase:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chunk_max_chars: int = 800,
        chunk_overlap_chars: int = 120,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    async def ingest(self, source: str, text: str) -> IngestResult:
        document_id = str(uuid.uuid4())
        with span("rag.ingest", source=source):
            chunks = chunk_text(text, self.chunk_max_chars, self.chunk_overlap_chars)
            stored: list[StoredChunk] = []
            for start in range(0, len(chunks), EMBED_BATCH_SIZE):
                batch = chunks[start : start + EMBED_BATCH_SIZE]
                vectors = await self.embedder.embed([c.text for c in batch])
                stored.extend(
                    StoredChunk(document_id, source, c.index, c.text, v, c.heading)
                    for c, v in zip(batch, vectors, strict=True)
                )
            await self.store.add(stored)
        logger.info("document ingested", extra={"source": source, "chunks": len(stored)})
        return IngestResult(document_id=document_id, source=source, chunks=len(stored))

    async def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[SearchHit]:
        with span("rag.search", k=k):
            [vector] = await self.embedder.embed([query])
            hits = await self.store.search(vector, k=k)
        return [h for h in hits if h.score > min_score]
