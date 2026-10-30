import math

import pytest

from agent_backend.rag.embeddings import HashingEmbedder
from agent_backend.rag.service import KnowledgeBase
from agent_backend.rag.store import InMemoryVectorStore, cosine
from agent_backend.tools.knowledge import KnowledgeSearchArgs, KnowledgeSearchTool

DOCS = {
    "redis.md": "# Redis\nRedis stores hot conversation state with a TTL. "
    "Distributed locks use SET NX PX with a random token.",
    "postgres.md": "# Postgres\nPostgres is the durable source of truth for message history. "
    "pgvector adds vector similarity search with HNSW indexes.",
    "cooking.md": "# Pasta\nBoil salted water, add spaghetti and cook for nine minutes.",
}


@pytest.fixture
async def kb() -> KnowledgeBase:
    knowledge_base = KnowledgeBase(HashingEmbedder(dim=256), InMemoryVectorStore())
    for source, text in DOCS.items():
        await knowledge_base.ingest(source, text)
    return knowledge_base


async def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dim=64)
    [a, b] = await embedder.embed(["distributed locks", "distributed locks"])
    assert a == b
    assert len(a) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0)
    [empty] = await embedder.embed([""])
    assert all(v == 0 for v in empty)


def test_cosine() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


@pytest.mark.parametrize(
    ("query", "expected_source"),
    [
        ("how do distributed locks work in redis", "redis.md"),
        ("vector similarity search hnsw", "postgres.md"),
        ("how long to cook spaghetti", "cooking.md"),
    ],
)
async def test_search_ranks_the_relevant_document_first(
    kb: KnowledgeBase, query: str, expected_source: str
) -> None:
    hits = await kb.search(query, k=3)
    assert hits[0].source == expected_source
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


async def test_ingest_reports_chunks(kb: KnowledgeBase) -> None:
    result = await kb.ingest("big.md", "# Big\n\n" + "\n\n".join(["sentence " * 40] * 10))
    assert result.chunks > 1
    assert await kb.store.count() == 3 + result.chunks


async def test_knowledge_tool_formats_hits_with_sources(kb: KnowledgeBase) -> None:
    tool = KnowledgeSearchTool(kb)
    output = await tool.run(KnowledgeSearchArgs(query="redis locks", top_k=2))
    assert output.startswith("[1] (source: redis.md")
    empty = KnowledgeSearchTool(KnowledgeBase(HashingEmbedder(), InMemoryVectorStore()))
    assert await empty.run(KnowledgeSearchArgs(query="anything")) == "No relevant passages found."
