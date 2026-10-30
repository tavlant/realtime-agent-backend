"""Knowledge-base search tool backed by the RAG pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_backend.rag.service import KnowledgeBase
from agent_backend.tools.base import Tool


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="What to look up.")
    top_k: int = Field(default=3, ge=1, le=8, description="Number of passages to return.")


class KnowledgeSearchTool(Tool[KnowledgeSearchArgs]):
    name = "search_knowledge_base"
    description = (
        "Semantic search over documents ingested into the knowledge base. Returns the most "
        "relevant passages with their source; cite the source when you use a passage."
    )
    args_model = KnowledgeSearchArgs

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.kb = knowledge_base

    async def run(self, args: KnowledgeSearchArgs) -> str:
        hits = await self.kb.search(args.query, k=args.top_k)
        if not hits:
            return "No relevant passages found."
        return "\n\n".join(
            f"[{i}] (source: {h.source}, score: {h.score:.2f})\n{h.content}"
            for i, h in enumerate(hits, start=1)
        )
