"""REST endpoints: health, conversations, history, documents."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from agent_backend import __version__
from agent_backend.api.deps import get_container
from agent_backend.api.schemas import (
    ConversationCreate,
    ConversationList,
    ConversationOut,
    DocumentIngest,
    DocumentIngested,
    MessageList,
    MessageOut,
    SearchHitOut,
    SearchResults,
)
from agent_backend.container import Container

router = APIRouter()


@router.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    """Liveness: the process is up and serving requests."""
    return {"status": "ok", "version": __version__}


@router.get("/readyz", tags=["ops"])
async def readyz(container: Container = Depends(get_container)) -> JSONResponse:
    """Readiness: dependencies (Postgres, Redis) are reachable."""
    db_ok = await container.db.ping()
    try:
        redis_ok = bool(await container.redis.ping())
    except Exception:
        redis_ok = False
    ok = db_ok and redis_ok
    return JSONResponse(
        status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if ok else "degraded",
            "database": db_ok,
            "redis": redis_ok,
            "llm_provider": container.llm.name,
        },
    )


@router.post(
    "/conversations",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
    tags=["conversations"],
)
async def create_conversation(
    body: ConversationCreate | None = None, container: Container = Depends(get_container)
) -> ConversationOut:
    conversation = await container.repo.create(title=body.title if body else None)
    return ConversationOut.model_validate(conversation)


@router.get("/conversations", response_model=ConversationList, tags=["conversations"])
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    container: Container = Depends(get_container),
) -> ConversationList:
    items, total = await container.repo.list(limit=limit, offset=offset)
    return ConversationList(
        items=[ConversationOut.model_validate(c) for c in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/conversations/{conversation_id}", response_model=ConversationOut, tags=["conversations"]
)
async def get_conversation(
    conversation_id: str, container: Container = Depends(get_container)
) -> ConversationOut:
    conversation = await container.repo.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationOut.model_validate(conversation)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessageList,
    tags=["conversations"],
)
async def get_history(
    conversation_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    before_seq: int | None = Query(default=None, ge=1),
    container: Container = Depends(get_container),
) -> MessageList:
    """Full history from Postgres (the source of truth), paginated by sequence number."""
    if await container.repo.get(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    rows = await container.repo.history(conversation_id, limit=limit, before_seq=before_seq)
    return MessageList(
        conversation_id=conversation_id, items=[MessageOut.model_validate(r) for r in rows]
    )


@router.post(
    "/documents",
    response_model=DocumentIngested,
    status_code=status.HTTP_201_CREATED,
    tags=["knowledge base"],
)
async def ingest_document(
    body: DocumentIngest, container: Container = Depends(get_container)
) -> DocumentIngested:
    """Ingest a text or Markdown document: chunk -> embed -> store."""
    result = await container.knowledge_base.ingest(body.source, body.content)
    return DocumentIngested(
        document_id=result.document_id, source=result.source, chunks=result.chunks
    )


@router.get("/documents/search", response_model=SearchResults, tags=["knowledge base"])
async def search_documents(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(default=4, ge=1, le=20),
    container: Container = Depends(get_container),
) -> SearchResults:
    hits = await container.knowledge_base.search(q, k=k)
    return SearchResults(
        query=q,
        hits=[
            SearchHitOut(
                source=h.source,
                chunk_index=h.chunk_index,
                heading=h.heading,
                content=h.content,
                score=h.score,
            )
            for h in hits
        ],
    )
