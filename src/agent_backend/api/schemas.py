"""Request/response models for the REST API and the WebSocket protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str | None
    message_count: int
    summary: str | None = None
    created_at: datetime
    updated_at: datetime


class ConversationList(BaseModel):
    items: list[ConversationOut]
    total: int
    limit: int
    offset: int


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seq: int
    turn_id: str
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    created_at: datetime


class MessageList(BaseModel):
    conversation_id: str
    items: list[MessageOut]


class DocumentIngest(BaseModel):
    source: str = Field(..., min_length=1, max_length=500, examples=["handbook.md"])
    content: str = Field(..., min_length=1, max_length=2_000_000)


class DocumentIngested(BaseModel):
    document_id: str
    source: str
    chunks: int


class SearchHitOut(BaseModel):
    source: str
    chunk_index: int
    heading: str | None
    content: str
    score: float


class SearchResults(BaseModel):
    query: str
    hits: list[SearchHitOut]


# --- WebSocket protocol ----------------------------------------------------------------


class ClientUserMessage(BaseModel):
    type: Literal["user_message"]
    content: str = Field(..., min_length=1, max_length=32_000)


class ClientPing(BaseModel):
    type: Literal["ping"]
