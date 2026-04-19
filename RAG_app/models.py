from typing import Any, Literal

from pydantic import BaseModel, Field


class RagDocument(BaseModel):
    id: str
    entity_type: Literal["doctor", "package"]
    title: str
    category: str | None = None
    source_url: str
    text: str
    search_text: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    id: str
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    title: str
    entity_type: str
    category: str | None = None
    source_url: str | None = None
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)


class RagAnswer(BaseModel):
    answer: str
    sources: list[dict[str, str | None]]
    confidence: str
    intent: str
    used_context_ids: list[str]
