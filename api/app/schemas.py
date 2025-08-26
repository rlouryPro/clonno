from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Item(BaseModel):
    id: str = Field(..., description="Unique ID of the document")
    text: str = Field(..., description="Raw text to embed and index")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    items: List[Item]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    with_payload: bool = True


class SearchHit(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    hits: List[SearchHit]