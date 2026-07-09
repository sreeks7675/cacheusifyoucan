from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from ..models.models import KnowledgeCategory, SourceType


class KnowledgeDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    category: KnowledgeCategory
    source_type: SourceType
    title: str
    original_filename: Optional[str] = None
    chunk_count: int
    created_at: datetime


class KnowledgeUploadResponse(BaseModel):
    document: KnowledgeDocumentResponse
    message: str = "Document ingested into the knowledge base."


class StructuredEntryRequest(BaseModel):
    category: KnowledgeCategory
    title: str
    entry: Dict[str, Any]


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    category: Optional[KnowledgeCategory] = None


class KnowledgeSearchResult(BaseModel):
    chunk_id: str
    text: str
    category: Optional[str] = None
    title: Optional[str] = None
    doc_id: Optional[str] = None
    distance: Optional[float] = None
    structured_data: Optional[Dict[str, Any]] = None


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: List[KnowledgeSearchResult]


class KnowledgeListResponse(BaseModel):
    total: int
    items: List[KnowledgeDocumentResponse]
