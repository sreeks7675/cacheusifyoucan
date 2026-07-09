"""
Knowledge ingestion + retrieval helpers for the Shared Knowledge & Memory layer.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..models.models import KnowledgeCategory, KnowledgeDocument, SourceType
from .chunker import chunk_text
from .knowledge_loader import extract_text_from_upload, flatten_structured_entry
from .vector_store import add_documents, delete_document, query_documents


def _new_doc_id() -> str:
    return str(uuid.uuid4())


def ingest_document(
    db: Session,
    category: KnowledgeCategory,
    title: str,
    filename: str,
    file_bytes: bytes,
) -> KnowledgeDocument:
    text = extract_text_from_upload(filename=filename, file_bytes=file_bytes)
    if not text:
        raise ValueError("Uploaded document has no extractable text.")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Uploaded document produced no chunks after parsing.")

    doc_id = _new_doc_id()
    chunk_ids = [f"{doc_id}:{idx}" for idx in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "category": category.value,
            "title": title,
            "chunk_index": str(idx),
            "source_type": SourceType.FILE.value,
        }
        for idx in range(len(chunks))
    ]

    add_documents(ids=chunk_ids, documents=chunks, metadatas=metadatas)

    record = KnowledgeDocument(
        id=doc_id,
        category=category,
        source_type=SourceType.FILE,
        title=title,
        original_filename=filename,
        chunk_count=len(chunks),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def ingest_structured_entry(
    db: Session,
    category: KnowledgeCategory,
    title: str,
    entry: Dict[str, Any],
) -> KnowledgeDocument:
    text = flatten_structured_entry(entry)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Structured entry produced no searchable content.")

    doc_id = _new_doc_id()
    chunk_ids = [f"{doc_id}:{idx}" for idx in range(len(chunks))]
    entry_json = json.dumps(entry)
    metadatas = [
        {
            "doc_id": doc_id,
            "category": category.value,
            "title": title,
            "chunk_index": str(idx),
            "source_type": SourceType.STRUCTURED.value,
            "structured_data": entry_json,
        }
        for idx in range(len(chunks))
    ]

    add_documents(ids=chunk_ids, documents=chunks, metadatas=metadatas)

    record = KnowledgeDocument(
        id=doc_id,
        category=category,
        source_type=SourceType.STRUCTURED,
        title=title,
        original_filename=None,
        chunk_count=len(chunks),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def search_knowledge(
    query_text: str,
    top_k: int = 5,
    category: Optional[KnowledgeCategory] = None,
) -> List[Dict[str, Any]]:
    raw = query_documents(
        query_text=query_text,
        top_k=top_k,
        category=category.value if isinstance(category, KnowledgeCategory) else category,
    )

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    results: List[Dict[str, Any]] = []
    for idx in range(min(len(ids), len(docs), len(metas), len(dists))):
        metadata = metas[idx] or {}
        structured_data = metadata.get("structured_data")
        parsed_structured = None
        if isinstance(structured_data, str):
            try:
                parsed_structured = json.loads(structured_data)
            except json.JSONDecodeError:
                parsed_structured = None

        results.append(
            {
                "chunk_id": ids[idx],
                "text": docs[idx],
                "category": metadata.get("category"),
                "title": metadata.get("title"),
                "doc_id": metadata.get("doc_id"),
                "distance": float(dists[idx]) if dists[idx] is not None else None,
                "structured_data": parsed_structured,
            }
        )

    return results


def list_knowledge_documents(
    db: Session,
    category: Optional[KnowledgeCategory] = None,
    skip: int = 0,
    limit: int = 50,
) -> Tuple[int, List[KnowledgeDocument]]:
    query = db.query(KnowledgeDocument)
    if category is not None:
        query = query.filter(KnowledgeDocument.category == category)

    total = query.count()
    items = (
        query.order_by(KnowledgeDocument.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return total, items


def delete_knowledge_document(db: Session, doc_id: str) -> bool:
    record = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
    if record is None:
        return False

    # Delete all chunks for this document from vector DB.
    for idx in range(max(record.chunk_count, 0)):
        delete_document(f"{doc_id}:{idx}")

    db.delete(record)
    db.commit()
    return True


__all__ = [
    "ingest_document",
    "ingest_structured_entry",
    "search_knowledge",
    "list_knowledge_documents",
    "delete_knowledge_document",
]