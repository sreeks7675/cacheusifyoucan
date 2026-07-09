"""
High-level ingestion and retrieval functions for the Shared Knowledge &
Memory module. Routes call these; these call chunker/knowledge_loader/
vector_store and keep the SQLite `knowledge_documents` registry in sync
with what's actually stored in the vector DB.
"""

import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.models import KnowledgeDocument, KnowledgeCategory, SourceType
from app.rag import vector_store
from app.rag.chunker import chunk_text
from app.rag.knowledge_loader import extract_text_from_upload, flatten_structured_entry
from app.utils.logger import logger


def ingest_document(
    db: Session,
    category: KnowledgeCategory,
    title: str,
    filename: str,
    file_bytes: bytes,
) -> KnowledgeDocument:
    """
    Ingests a free-form document (PDF/txt) into the knowledge base:
    extract text -> chunk -> embed & store -> record in SQLite.
    Used for the Deepfake Knowledge Base and External Retrieval boxes.
    """
    text = extract_text_from_upload(filename, file_bytes)
    if not text:
        raise ValueError("No extractable text found in the uploaded file.")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Document produced no chunks after text extraction.")

    doc = KnowledgeDocument(
        category=category,
        source_type=SourceType.FILE,
        title=title,
        original_filename=filename,
        chunk_count=len(chunks),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    ids = [f"{doc.id}::chunk::{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc.id,
            "category": category.value,
            "title": title,
            "chunk_index": i,
            "source_type": SourceType.FILE.value,
        }
        for i in range(len(chunks))
    ]
    vector_store.add_chunks(ids=ids, documents=chunks, metadatas=metadatas)

    logger.info(f"Ingested document '{title}' ({category.value}) -> {len(chunks)} chunks")
    return doc


def ingest_structured_entry(
    db: Session,
    category: KnowledgeCategory,
    title: str,
    entry: Dict[str, Any],
) -> KnowledgeDocument:
    """
    Ingests a single structured JSON entry (a manipulation pattern,
    forensic rule, model registry entry, or a past case/feedback
    record) as one embedded chunk, with the original JSON preserved in
    metadata for exact retrieval later.
    """
    text = flatten_structured_entry(entry)

    doc = KnowledgeDocument(
        category=category,
        source_type=SourceType.STRUCTURED,
        title=title,
        chunk_count=1,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    chunk_id = f"{doc.id}::chunk::0"
    metadata = {
        "doc_id": doc.id,
        "category": category.value,
        "title": title,
        "chunk_index": 0,
        "source_type": SourceType.STRUCTURED.value,
        "structured_data": json.dumps(entry),
    }
    vector_store.add_chunks(ids=[chunk_id], documents=[text], metadatas=[metadata])

    logger.info(f"Ingested structured entry '{title}' ({category.value})")
    return doc


def search_knowledge(
    query_text: str,
    top_k: int = 5,
    category: Optional[KnowledgeCategory] = None,
) -> List[Dict[str, Any]]:
    """
    Runs a semantic search over the knowledge base and returns a clean
    list of results (text, metadata, and similarity distance), ready to
    hand to the Retrieval / External Retrieval agent or an LLM's
    context window.
    """
    raw = vector_store.query(
        query_text=query_text,
        top_k=top_k,
        category=category.value if category else None,
    )

    results = []
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    ids = raw.get("ids", [[]])[0]

    for i in range(len(documents)):
        meta = metadatas[i] or {}
        structured = meta.get("structured_data")
        results.append(
            {
                "chunk_id": ids[i],
                "text": documents[i],
                "category": meta.get("category"),
                "title": meta.get("title"),
                "doc_id": meta.get("doc_id"),
                "distance": distances[i],
                "structured_data": json.loads(structured) if structured else None,
            }
        )

    return results


def delete_knowledge_document(db: Session, doc_id: str) -> bool:
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
    if not doc:
        return False

    vector_store.delete_document(doc_id)
    db.delete(doc)
    db.commit()
    logger.info(f"Deleted knowledge document '{doc_id}'")
    return True


def list_knowledge_documents(
    db: Session,
    category: Optional[KnowledgeCategory] = None,
    skip: int = 0,
    limit: int = 50,
):
    query = db.query(KnowledgeDocument)
    if category:
        query = query.filter(KnowledgeDocument.category == category)
    query = query.order_by(KnowledgeDocument.created_at.desc())

    total = query.count()
    items = query.offset(skip).limit(limit).all()
    return total, items
