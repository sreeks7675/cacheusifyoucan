"""
ChromaDB-backed persistent vector store for the EADIS knowledge base.

This module creates or reuses a persistent ChromaDB collection named
`knowledge_base` and exposes simple helpers for document ingestion,
semantic query, and cleanup.
"""

from typing import Dict, List, Optional

import chromadb
from chromadb.api.types import Include
from chromadb.config import Settings

from app.config import KNOWLEDGE_COLLECTION_NAME, VECTOR_DB_DIR
from app.rag.embedder import get_embedding_function

_client = chromadb.Client(
    Settings(
        persist_directory=VECTOR_DB_DIR,
        is_persistent=True,
    )
)


def get_collection() -> chromadb.api.models.Collection.Collection:
    """Return the persistent knowledge collection, creating it if needed."""
    return _client.get_or_create_collection(
        name=KNOWLEDGE_COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


def add_documents(
    ids: List[str],
    documents: List[str],
    metadatas: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Add one or more documents to the Chroma collection."""
    collection = get_collection()
    collection.add(ids=ids, documents=documents, metadatas=metadatas)


def add_chunks(
    ids: List[str],
    documents: List[str],
    metadatas: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Alias for add_documents to preserve prior vector_store usage."""
    add_documents(ids=ids, documents=documents, metadatas=metadatas)


def query_documents(
    query_text: str,
    top_k: int = 5,
    category: Optional[str] = None,
) -> Dict[str, List]:
    """Query the knowledge collection and return raw Chroma results."""
    collection = get_collection()
    where = {"category": category} if category else None
    return collection.query(
        query_texts=[query_text],
        n_results=top_k,
        where=where,
        include=[
            Include.DOCUMENTS,
            Include.METADATAS,
            Include.DISTANCES,
            Include.IDS,
        ],
    )


def query(
    query_text: str,
    top_k: int = 5,
    category: Optional[str] = None,
) -> Dict[str, List]:
    """Alias for query_documents to preserve prior vector_store usage."""
    return query_documents(query_text=query_text, top_k=top_k, category=category)


def delete_document(document_id: str) -> None:
    """Delete a single document / chunk from the collection."""
    collection = get_collection()
    collection.delete(ids=[document_id])


def delete_all_documents() -> None:
    """Delete all documents from the knowledge collection."""
    collection = get_collection()
    collection.delete()


__all__ = [
    "get_collection",
    "add_documents",
    "add_chunks",
    "query_documents",
    "query",
    "delete_document",
    "delete_all_documents",
]
