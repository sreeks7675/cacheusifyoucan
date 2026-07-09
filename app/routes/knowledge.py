"""
Routes for the Shared Knowledge & Memory (RAG) module:

    POST   /knowledge/documents        - ingest a PDF/txt document
    POST   /knowledge/entries          - ingest a structured JSON entry
                                          (pattern / rule / registry / past case)
    POST   /knowledge/search           - semantic search across the knowledge base
    GET    /knowledge/documents        - list ingested documents/entries
    DELETE /knowledge/documents/{id}   - remove a document and its chunks
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.models import KnowledgeCategory
from app.rag.retriever import (
    ingest_document,
    ingest_structured_entry,
    search_knowledge,
    delete_knowledge_document,
    list_knowledge_documents,
)
from app.schemas.knowledge_schemas import (
    KnowledgeUploadResponse,
    StructuredEntryRequest,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeListResponse,
)
from app.utils.logger import logger

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base (RAG)"])

ALLOWED_DOC_EXTENSIONS = {".pdf", ".txt", ".md"}


@router.post("/documents", response_model=KnowledgeUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    category: KnowledgeCategory = Form(...),
    title: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Ingests a free-form document (PDF or .txt/.md) into the knowledge
    base: extracts text, chunks it, embeds each chunk, and stores it in
    the vector database. Intended for the Deepfake Knowledge Base and
    External Retrieval categories, but works for any category.
    """
    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_DOC_EXTENSIONS)}",
        )

    file_bytes = await file.read()

    try:
        doc = ingest_document(
            db=db,
            category=category,
            title=title,
            filename=file.filename,
            file_bytes=file_bytes,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest document into the knowledge base.",
        )

    return KnowledgeUploadResponse(document=doc)


@router.post("/entries", response_model=KnowledgeUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_structured_entry(payload: StructuredEntryRequest, db: Session = Depends(get_db)):
    """
    Ingests a single structured JSON entry - a manipulation pattern, a
    forensic rule, a model registry record, or a past case/feedback
    note. Example body:

        {
          "category": "manipulation_patterns",
          "title": "Reflection mismatch",
          "entry": {
            "pattern": "Reflection mismatch",
            "description": "Reflection does not match lighting.",
            "severity": "High"
          }
        }
    """
    try:
        doc = ingest_structured_entry(
            db=db,
            category=payload.category,
            title=payload.title,
            entry=payload.entry,
        )
    except Exception as e:
        logger.error(f"Structured entry ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest entry into the knowledge base.",
        )

    return KnowledgeUploadResponse(document=doc)


@router.post("/search", response_model=KnowledgeSearchResponse)
def search(payload: KnowledgeSearchRequest):
    """
    Semantic search over the knowledge base. Optionally scoped to one
    category (e.g. only search 'forensic_rules'). This is what the
    Retrieval Agent / Fusion Agent would call to pull supporting
    evidence for an investigation.
    """
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty."
        )

    try:
        results = search_knowledge(
            query_text=payload.query,
            top_k=payload.top_k,
            category=payload.category,
        )
    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Knowledge base search failed.",
        )

    return KnowledgeSearchResponse(query=payload.query, results=results)


@router.get("/documents", response_model=KnowledgeListResponse)
def list_documents(
    category: KnowledgeCategory = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Lists ingested documents/entries, optionally filtered by category."""
    total, items = list_knowledge_documents(db, category=category, skip=skip, limit=limit)
    return KnowledgeListResponse(total=total, items=items)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """Removes a knowledge document and all of its embedded chunks."""
    deleted = delete_knowledge_document(db, doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No knowledge document found with id '{doc_id}'.",
        )
