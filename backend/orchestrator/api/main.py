from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="TruthLens Orchestrator API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvestigationCreate(BaseModel):
    title: Optional[str] = Field(default="New Investigation")
    description: Optional[str] = None
    topic: Optional[str] = None
    source_type: Optional[str] = None
    source_url: Optional[str] = None
    question: Optional[str] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class InvestigationResponse(BaseModel):
    id: str
    status: str
    title: str
    description: Optional[str] = None
    topic: Optional[str] = None
    source_type: Optional[str] = None
    source_url: Optional[str] = None
    question: Optional[str] = None
    message: str


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "truthlens-orchestrator",
        "timestamp": "live",
    }


@app.post("/investigations", response_model=InvestigationResponse)
def create_investigation(payload: InvestigationCreate) -> InvestigationResponse:
    investigation_id = str(uuid4())[:8]
    return InvestigationResponse(
        id=investigation_id,
        status="queued",
        title=payload.title or "New Investigation",
        description=payload.description,
        topic=payload.topic,
        source_type=payload.source_type,
        source_url=payload.source_url,
        question=payload.question,
        message="Investigation received and queued for processing.",
    )