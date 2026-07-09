"""
Pydantic schemas for request validation and response serialization.

Kept separate from the SQLAlchemy models (app/models) on purpose:
ORM models describe DB structure, Pydantic schemas describe the API
contract. Mixing them tends to cause pain the moment either changes.
"""

from datetime import datetime
from typing import Optional, Any, Dict, List

from pydantic import BaseModel, Field, ConfigDict

from app.models.models import InvestigationStatus, Verdict


# ---------------------------------------------------------------------------
# /upload
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    investigation_id: str
    original_filename: str
    status: InvestigationStatus
    message: str = "Image uploaded successfully. Call /investigate to start analysis."


# ---------------------------------------------------------------------------
# /investigate
# ---------------------------------------------------------------------------
class InvestigateRequest(BaseModel):
    investigation_id: str = Field(..., description="ID returned by /upload")


class AgentOutputSchema(BaseModel):
    """
    Generic schema describing the JSON every agent is expected to return.
    Individual agents can put whatever they need inside `data`; this
    wrapper just guarantees a consistent, storable shape.
    """

    agent_name: str
    data: Dict[str, Any]


class InvestigateResponse(BaseModel):
    investigation_id: str
    status: InvestigationStatus
    report_id: Optional[str] = None
    verdict: Optional[Verdict] = None
    confidence_score: Optional[float] = None
    risk_level: Optional[str] = None
    threat_score: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    calibration_ece: Optional[float] = None
    threshold_used: Optional[float] = None
    decision_justification: Optional[str] = None
    uncertainty: Optional[Dict[str, float]] = None
    human_review_required: Optional[bool] = None
    review_reason: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# /report/{id}
# ---------------------------------------------------------------------------
class ReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    investigation_id: str
    verdict: Verdict
    confidence_score: float
    risk_level: Optional[str] = None
    threat_score: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    calibration_ece: Optional[float] = None
    threshold_used: Optional[float] = None
    decision_justification: Optional[str] = None
    uncertainty: Optional[Dict[str, float]] = None
    human_review_required: Optional[bool] = None
    review_reason: Optional[str] = None
    evidence_summary: Optional[str] = None
    explainable_reasoning: Optional[str] = None
    recommendations: Optional[str] = None
    highlighted_regions: Optional[Any] = None
    agent_outputs: Optional[List[Dict[str, Any]]] = None
    created_at: datetime


# ---------------------------------------------------------------------------
# /history
# ---------------------------------------------------------------------------
class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    investigation_id: str = Field(alias="id")
    original_filename: str
    status: InvestigationStatus
    verdict: Optional[Verdict] = None
    confidence_score: Optional[float] = None
    created_at: datetime


class HistoryResponse(BaseModel):
    total: int
    items: List[HistoryItem]


# ---------------------------------------------------------------------------
# Generic error response
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    detail: str
