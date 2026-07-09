from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import json

from app.database import get_db
from app.models.models import InvestigationStatus
from app.schemas.schemas import InvestigateRequest, InvestigateResponse
from app.services.investigation_service import get_investigation
from app.services.pipeline_service import run_investigation_pipeline
from app.utils.logger import logger

router = APIRouter(tags=["Investigate"])


@router.post("/investigate", response_model=InvestigateResponse)
def investigate(payload: InvestigateRequest, db: Session = Depends(get_db)):
    """
    Runs the full agent pipeline (Planner -> Forensic -> Semantic ->
    Retrieval -> Fusion -> Decision -> Report) for a previously uploaded
    image and persists the resulting report.
    """
    investigation = get_investigation(db, payload.investigation_id)
    if not investigation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No investigation found with id '{payload.investigation_id}'.",
        )

    if investigation.status == InvestigationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This investigation has already completed. Fetch it via /report/{id}.",
        )

    try:
        report = run_investigation_pipeline(db, investigation)
    except Exception as e:
        logger.error(f"Investigation {investigation.id} failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation pipeline failed: {e}",
        )

    decision_data = json.loads(report.full_report_json or "{}").get("decision", {})

    return InvestigateResponse(
        investigation_id=investigation.id,
        status=investigation.status,
        report_id=report.id,
        verdict=report.verdict,
        confidence_score=report.confidence_score,
        risk_level=decision_data.get("risk_level"),
        threat_score=decision_data.get("threat_score"),
        confidence_interval=decision_data.get("confidence_interval"),
        calibration_ece=decision_data.get("calibration_ece"),
        threshold_used=decision_data.get("threshold_used"),
        decision_justification=decision_data.get("decision_justification"),
        uncertainty=decision_data.get("uncertainty"),
        human_review_required=decision_data.get("human_review_required"),
        review_reason=decision_data.get("review_reason"),
        message="Investigation completed successfully.",
    )
