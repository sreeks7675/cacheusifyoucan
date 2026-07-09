from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import ReportResponse
from app.services.investigation_service import (
    get_investigation,
    get_report_by_investigation_id,
    serialize_report,
)

router = APIRouter(tags=["Report"])


@router.get("/report/{investigation_id}", response_model=ReportResponse)
def get_report(investigation_id: str, db: Session = Depends(get_db)):
    """
    Returns the full forensic report (verdict, confidence, evidence
    summary, explainable reasoning, and raw agent outputs) for a
    completed investigation.
    """
    investigation = get_investigation(db, investigation_id)
    if not investigation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No investigation found with id '{investigation_id}'.",
        )

    report = get_report_by_investigation_id(db, investigation_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No report yet for investigation '{investigation_id}' "
                f"(status: {investigation.status.value}). "
                f"Call /investigate first."
            ),
        )

    return serialize_report(db, report)
