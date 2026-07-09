from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import HistoryResponse, HistoryItem
from app.services.investigation_service import list_history, get_report_by_investigation_id

router = APIRouter(tags=["History"])


@router.get("/history", response_model=HistoryResponse)
def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """
    Returns a paginated list of all past investigations with their
    status and (if available) verdict/confidence summary.
    """
    total, investigations = list_history(db, skip=skip, limit=limit)

    items = []
    for inv in investigations:
        report = get_report_by_investigation_id(db, inv.id)
        items.append(
            HistoryItem(
                id=inv.id,
                original_filename=inv.original_filename,
                status=inv.status,
                verdict=report.verdict if report else None,
                confidence_score=report.confidence_score if report else None,
                created_at=inv.created_at,
            )
        )

    return HistoryResponse(total=total, items=items)
