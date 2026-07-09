"""
CRUD / query helpers for Investigation, Report, and AgentOutput rows.
Kept separate from the route handlers so routes stay thin and this
logic is independently testable.
"""

import json
from typing import Optional, List

from sqlalchemy.orm import Session

from app.models.models import Investigation, Report, AgentOutput


def create_investigation(db: Session, upload_meta: dict) -> Investigation:
    investigation = Investigation(
        original_filename=upload_meta["original_filename"],
        stored_filename=upload_meta["stored_filename"],
        file_path=upload_meta["file_path"],
        content_type=upload_meta.get("content_type"),
    )
    db.add(investigation)
    db.commit()
    db.refresh(investigation)
    return investigation


def get_investigation(db: Session, investigation_id: str) -> Optional[Investigation]:
    return (
        db.query(Investigation)
        .filter(Investigation.id == investigation_id)
        .first()
    )


def get_report_by_investigation_id(db: Session, investigation_id: str) -> Optional[Report]:
    return (
        db.query(Report)
        .filter(Report.investigation_id == investigation_id)
        .first()
    )


def get_report_by_id(db: Session, report_id: str) -> Optional[Report]:
    return db.query(Report).filter(Report.id == report_id).first()


def get_agent_outputs(db: Session, investigation_id: str) -> List[AgentOutput]:
    return (
        db.query(AgentOutput)
        .filter(AgentOutput.investigation_id == investigation_id)
        .order_by(AgentOutput.created_at.asc())
        .all()
    )


def list_history(db: Session, skip: int = 0, limit: int = 50):
    query = db.query(Investigation).order_by(Investigation.created_at.desc())
    total = query.count()
    items = query.offset(skip).limit(limit).all()
    return total, items


def serialize_report(db: Session, report: Report) -> dict:
    """Builds the dict expected by the ReportResponse schema, including
    parsed agent outputs for full transparency/explainability."""
    agent_outputs = get_agent_outputs(db, report.investigation_id)
    full_report = json.loads(report.full_report_json or "{}")
    decision_data = full_report.get("decision", {})

    return {
        "id": report.id,
        "investigation_id": report.investigation_id,
        "verdict": report.verdict,
        "confidence_score": report.confidence_score,
        "risk_level": decision_data.get("risk_level"),
        "threat_score": decision_data.get("threat_score"),
        "confidence_interval": decision_data.get("confidence_interval"),
        "calibration_ece": decision_data.get("calibration_ece"),
        "threshold_used": decision_data.get("threshold_used"),
        "decision_justification": decision_data.get("decision_justification"),
        "uncertainty": decision_data.get("uncertainty"),
        "human_review_required": decision_data.get("human_review_required"),
        "review_reason": decision_data.get("review_reason"),
        "evidence_summary": report.evidence_summary,
        "explainable_reasoning": report.explainable_reasoning,
        "recommendations": report.recommendations,
        "highlighted_regions": json.loads(report.highlighted_regions_json or "[]"),
        "agent_outputs": [
            {
                "agent_name": ao.agent_name,
                "data": json.loads(ao.output_json),
                "created_at": ao.created_at.isoformat(),
            }
            for ao in agent_outputs
        ],
        "created_at": report.created_at,
    }
