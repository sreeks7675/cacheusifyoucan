"""
Orchestrates the multi-agent investigation pipeline for a single
investigation, persisting each agent's output and the final report.
"""

import json

from sqlalchemy.orm import Session

from app.models.models import Investigation, AgentOutput, Report, InvestigationStatus, Verdict
from app.services.agent_interface import AGENT_PIPELINE
from app.utils.logger import logger


def run_investigation_pipeline(db: Session, investigation: Investigation) -> Report:
    """
    Runs every agent in AGENT_PIPELINE against the investigation's image,
    storing each agent's raw JSON output in the `agent_outputs` table,
    then builds and stores the final `Report` row.

    Any exception marks the investigation FAILED (with the error message
    saved) and re-raises, so the route layer can return a proper 500/422.
    """
    investigation.status = InvestigationStatus.PROCESSING
    db.commit()

    context: dict = {}

    try:
        for agent_key, agent_fn in AGENT_PIPELINE:
            logger.info(
                f"[{investigation.id}] running agent '{agent_key}'"
            )
            result = agent_fn(investigation.file_path, context)
            agent_data = result["data"]
            context[agent_key] = agent_data

            db.add(
                AgentOutput(
                    investigation_id=investigation.id,
                    agent_name=agent_key,
                    output_json=json.dumps(agent_data),
                )
            )

        decision = context["decision"]
        report_data = context["report"]

        verdict_str = decision.get("verdict", "uncertain")
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.UNCERTAIN

        # Prepare writable report kwargs. Existing DB columns are preserved
        # and any optional new decision fields will be added if the model
        # already defines matching columns.
        report_kwargs = {
            "investigation_id": investigation.id,
            "verdict": verdict,
            "confidence_score": float(decision.get("confidence_score", 0.0)),
            "evidence_summary": context.get("fusion", {}).get("evidence_summary"),
            "explainable_reasoning": report_data.get("explainable_reasoning"),
            "recommendations": report_data.get("recommendations"),
            "highlighted_regions_json": json.dumps(
                report_data.get("highlighted_regions", [])
            ),
            "full_report_json": json.dumps(context),
        }

        # Only set extra report fields if the Report model already defines them.
        for field in [
            "risk_level",
            "threat_score",
            "confidence_interval",
            "calibration_ece",
            "threshold_used",
            "decision_justification",
            "uncertainty",
            "human_review_required",
            "review_reason",
        ]:
            if hasattr(Report, field):
                report_kwargs[field] = decision.get(field)

        report = Report(**report_kwargs)
        db.add(report)

        investigation.status = InvestigationStatus.COMPLETED
        db.commit()
        db.refresh(report)

        logger.info(
            f"[{investigation.id}] investigation completed -> "
            f"verdict={verdict.value}, confidence={report.confidence_score}"
        )
        return report

    except Exception as e:
        db.rollback()
        investigation.status = InvestigationStatus.FAILED
        investigation.error_message = str(e)
        db.commit()
        logger.error(f"[{investigation.id}] pipeline failed: {e}")
        raise
