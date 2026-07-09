from __future__ import annotations

from typing import Any

from backend.agents.fusion_debate.efda.agent import EvidenceFusionDebateAgent
from backend.agents.fusion_debate.efda.schemas import Claim, EvidenceClaim, EvidenceItem
from backend.orchestrator.state import AgentAnalysis, PipelineState


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _infer_claim(agent_analysis: AgentAnalysis) -> Claim:
    raw_output = agent_analysis.raw_output or {}

    # Prefer explicit verdict-like fields when present.
    for key in ("verdict", "critic_verdict", "label", "prediction", "decision"):
        verdict_text = _normalize_text(raw_output.get(key))
        if "fake" in verdict_text or "manip" in verdict_text:
            return Claim.FAKE
        if "real" in verdict_text or "authentic" in verdict_text:
            return Claim.REAL
        if "uncertain" in verdict_text or "inconclusive" in verdict_text:
            return Claim.UNCERTAIN

    # Fall back to findings text if no explicit verdict field exists.
    findings_text = " ".join(agent_analysis.findings or []).lower()
    if "fake" in findings_text or "manip" in findings_text:
        return Claim.FAKE
    if "real" in findings_text or "authentic" in findings_text:
        return Claim.REAL

    return Claim.UNCERTAIN


def _build_evidence_items(agent_analysis: AgentAnalysis) -> list[EvidenceItem]:
    evidence_items: list[EvidenceItem] = []

    for finding in agent_analysis.findings or []:
        evidence_items.append(
            EvidenceItem(
                kind="note",
                description=str(finding),
            )
        )

    for key, value in (agent_analysis.raw_output or {}).items():
        if isinstance(value, (int, float)):
            evidence_items.append(
                EvidenceItem(
                    kind="score",
                    description=str(key),
                    value=float(value),
                )
            )
        elif isinstance(value, str) and value.strip():
            evidence_items.append(
                EvidenceItem(
                    kind="note",
                    description=f"{key}: {value}",
                )
            )

    return evidence_items


def _analysis_to_claim(agent_key: str, analysis: AgentAnalysis) -> EvidenceClaim:
    claim = _infer_claim(analysis)
    confidence = max(0.0, min(1.0, float(analysis.confidence_score or 0.0)))

    return EvidenceClaim(
        agent=agent_key,
        claim=claim,
        confidence=confidence,
        evidence=_build_evidence_items(analysis),
    )


def _to_fake_score(verdict: Claim, confidence: float) -> float:
    if verdict == Claim.FAKE:
        return confidence
    if verdict == Claim.REAL:
        return 1.0 - confidence
    return 0.5


def fusion_debate_node(state: PipelineState) -> dict:
    print("\n--- [ORCHESTRATOR] Running Evidence Fusion Debate Agent (EFDA) ---")

    claims: list[EvidenceClaim] = []
    for key, analysis in (state.agent_responses or {}).items():
        try:
            claims.append(_analysis_to_claim(key, analysis))
        except Exception as exc:
            print(f"[FusionDebate] Skipping malformed analysis '{key}': {exc}")

    # Provide a minimal safe fallback claim so EFDA still returns a valid packet.
    if not claims:
        claims = [
            EvidenceClaim(
                agent="fallback",
                claim=Claim.UNCERTAIN,
                confidence=0.5,
                evidence=[EvidenceItem(kind="note", description="No upstream agent output was available.")],
            )
        ]

    try:
        efda = EvidenceFusionDebateAgent()
        final_output = efda.run(claims=claims)

        updated_evidence = state.consolidated_evidence.copy()
        updated_evidence.update(
            {
                "fusion_verdict": final_output.verdict.value,
                "fusion_confidence": final_output.confidence,
                "fusion_uncertainty": final_output.uncertainty_score,
                "fusion_risk_level": final_output.risk_level,
                "fusion_recommended_actions": final_output.recommended_actions,
                "fused_fake_score": _to_fake_score(final_output.verdict, final_output.confidence),
                "efda": {
                    "verdict": final_output.verdict.value,
                    "confidence": final_output.confidence,
                    "uncertainty_score": final_output.uncertainty_score,
                    "risk_level": final_output.risk_level,
                    "per_agent_contributions": final_output.per_agent_contributions,
                    "recommended_actions": final_output.recommended_actions,
                    "dissent": None,
                },
            }
        )

        if final_output.dissent is not None:
            updated_evidence["efda"]["dissent"] = {
                "disagreement_type": final_output.dissent.disagreement_type.value,
                "unresolved_question": final_output.dissent.unresolved_question,
                "uncertainty_score": final_output.dissent.uncertainty_score,
                "recommended_action": final_output.dissent.recommended_action.value,
                "conflicting_agents": [c.agent for c in final_output.dissent.conflicting_claims],
            }

        logs = [
            f"[EFDA] Verdict={final_output.verdict.value} Confidence={final_output.confidence:.3f} "
            f"Uncertainty={final_output.uncertainty_score:.3f}",
        ]
        if final_output.dissent is not None:
            logs.append(
                "[EFDA] Dissent raised: "
                f"{final_output.dissent.disagreement_type.value} -> "
                f"{final_output.dissent.recommended_action.value}"
            )

        return {
            "consolidated_evidence": updated_evidence,
            "debate_logs": logs,
        }

    except Exception as exc:
        print(f"[FusionDebate] EFDA execution failed: {exc}")
        fallback_evidence = state.consolidated_evidence.copy()
        fallback_evidence["fusion_error"] = str(exc)
        fallback_evidence["fused_fake_score"] = fallback_evidence.get("fused_fake_score", 0.5)

        return {
            "consolidated_evidence": fallback_evidence,
            "debate_logs": [f"[EFDA] Execution failed: {exc}"],
        }