"""
Self-reflection & failure analysis.

Triggered when uncertainty_score exceeds UNCERTAINTY_THRESHOLD. Classifies
*why* the agents disagree (not all disagreement is equal), writes a
structured dissent report, and decides whether to loop back for more
analysis or escalate straight to human review.
"""

from .schemas import (
    Claim,
    CriticVerdict,
    DisagreementType,
    DissentReport,
    EvidenceClaim,
    RemediationAction,
)

UNCERTAINTY_THRESHOLD = 0.45
MAX_REANALYSIS_ATTEMPTS = 1  # avoid infinite re-dispatch loops


def classify_disagreement(claims: list[EvidenceClaim], critic_verdicts: list[CriticVerdict]) -> DisagreementType:
    critic_by_agent = {v.agent: v for v in critic_verdicts}
    confidences = [critic_by_agent[c.agent].adjusted_confidence for c in claims]

    strong_claims = [c for c in claims if critic_by_agent[c.agent].adjusted_confidence >= 0.6]
    opposing_strong = {c.claim for c in strong_claims if c.claim != Claim.UNCERTAIN}

    if len(opposing_strong) >= 2:
        return DisagreementType.GENUINE_CONFLICT

    if all(conf < 0.4 for conf in confidences):
        return DisagreementType.WEAK_EVIDENCE

    # Retrieval-style agent found nothing, but another agent flagged anomalies
    # with real confidence -> treat as a potentially novel manipulation, not "real".
    retrieval_claims = [c for c in claims if "Retrieval" in c.agent]
    other_claims = [c for c in claims if "Retrieval" not in c.agent]
    if retrieval_claims and all(c.claim == Claim.UNCERTAIN for c in retrieval_claims) and \
       any(critic_by_agent[c.agent].adjusted_confidence >= 0.5 for c in other_claims):
        return DisagreementType.NOVEL_PATTERN

    return DisagreementType.GENUINE_CONFLICT  # default to the safer, more conservative bucket


def generate_dissent_report(
    claims: list[EvidenceClaim],
    critic_verdicts: list[CriticVerdict],
    uncertainty_score: float,
    reanalysis_attempts: int = 0,
) -> DissentReport:
    disagreement_type = classify_disagreement(claims, critic_verdicts)
    critic_by_agent = {v.agent: v for v in critic_verdicts}

    conflicting = [
        c for c in claims
        if critic_by_agent[c.agent].adjusted_confidence >= 0.4 and c.claim != Claim.UNCERTAIN
    ]

    question = _build_unresolved_question(disagreement_type, conflicting)
    action = _decide_action(disagreement_type, reanalysis_attempts)

    return DissentReport(
        disagreement_type=disagreement_type,
        conflicting_claims=conflicting,
        unresolved_question=question,
        uncertainty_score=uncertainty_score,
        recommended_action=action,
    )


def _build_unresolved_question(disagreement_type: DisagreementType, conflicting: list[EvidenceClaim]) -> str:
    if disagreement_type == DisagreementType.WEAK_EVIDENCE:
        return ("All agents report low confidence, likely due to degraded input quality "
                "(compression, low resolution, or heavy noise). No agent found strong signal either way.")
    if disagreement_type == DisagreementType.NOVEL_PATTERN:
        return ("Forensic/semantic analysis flagged anomalies with moderate-to-high confidence, "
                "but no matching generator fingerprint was found in the retrieval database. "
                "This may indicate a manipulation technique not yet represented in the knowledge base "
                "rather than evidence of authenticity.")
    if len(conflicting) >= 2:
        a, b = conflicting[0], conflicting[1]
        return (f"{a.agent} claims '{a.claim.value}' (confidence {a.confidence:.2f}) while "
                f"{b.agent} claims '{b.claim.value}' (confidence {b.confidence:.2f}) — "
                "both supported by evidence that survived critique. Evidence itself is in tension.")
    return "Agents disagree without a clear resolving factor after cross-examination."


def _decide_action(disagreement_type: DisagreementType, reanalysis_attempts: int) -> RemediationAction:
    if reanalysis_attempts >= MAX_REANALYSIS_ATTEMPTS:
        return RemediationAction.FLAG_HUMAN_REVIEW
    if disagreement_type in (DisagreementType.WEAK_EVIDENCE, DisagreementType.NOVEL_PATTERN):
        return RemediationAction.REQUEST_REANALYSIS
    return RemediationAction.FLAG_HUMAN_REVIEW
