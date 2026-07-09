"""
EvidenceFusionDebateAgent — Agent 5 in the investigation pipeline.

Wires together every stage built in steps 1-6:
  collect -> critic cross-examination -> reliability-weighted fusion ->
  confidence & uncertainty -> (low: conclude | high: self-reflection) -> output

Framework-agnostic: call `.run(...)` directly, or wrap it as a node in
LangGraph/CrewAI/AutoGen — the signature is a plain function of
(claims, case_context) -> FinalOutput, so it drops into any orchestration
layer without changes.
"""

from .critic import BaseCritic, RuleBasedCritic
from .fusion import fuse
from .reliability_store import ReliabilityStore
from .schemas import (
    Claim,
    EvidenceClaim,
    FinalOutput,
    RemediationAction,
)
from .self_reflection import UNCERTAINTY_THRESHOLD, generate_dissent_report
from .uncertainty import compute_uncertainty


class EvidenceFusionDebateAgent:
    def __init__(
        self,
        reliability_store: ReliabilityStore | None = None,
        critic: BaseCritic | None = None,
    ):
        self.reliability_store = reliability_store or ReliabilityStore()
        self.critic = critic or RuleBasedCritic()

    def run(
        self,
        claims: list[EvidenceClaim],
        case_type: str = "default",
        reanalysis_attempts: int = 0,
    ) -> FinalOutput:
        # Stage 1: fill in reliability priors from the shared store
        for c in claims:
            c.reliability_prior = self.reliability_store.get(c.agent, case_type)

        # Stage 2: critic cross-examination (the debate)
        critic_verdicts = self.critic.cross_examine(claims)

        # Stage 3: reliability-weighted fusion
        fusion_result = fuse(claims, critic_verdicts)

        # Stage 4: confidence & uncertainty (already folded into fusion_result,
        # recomputed explicitly here for clarity / standalone use)
        uncertainty_score = compute_uncertainty(claims, critic_verdicts)

        risk_level = self._risk_level(fusion_result.verdict, fusion_result.confidence)
        evidence_summary = [e for c in claims for e in c.evidence]

        # Stage 5: branch on uncertainty
        if uncertainty_score < UNCERTAINTY_THRESHOLD:
            return FinalOutput(
                verdict=fusion_result.verdict,
                confidence=fusion_result.confidence,
                uncertainty_score=uncertainty_score,
                risk_level=risk_level,
                per_agent_contributions=fusion_result.per_agent_weights,
                evidence_summary=evidence_summary,
                dissent=None,
                recommended_actions=["Standard confidence — proceed to final decision."],
            )

        # Stage 5b: self-reflection / failure analysis
        dissent = generate_dissent_report(claims, critic_verdicts, uncertainty_score, reanalysis_attempts)

        recommended = []
        if dissent.recommended_action == RemediationAction.REQUEST_REANALYSIS:
            recommended.append(
                "Re-dispatch to Investigation Planner Agent: request additional analysis "
                "(alternate crop, adjacent frame, or higher-resolution re-scan)."
            )
        else:
            recommended.append("Flag for human review before finalizing verdict.")
        recommended.append(f"Unresolved question: {dissent.unresolved_question}")

        return FinalOutput(
            verdict=fusion_result.verdict,
            confidence=fusion_result.confidence,
            uncertainty_score=uncertainty_score,
            risk_level=risk_level,
            per_agent_contributions=fusion_result.per_agent_weights,
            evidence_summary=evidence_summary,
            dissent=dissent,
            recommended_actions=recommended,
        )

    def record_feedback(self, agent: str, case_type: str, was_correct: bool) -> float:
        """Stage 6 (offline): call this once ground truth / human review
        resolves a case, to update that agent's reliability prior."""
        return self.reliability_store.update(agent, case_type, was_correct)

    @staticmethod
    def _risk_level(verdict: Claim, confidence: float) -> str:
        if verdict == Claim.UNCERTAIN:
            return "medium"
        if confidence >= 0.75:
            return "high" if verdict == Claim.FAKE else "low"
        return "medium"
