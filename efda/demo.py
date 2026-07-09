"""
End-to-end demo of the Evidence Fusion & Debate Agent.

Run: python3 -m efda.demo
"""

from .agent import EvidenceFusionDebateAgent
from .schemas import Claim, EvidenceClaim, EvidenceItem


def scenario_confident_fake():
    print("=" * 70)
    print("SCENARIO 1: confident fake (mirrors the reference report — 92% fake)")
    print("=" * 70)
    claims = [
        EvidenceClaim(
            agent="ForensicAnalysisAgent", claim=Claim.FAKE, confidence=0.88,
            evidence=[
                EvidenceItem(kind="bbox", description="Lighting mismatch, right side of face",
                             location={"x": 210, "y": 40, "w": 60, "h": 60}),
                EvidenceItem(kind="frequency_signature", description="Frequency-domain anomaly detected", value=0.81),
            ],
        ),
        EvidenceClaim(
            agent="SemanticContextAgent", claim=Claim.FAKE, confidence=0.72,
            evidence=[EvidenceItem(kind="note", description="Reflection inconsistent with background lighting")],
        ),
        EvidenceClaim(
            agent="RetrievalComparisonAgent", claim=Claim.FAKE, confidence=0.81,
            evidence=[EvidenceItem(kind="similarity_score", description="High similarity to Stable Diffusion v2.1 samples", value=0.81)],
        ),
    ]
    agent = EvidenceFusionDebateAgent()
    result = agent.run(claims, case_type="face_swap")
    _print_result(result)


def scenario_genuine_conflict():
    print()
    print("=" * 70)
    print("SCENARIO 2: genuine conflict -> should trigger self-reflection")
    print("=" * 70)
    claims = [
        EvidenceClaim(
            agent="ForensicAnalysisAgent", claim=Claim.FAKE, confidence=0.81,
            evidence=[
                EvidenceItem(kind="bbox", description="Blending artifact along jawline"),
                EvidenceItem(kind="frequency_signature", description="Compression irregularity", value=0.74),
            ],
        ),
        EvidenceClaim(
            agent="RetrievalComparisonAgent", claim=Claim.REAL, confidence=0.77,
            evidence=[EvidenceItem(kind="similarity_score", description="Exact match to a verified real-photo database entry", value=0.9)],
        ),
    ]
    agent = EvidenceFusionDebateAgent()
    result = agent.run(claims, case_type="face_swap")
    _print_result(result)

    if result.dissent:
        print("\n-- Simulating resolution: human reviewer confirms it was FAKE (edited-then-reposted) --")
        agent.record_feedback("RetrievalComparisonAgent", "face_swap", was_correct=False)
        agent.record_feedback("ForensicAnalysisAgent", "face_swap", was_correct=True)
        print("Updated reliability snapshot:", agent.reliability_store.snapshot())


def _print_result(result):
    print(f"Verdict:            {result.verdict.value}")
    print(f"Confidence:         {result.confidence:.0%}")
    print(f"Uncertainty score:  {result.uncertainty_score:.2f}")
    print(f"Risk level:         {result.risk_level}")
    print(f"Per-agent weights:  {result.per_agent_contributions}")
    print("Recommended actions:")
    for a in result.recommended_actions:
        print(f"  - {a}")
    if result.dissent:
        print(f"Disagreement type:  {result.dissent.disagreement_type.value}")
        print(f"Unresolved question: {result.dissent.unresolved_question}")


if __name__ == "__main__":
    scenario_confident_fake()
    scenario_genuine_conflict()
