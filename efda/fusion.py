"""
Reliability-weighted fusion.

Combines critic-adjusted claims into a single verdict. Each claim is
converted to a signed score (fake = +confidence, real = -confidence,
uncertain = 0), weighted by the agent's reliability prior, and averaged.
The sign of the weighted sum gives the verdict; its magnitude gives the
fused confidence.

This is intentionally simple (weighted averaging) rather than a full
Dempster-Shafer combination rule — enough to satisfy "evidence-driven,
reliability-weighted decisions" without over-engineering a hackathon
timeline. Swap in Dempster-Shafer or a Bayesian network later if you
need more rigorous conflict handling.
"""

from .schemas import Claim, CriticVerdict, EvidenceClaim, FusionResult


def _signed_score(claim: Claim, confidence: float) -> float:
    if claim == Claim.FAKE:
        return confidence
    if claim == Claim.REAL:
        return -confidence
    return 0.0


def fuse(claims: list[EvidenceClaim], critic_verdicts: list[CriticVerdict]) -> FusionResult:
    verdict_by_agent = {c.agent: c for c in claims}
    critic_by_agent = {v.agent: v for v in critic_verdicts}

    weighted_sum = 0.0
    total_weight = 0.0
    per_agent_weights: dict[str, float] = {}

    for agent, claim in verdict_by_agent.items():
        critic = critic_by_agent[agent]
        # A claim that didn't survive critique still contributes, but
        # discounted — it's not deleted, just down-weighted (traceable).
        effective_confidence = critic.adjusted_confidence
        weight = claim.reliability_prior * effective_confidence
        signed = _signed_score(claim.claim, effective_confidence)

        weighted_sum += weight * (1 if signed >= 0 else -1) * abs(signed)
        total_weight += weight
        per_agent_weights[agent] = round(weight, 4)

    if total_weight == 0:
        fused_verdict, fused_confidence = Claim.UNCERTAIN, 0.0
    else:
        normalized = weighted_sum / total_weight  # in [-1, 1]
        fused_confidence = round(abs(normalized), 4)
        if normalized > 0.15:
            fused_verdict = Claim.FAKE
        elif normalized < -0.15:
            fused_verdict = Claim.REAL
        else:
            fused_verdict = Claim.UNCERTAIN

    uncertainty = _uncertainty_placeholder(claims, critic_verdicts)

    return FusionResult(
        verdict=fused_verdict,
        confidence=fused_confidence,
        per_agent_weights=per_agent_weights,
        uncertainty_score=uncertainty,
    )


def _uncertainty_placeholder(claims, critic_verdicts) -> float:
    # Real uncertainty scoring happens in uncertainty.py (step 5).
    # Placeholder keeps FusionResult self-contained if used standalone.
    from .uncertainty import compute_uncertainty
    return compute_uncertainty(claims, critic_verdicts)
