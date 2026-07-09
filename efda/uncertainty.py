"""
Confidence vs. uncertainty — kept as two separate numbers on purpose.
"92% fake, low uncertainty" and "92% fake, high uncertainty" should never
collapse into the same headline number.

uncertainty_score combines:
  - variance of signed confidences across agents (do they point the same way?)
  - entropy of the claim distribution (is one outcome dominant, or a toss-up?)
"""

import math

from .schemas import Claim, CriticVerdict, EvidenceClaim


def _signed(claim: Claim, confidence: float) -> float:
    if claim == Claim.FAKE:
        return confidence
    if claim == Claim.REAL:
        return -confidence
    return 0.0


def compute_uncertainty(claims: list[EvidenceClaim], critic_verdicts: list[CriticVerdict]) -> float:
    critic_by_agent = {v.agent: v for v in critic_verdicts}
    signed_scores = [
        _signed(c.claim, critic_by_agent[c.agent].adjusted_confidence)
        for c in claims
    ]

    if len(signed_scores) < 2:
        return 0.0

    mean = sum(signed_scores) / len(signed_scores)
    variance = sum((s - mean) ** 2 for s in signed_scores) / len(signed_scores)
    variance_component = min(1.0, variance / 0.5)  # normalize, cap at 1.0

    # Entropy over {fake, real, uncertain} weighted by confidence mass
    mass = {Claim.FAKE: 0.0, Claim.REAL: 0.0, Claim.UNCERTAIN: 0.0}
    for c in claims:
        mass[c.claim] += critic_by_agent[c.agent].adjusted_confidence
    total_mass = sum(mass.values()) or 1e-9
    probs = [v / total_mass for v in mass.values() if v > 0]
    entropy = -sum(p * math.log(p, 3) for p in probs) if probs else 0.0  # log base 3 -> normalized to [0,1]

    uncertainty = round(0.6 * variance_component + 0.4 * entropy, 4)
    return min(1.0, uncertainty)
