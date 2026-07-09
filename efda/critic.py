"""
Critic / cross-examination stage.

Before claims get fused, each one is challenged: does it hold up against
the OTHER agents' evidence? A claim that survives keeps its confidence;
one that's contradicted by stronger, more specific evidence gets its
confidence discounted (never silently dropped — the discount is logged
in the critique).

Two implementations:
- RuleBasedCritic: fast, deterministic, no external calls. Good default
  and good for hackathon demos / unit tests.
- LLMCritic: delegates the cross-examination reasoning to an LLM via a
  pluggable `llm_call` function — swap in your Anthropic/OpenAI client.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from .schemas import Claim, CriticVerdict, EvidenceClaim

# A claim with specific, localized evidence (a bbox, a named artifact) is
# harder to dismiss than a vague high-level one. Used as a tiebreaker.
SPECIFIC_EVIDENCE_KINDS = {"bbox", "similarity_score", "frequency_signature"}


class BaseCritic(ABC):
    @abstractmethod
    def cross_examine(self, claims: list[EvidenceClaim]) -> list[CriticVerdict]:
        ...


class RuleBasedCritic(BaseCritic):
    """Heuristic critic: for each claim, checks whether any opposing claim
    has both higher confidence AND more specific evidence. If so, the
    original claim's confidence is discounted proportionally."""

    def cross_examine(self, claims: list[EvidenceClaim]) -> list[CriticVerdict]:
        verdicts = []
        for c in claims:
            opposing = [
                o for o in claims
                if o.agent != c.agent and o.claim != c.claim and o.claim != Claim.UNCERTAIN
            ]
            if not opposing:
                verdicts.append(CriticVerdict(
                    agent=c.agent, survived=True, adjusted_confidence=c.confidence,
                    critique="No opposing claims to reconcile against.",
                ))
                continue

            strongest_opposition = max(opposing, key=lambda o: self._specificity_score(o))
            c_specificity = self._specificity_score(c)
            o_specificity = self._specificity_score(strongest_opposition)

            if strongest_opposition.confidence > c.confidence and o_specificity > c_specificity:
                discount = 0.35  # claim contradicted by stronger, more specific evidence
                adjusted = round(c.confidence * (1 - discount), 4)
                verdicts.append(CriticVerdict(
                    agent=c.agent, survived=False, adjusted_confidence=adjusted,
                    critique=(
                        f"Contradicted by {strongest_opposition.agent} "
                        f"(confidence {strongest_opposition.confidence:.2f}, "
                        f"more specific evidence). Confidence discounted {discount:.0%}."
                    ),
                ))
            else:
                verdicts.append(CriticVerdict(
                    agent=c.agent, survived=True, adjusted_confidence=c.confidence,
                    critique="Held up against opposing claims; evidence at least as strong.",
                ))
        return verdicts

    @staticmethod
    def _specificity_score(claim: EvidenceClaim) -> float:
        if not claim.evidence:
            return 0.0
        specific = sum(1 for e in claim.evidence if e.kind in SPECIFIC_EVIDENCE_KINDS)
        return specific / len(claim.evidence)


class LLMCritic(BaseCritic):
    """Delegates cross-examination to an LLM. `llm_call` should accept a
    prompt string and return the model's text response — wire this to
    your Anthropic/OpenAI client. Falls back to RuleBasedCritic reasoning
    as a safety net if parsing the LLM response fails."""

    def __init__(self, llm_call: Callable[[str], str]):
        self.llm_call = llm_call
        self._fallback = RuleBasedCritic()

    def cross_examine(self, claims: list[EvidenceClaim]) -> list[CriticVerdict]:
        prompt = self._build_prompt(claims)
        try:
            response = self.llm_call(prompt)
            return self._parse(response, claims)
        except Exception:
            return self._fallback.cross_examine(claims)

    @staticmethod
    def _build_prompt(claims: list[EvidenceClaim]) -> str:
        lines = ["You are a skeptical forensic critic reviewing conflicting evidence claims about whether an image is a deepfake.",
                  "For each claim, decide if it survives scrutiny given the others, and adjust its confidence if contradicted by stronger, more specific evidence.",
                  "Respond as JSON: a list of {agent, survived, adjusted_confidence, critique}.", ""]
        for c in claims:
            lines.append(f"- {c.agent}: claims '{c.claim.value}' at confidence {c.confidence:.2f}, "
                          f"evidence: {[e.description for e in c.evidence]}")
        return "\n".join(lines)

    @staticmethod
    def _parse(response: str, claims: list[EvidenceClaim]) -> list[CriticVerdict]:
        import json
        parsed = json.loads(response)
        return [
            CriticVerdict(
                agent=item["agent"], survived=item["survived"],
                adjusted_confidence=float(item["adjusted_confidence"]),
                critique=item["critique"],
            )
            for item in parsed
        ]
