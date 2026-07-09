"""
Shared data schemas for the Evidence Fusion & Debate Agent (Agent 5).

These are the contracts every upstream agent (Forensic Analysis, Semantic &
Context, Retrieval & Comparison) must speak, and the contracts this agent
hands downstream to the Decision & Confidence Agent (Agent 6).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Claim(str, Enum):
    FAKE = "fake"
    REAL = "real"
    UNCERTAIN = "uncertain"


class DisagreementType(str, Enum):
    GENUINE_CONFLICT = "genuine_conflict"
    WEAK_EVIDENCE = "weak_evidence"
    NOVEL_PATTERN = "novel_pattern"
    NONE = "none"


class RemediationAction(str, Enum):
    CONCLUDE = "conclude"
    REQUEST_REANALYSIS = "request_reanalysis"
    FLAG_HUMAN_REVIEW = "flag_human_review"


@dataclass
class EvidenceItem:
    """A single piece of supporting evidence (a bbox, a score, a note)."""
    kind: str                     # e.g. "bbox", "similarity_score", "note"
    description: str
    location: Optional[dict] = None   # e.g. {"x":.., "y":.., "w":.., "h":..}
    value: Optional[float] = None     # e.g. a raw similarity/anomaly score


@dataclass
class EvidenceClaim:
    """What one upstream agent believes, and why."""
    agent: str                    # "ForensicAnalysisAgent", etc.
    claim: Claim
    confidence: float             # 0.0 - 1.0, agent's own confidence
    evidence: list[EvidenceItem] = field(default_factory=list)
    reliability_prior: float = 0.5  # filled in by the reliability store, not the agent itself

    def __post_init__(self) -> None:
        if not self.agent or not isinstance(self.agent, str):
            raise ValueError("EvidenceClaim.agent must be a non-empty string")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"EvidenceClaim.confidence must be in [0.0, 1.0], got {self.confidence!r} "
                f"(agent: {self.agent})"
            )


@dataclass
class CriticVerdict:
    """Result of cross-examining one EvidenceClaim."""
    agent: str
    survived: bool                # did the claim hold up under scrutiny?
    adjusted_confidence: float     # confidence after critique (may be lowered)
    critique: str                 # human-readable reason for any adjustment


@dataclass
class FusionResult:
    verdict: Claim
    confidence: float              # 0.0 - 1.0 fused confidence in `verdict`
    per_agent_weights: dict[str, float]
    uncertainty_score: float       # 0.0 (agents fully agree) - 1.0 (max disagreement)


@dataclass
class DissentReport:
    disagreement_type: DisagreementType
    conflicting_claims: list[EvidenceClaim]
    unresolved_question: str
    uncertainty_score: float
    recommended_action: RemediationAction


@dataclass
class FinalOutput:
    """The packet handed to the Decision & Confidence Agent (Agent 6)."""
    verdict: Claim
    confidence: float
    uncertainty_score: float
    risk_level: str                        # "low" / "medium" / "high"
    per_agent_contributions: dict[str, float]
    evidence_summary: list[EvidenceItem]
    dissent: Optional[DissentReport] = None
    recommended_actions: list[str] = field(default_factory=list)
