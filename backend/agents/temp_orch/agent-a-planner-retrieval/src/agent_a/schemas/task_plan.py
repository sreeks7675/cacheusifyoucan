"""Structured I/O contracts for Adapter A (orchestration & tool-calling).

Both the Planner Agent (1) and Retrieval Agent (4) must emit JSON that validates
against one of these schemas. Keeping the schemas here, separate from the agents,
means the SLM client, training-data prep, and unit tests can all import them
without depending on agent logic.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentName(str, Enum):
    """Full agent roster (system-wide). Not all of these are choosable by the
    Planner — see PlannerDispatchableAgent below."""

    FORENSIC = "forensic_analysis"
    SEMANTIC = "semantic_context"
    RETRIEVAL = "retrieval_comparison"
    FUSION_DEBATE = "evidence_fusion_debate"
    DECISION = "decision_confidence"
    REPORT = "report_generation"
    VOICE_ANTISPOOF = "voice_antispoofing"


class PlannerDispatchableAgent(str, Enum):
    """The ONLY agents the Planner is allowed to dispatch to. Per the current
    design, Evidence Fusion & Debate always runs as a fixed step *after*
    whichever of these three ran — it's an orchestrator-level rule, not a
    planner decision. Voice anti-spoofing is a separate audio-track branch,
    also not part of this planner's decision space."""

    FORENSIC = "forensic_analysis"
    SEMANTIC = "semantic_context"
    RETRIEVAL = "retrieval_comparison"


class IssueCategory(str, Enum):
    """Surface-level cues the Planner looks for directly on the image. Each
    category maps to exactly one downstream agent (see CATEGORY_TO_AGENT) —
    this is what makes the planner a 'jack of all trades, surface only' triage
    step rather than a duplicate of the deep specialist agents."""

    # -> forensic_analysis
    BLENDING_BOUNDARY = "blending_boundary"
    FREQUENCY_ANALYSIS = "frequency_analysis"
    FACE_LANDMARKS_EYES = "face_landmarks_eyes"
    REFLECTION_SPECULAR = "reflection_specular"
    NOISE_COMPRESSION = "noise_compression"
    METADATA_ANOMALY = "metadata_anomaly"

    # -> semantic_context
    OBJECT_SCENE_CONSISTENCY = "object_scene_consistency"
    VLM_SEMANTIC_CHECK = "vlm_semantic_check"
    TEXT_WATERMARK = "text_watermark"
    CONTEXT_REAL_WORLD = "context_real_world"
    LIGHTING_SHADOW_CONSISTENCY = "lighting_shadow_consistency"

    # -> retrieval_comparison
    PATTERN_COMPARISON = "pattern_comparison"
    GENERATOR_MODEL_MATCH = "generator_model_match"
    REVERSE_IMAGE_CANDIDATE = "reverse_image_candidate"


CATEGORY_TO_AGENT: Dict[IssueCategory, PlannerDispatchableAgent] = {
    IssueCategory.BLENDING_BOUNDARY: PlannerDispatchableAgent.FORENSIC,
    IssueCategory.FREQUENCY_ANALYSIS: PlannerDispatchableAgent.FORENSIC,
    IssueCategory.FACE_LANDMARKS_EYES: PlannerDispatchableAgent.FORENSIC,
    IssueCategory.REFLECTION_SPECULAR: PlannerDispatchableAgent.FORENSIC,
    IssueCategory.NOISE_COMPRESSION: PlannerDispatchableAgent.FORENSIC,
    IssueCategory.METADATA_ANOMALY: PlannerDispatchableAgent.FORENSIC,
    IssueCategory.OBJECT_SCENE_CONSISTENCY: PlannerDispatchableAgent.SEMANTIC,
    IssueCategory.VLM_SEMANTIC_CHECK: PlannerDispatchableAgent.SEMANTIC,
    IssueCategory.TEXT_WATERMARK: PlannerDispatchableAgent.SEMANTIC,
    IssueCategory.CONTEXT_REAL_WORLD: PlannerDispatchableAgent.SEMANTIC,
    IssueCategory.LIGHTING_SHADOW_CONSISTENCY: PlannerDispatchableAgent.SEMANTIC,
    IssueCategory.PATTERN_COMPARISON: PlannerDispatchableAgent.RETRIEVAL,
    IssueCategory.GENERATOR_MODEL_MATCH: PlannerDispatchableAgent.RETRIEVAL,
    IssueCategory.REVERSE_IMAGE_CANDIDATE: PlannerDispatchableAgent.RETRIEVAL,
}


class DetectedIssue(BaseModel):
    category: IssueCategory
    note: str = Field(..., min_length=1, description="One short phrase, e.g. 'iris shape looks warped'")
    region: Optional[str] = Field(None, description="Where the issue was seen, e.g. 'right_face'")
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("note")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("note must not be empty")
        return v


class TaskPlan(BaseModel):
    """Structured output required from the Planner Agent.

    `agents_to_call` is intentionally restricted to PlannerDispatchableAgent —
    Fusion & Debate is never a planner choice, it's a fixed post-step the
    orchestrator runs after whichever of these three actually ran.
    """

    detected_issues: List[DetectedIssue] = Field(default_factory=list)
    agents_to_call: List[PlannerDispatchableAgent] = Field(..., min_length=1)
    parallelizable: List[List[PlannerDispatchableAgent]] = Field(
        default_factory=list,
        description="Groups of agents that can run in parallel, in call order.",
    )
    reasoning: str = Field(..., min_length=1, description="Why these agents were chosen; feeds explainability.")

    @model_validator(mode="after")
    def agents_cover_detected_categories(self) -> "TaskPlan":
        """Internal consistency check: every flagged category's owning agent
        must actually be dispatched. Catches the failure mode where the model
        notices an issue but 'forgets' to call the agent that handles it —
        this doubles as a training-data quality check (see prepare_planner_dataset.py)."""
        required_agents = {CATEGORY_TO_AGENT[issue.category] for issue in self.detected_issues}
        missing = required_agents - set(self.agents_to_call)
        if missing:
            raise ValueError(
                f"detected_issues imply agents {sorted(a.value for a in missing)} "
                f"but they are missing from agents_to_call {self.agents_to_call}"
            )
        return self


class QueryType(str, Enum):
    REVERSE_IMAGE = "reverse_image"
    GENERATOR_MATCH = "generator_match"
    WEB_SEARCH = "web_search"


# Lets the Retrieval Agent turn a Planner-flagged retrieval category into a sensible
# default query type when crafting its own queries — the "consensus" link between
# what the Planner noticed and what Retrieval actually goes and does.
RETRIEVAL_CATEGORY_TO_QUERY_TYPE: Dict[IssueCategory, QueryType] = {
    IssueCategory.GENERATOR_MODEL_MATCH: QueryType.GENERATOR_MATCH,
    IssueCategory.PATTERN_COMPARISON: QueryType.GENERATOR_MATCH,
    IssueCategory.REVERSE_IMAGE_CANDIDATE: QueryType.REVERSE_IMAGE,
}


class RetrievalQuery(BaseModel):
    query_text: str = Field(..., min_length=1)
    query_type: QueryType
    top_k: int = Field(5, ge=1, le=50)


class QueryPlan(BaseModel):
    """Structured output required from the Retrieval Agent's query-crafting step."""

    queries_used: List[RetrievalQuery] = Field(..., min_length=1)


class RetrievalMatch(BaseModel):
    source: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    label: str = Field(..., description="'real' | 'fake' | 'unknown'")
    metadata: Dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    queries_used: List[RetrievalQuery]
    matches: List[RetrievalMatch]
    summary: str
