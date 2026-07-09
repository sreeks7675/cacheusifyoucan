import pytest
from pydantic import ValidationError

from agent_a.schemas.task_plan import (
    DetectedIssue,
    IssueCategory,
    PlannerDispatchableAgent,
    QueryPlan,
    QueryType,
    TaskPlan,
)


def test_task_plan_requires_at_least_one_agent():
    with pytest.raises(ValidationError):
        TaskPlan(agents_to_call=[], reasoning="no agents")


def test_task_plan_valid_minimal():
    plan = TaskPlan(agents_to_call=[PlannerDispatchableAgent.FORENSIC], reasoning="obvious artifact")
    assert plan.agents_to_call == [PlannerDispatchableAgent.FORENSIC]
    assert plan.detected_issues == []


def test_task_plan_requires_reasoning():
    with pytest.raises(ValidationError):
        TaskPlan(agents_to_call=[PlannerDispatchableAgent.FORENSIC], reasoning="")


def test_task_plan_rejects_agents_outside_dispatchable_set():
    with pytest.raises(ValidationError):
        TaskPlan(agents_to_call=["evidence_fusion_debate"], reasoning="not allowed")


def test_task_plan_consistency_validator_catches_missing_agent():
    """A category was flagged but its owning agent wasn't dispatched — must fail."""
    with pytest.raises(ValidationError, match="forensic_analysis"):
        TaskPlan(
            detected_issues=[
                DetectedIssue(category=IssueCategory.BLENDING_BOUNDARY, note="visible seam", confidence=0.7)
            ],
            agents_to_call=[PlannerDispatchableAgent.SEMANTIC],  # wrong agent for this category
            reasoning="inconsistent on purpose",
        )


def test_task_plan_consistency_validator_passes_when_covered():
    plan = TaskPlan(
        detected_issues=[
            DetectedIssue(category=IssueCategory.LIGHTING_SHADOW_CONSISTENCY, note="odd shadow", confidence=0.6)
        ],
        agents_to_call=[PlannerDispatchableAgent.SEMANTIC],
        reasoning="lighting flag routed correctly",
    )
    assert plan.agents_to_call == [PlannerDispatchableAgent.SEMANTIC]


def test_detected_issue_confidence_bounds():
    with pytest.raises(ValidationError):
        DetectedIssue(category=IssueCategory.NOISE_COMPRESSION, note="blocky", confidence=1.5)


def test_detected_issue_rejects_empty_note():
    with pytest.raises(ValidationError):
        DetectedIssue(category=IssueCategory.NOISE_COMPRESSION, note="   ", confidence=0.5)


def test_query_plan_requires_at_least_one_query():
    with pytest.raises(ValidationError):
        QueryPlan(queries_used=[])


def test_query_type_rejects_unknown_value():
    with pytest.raises(ValidationError):
        QueryPlan(queries_used=[{"query_text": "x", "query_type": "not_a_real_type", "top_k": 5}])


def test_query_type_accepts_known_values():
    plan = QueryPlan(queries_used=[{"query_text": "x", "query_type": QueryType.REVERSE_IMAGE, "top_k": 5}])
    assert plan.queries_used[0].query_type == QueryType.REVERSE_IMAGE
