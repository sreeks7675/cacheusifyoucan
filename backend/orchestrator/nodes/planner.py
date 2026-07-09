from __future__ import annotations

from backend.orchestrator.state import PipelineState


def planner_node(state: PipelineState) -> dict:
    print(f"\n--- [ORCHESTRATOR] Planner Stage for Session: {state.session_id} ---")

    # Default routing keeps full evidence coverage unless caller already specified a subset.
    selected = state.selected_agents or ["forensic", "semantic_context", "retrieval"]
    selected = [agent for agent in selected if agent in {"forensic", "semantic_context", "retrieval"}]

    if not selected:
        selected = ["retrieval"]

    rationale = (
        "Planner selected forensic, semantic_context, retrieval for multi-view evidence collection."
        if selected == ["forensic", "semantic_context", "retrieval"]
        else f"Planner honored requested agent subset: {', '.join(selected)}"
    )

    return {
        "selected_agents": selected,
        "routing_rationale": rationale,
        "debate_logs": [f"[Planner] selected={selected}"],
    }