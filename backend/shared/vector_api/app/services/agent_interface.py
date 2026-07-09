"""
Agent interface layer.

This is the ONLY file you need to edit to plug in your real agents
(Planner, Forensic, Semantic, Retrieval, Fusion, Decision, Report).

Contract: every agent function takes
    (image_path: str, context: dict)
and returns a JSON-serializable dict. `context` accumulates the outputs
of all agents that ran before it, so e.g. the Fusion agent can read
`context["forensic"]`, `context["semantic"]`, `context["retrieval"]`.

Right now each function is a lightweight mock so the API is runnable
and testable end-to-end without the actual models wired up. Replace
the body of each function with a call into your real agent/model and
the rest of the backend (DB storage, API responses) needs no changes.
"""

import random
from typing import Any, Dict

from app.services.decision_agent import DecisionAgent


def run_planner_agent(image_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "agent_name": "planner",
        "data": {
            "steps_selected": [
                "forensic_artifact_scan",
                "semantic_consistency_check",
                "reverse_image_retrieval",
            ],
            "reasoning": "Standard investigation plan for a single static image.",
        },
    }


def run_forensic_agent(image_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: replace with real artifact / noise / metadata analysis
    # (e.g. ViT/CNN artifact classifier, ELA, metadata parser).
    return {
        "agent_name": "forensic",
        "data": {
            "artifact_score": round(random.uniform(0, 1), 3),
            "noise_inconsistency": round(random.uniform(0, 1), 3),
            "metadata_flags": [],
        },
    }


def run_semantic_agent(image_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: replace with real scene-consistency model (CLIP/DINO based).
    return {
        "agent_name": "semantic",
        "data": {
            "scene_consistency_score": round(random.uniform(0, 1), 3),
            "anomalies_detected": [],
        },
    }


def run_retrieval_agent(image_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: replace with real reverse-image-search + knowledge base lookup.
    return {
        "agent_name": "retrieval",
        "data": {
            "similar_images_found": 0,
            "known_source_match": None,
        },
    }


def run_fusion_agent(image_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combines evidence from forensic / semantic / retrieval agents.
    Replace with your real evidence-fusion / debate logic; this is a
    simple weighted-average placeholder so the pipeline is runnable.
    """
    forensic = context.get("forensic", {}).get("artifact_score", 0.5)
    semantic = context.get("semantic", {}).get("scene_consistency_score", 0.5)
    fused_fake_score = round((forensic * 0.6) + ((1 - semantic) * 0.4), 3)

    return {
        "agent_name": "fusion",
        "data": {
            "fused_fake_score": fused_fake_score,
            "evidence_summary": (
                f"Forensic artifact score {forensic}, "
                f"semantic consistency {semantic}."
            ),
        },
    }


def run_decision_agent(image_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Produces the final decision bundle from prior agent outputs."""
    decision = DecisionAgent().decide(context)
    return {
        "agent_name": "decision",
        "data": decision,
    }


def run_report_agent(image_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates the human-readable explainable report from all prior
    agent outputs. Replace with real Grad-CAM/SHAP overlay + LLM
    report-writer call.
    """
    decision = context.get("decision", {})
    fusion = context.get("fusion", {})

    return {
        "agent_name": "report",
        "data": {
            "explainable_reasoning": (
                f"Verdict '{decision.get('verdict')}' was reached with "
                f"{decision.get('confidence_score')} confidence, based on "
                f"fusion evidence, forensic artifact signals, semantic consistency, "
                f"and retrieval analysis. {decision.get('decision_justification')}"
            ),
            "recommendations": (
                "Escalate to a human forensic analyst for corroboration."
                if decision.get("human_review_required")
                else "No further action recommended."
            ),
            "highlighted_regions": [],  # Grad-CAM/SHAP bounding boxes go here
        },
    }


# Ordered pipeline: (context_key, function)
AGENT_PIPELINE = [
    ("planner", run_planner_agent),
    ("forensic", run_forensic_agent),
    ("semantic", run_semantic_agent),
    ("retrieval", run_retrieval_agent),
    ("fusion", run_fusion_agent),
    ("decision", run_decision_agent),
    ("report", run_report_agent),
]
