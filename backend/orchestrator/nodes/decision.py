# backend/orchestrator/nodes/decision.py
from backend.orchestrator.state import PipelineState
# Grounded Import: Importing the real decision logic natively from your teammate's code
from backend.shared.vector_api.app.services.decision_agent import decide_case

def decision_node(state: PipelineState) -> dict:
    print(f"\n--- [ORCHESTRATOR] Invoking Real DecisionAgent Consensus Pass ---")

    # 1. Safely extract values accumulated from prior nodes to fit the teammate's exact schema
    forensic_data = state.agent_responses.get("forensic")
    semantic_data = state.agent_responses.get("semantic_context")
    retrieval_data = state.agent_responses.get("retrieval")

    # 2. Build the context payload structural contract required by the decide() logic
    teammate_context = {
        "planner": {
            "steps_selected": state.selected_agents
        },
        "forensic": {
            # Extract artifact_score from raw output or use confidence as proxy
            "artifact_score": state.consolidated_evidence.get("artifact_score", 0.5)
        },
        "semantic": {
            # Map clip metrics or custom layout parameters directly
            "scene_consistency_score": semantic_data.raw_output.get("clip_confidence", 0.5) if semantic_data else 0.5
        },
        "retrieval": {
            "known_source_match": state.consolidated_evidence.get("known_source_match", False),
            "similar_images_found": len(state.consolidated_evidence.get("matches", []))
        },
        "fusion": {
            "fused_fake_score": state.consolidated_evidence.get("fused_fake_score", 0.5),
            "fusion_confidence": state.consolidated_evidence.get("fusion_confidence"),
            "fusion_uncertainty": state.consolidated_evidence.get("fusion_uncertainty"),
            "fusion_risk_level": state.consolidated_evidence.get("fusion_risk_level"),
            "has_dissent": bool(
                state.consolidated_evidence.get("efda", {}).get("dissent")
            ),
        }
    }

    try:
        # 3. Compute verdict metrics natively through their execution module
        calibrated_output = decide_case(teammate_context)
        
        # 4. Extract and map the exact keys returned by their decide() payload structural dictionary
        final_verdict = {
            "verdict": calibrated_output.get("verdict", "uncertain").upper(),
            "confidence": float(calibrated_output.get("confidence_score", 0.0)),
            "confidence_interval": [
                calibrated_output.get("confidence_interval", {}).get("low", 0.0),
                calibrated_output.get("confidence_interval", {}).get("high", 1.0)
            ],
            "risk_level": calibrated_output.get("risk_level", "Medium").upper(),
            "threat_score": float(calibrated_output.get("threat_score", 5.0)),
            "justification": calibrated_output.get("decision_justification", ""),
            "timestamp": state.final_decision.get("timestamp", "N/A"),
            "calibration_ece": calibrated_output.get("calibration_ece"),
            "threshold_used": calibrated_output.get("threshold_used"),
            "uncertainty": calibrated_output.get("uncertainty", {}),
            "human_review_required": calibrated_output.get("human_review_required"),
            "review_reason": calibrated_output.get("review_reason")
        }

        print(f"[SUCCESS] Consensus computed: {final_verdict['verdict']} ({final_verdict['confidence'] * 100}%)")
        return {"final_decision": final_verdict}

    except Exception as e:
        print(f"[CRITICAL] Teammate DecisionAgent execution failed: {e}")
        # Fallback safety protocol matching structural parameters
        return {
            "final_decision": {
                "verdict": "UNCERTAIN",
                "confidence": 0.50,
                "risk_level": "MEDIUM",
                "threat_score": 5.0,
                "confidence_interval": [0.0, 1.0],
                "justification": f"Internal consensus evaluation failed: {str(e)}"
            }
        }