# backend/orchestrator/nodes/report.py
import os
import json
from backend.orchestrator.state import PipelineState
# Import the actual generator function written by your teammate
from backend.agents.report.new_report_gen import generate_report_payload

def report_node(state: PipelineState) -> dict:
    print(f"\n--- [ORCHESTRATOR] Invoking Real Report Agent for Session: {state.session_id} ---")

    # 1. Structure the current pipeline state into the exact schema the report generator expects
    case_data = {
        "case_id": state.session_id,
        "case_meta": {
            "category": "Deepfake Detection Analysis",
            "investigator": f"User_{state.user_id}",
            "opened_at": state.final_decision.get("timestamp", "N/A"),
            "priority": "HIGH"
        },
        "planner": {
            "agents_invoked": state.selected_agents,
            "routing_path": " -> ".join(state.selected_agents) if state.selected_agents else "Direct Route"
        },
        "exhibits": [
            {
                "exhibit_id": f"EX-{state.session_id[:8]}",
                "filename": os.path.basename(state.image_path),
                "media_type": "image",
                "image_path": state.image_path,
                "decision": {
                    "final_verdict": state.final_decision.get("verdict", "UNKNOWN"),
                    "calibrated_confidence": state.final_decision.get("confidence", 0.0),
                    "risk_level": state.final_decision.get("risk_level", "MEDIUM"),
                    "threat_score": state.final_decision.get("threat_score", 5.0),
                    "confidence_interval": state.final_decision.get("confidence_interval", [0.0, 1.0]),
                    "decision_justification": state.final_decision.get("justification", "")
                },
                # Gather actual structural values accumulated from the running agents
                "forensic": {
                    "verdict": state.agent_responses.get("forensic").findings[0] if "forensic" in state.agent_responses else "N/A",
                    "artifact_findings": state.consolidated_evidence.get("artifact_findings", []),
                    "assets": state.consolidated_evidence.get("assets", {}),
                    "metadata_flags": state.consolidated_evidence.get("metadata_flags", [])
                },
                "semantic": {
                    "verdict": "FAKE" if state.agent_responses.get("semantic_context").raw_output.get("recommend_human_review") else "REAL",
                    "semantic_conflicts": state.consolidated_evidence.get("semantic_conflicts", [])
                },
                "retrieval": {
                    "matches": state.consolidated_evidence.get("matches", [])
                },
                "evidence_fusion": {
                    "agent_weights": state.consolidated_evidence.get("agent_weights", {}),
                    "detected_conflicts": state.consolidated_evidence.get("detected_conflicts", [])
                },
                "debate": state.consolidated_evidence.get("debate_data", {
                    "consensus": {"votes_for": 0, "votes_total": 0},
                    "counterfactuals": []
                })
            }
        ]
    }

    # 2. Set up the persistence path for execution
    output_dir = "backend/orchestrator/storage"
    os.makedirs(output_dir, exist_ok=True)
    
    source_json_path = os.path.join(output_dir, f"case_{state.session_id}.json")
    target_payload_path = os.path.join(output_dir, f"report_{state.session_id}_payload.json")

    # Save the consolidated pipeline state run file to disk so the report generator can process it
    with open(source_json_path, "w") as f:
        json.dump(case_data, f, indent=2)

    try:
        # 3. Natively invoke your teammate's actual operational pipeline function
        # base_images_dir is set to "." because state.image_path provides a complete path
        ui_payload = generate_report_payload(
            case_json_path=source_json_path,
            base_images_dir=".",
            output_json_path=target_payload_path
        )
        
        print(f"[SUCCESS] Real report payload outputted by agent to: {target_payload_path}")
        return {"report_path": target_payload_path}

    except Exception as e:
        print(f"[CRITICAL] Teammate report agent execution failed: {e}")
        return {"report_path": None}