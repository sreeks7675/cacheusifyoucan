# backend/orchestrator/nodes/semantic_context.py
import os
from backend.orchestrator.state import PipelineState, AgentAnalysis
from backend.agents.semantic_context.semantic_agent import SemanticContextAgent

def semantic_context_node(state: PipelineState) -> dict:
    print("\n=======================================================")
    print(f"--- [ORCHESTRATOR] Activating Semantic & Context Agent ---")
    print(f"Target Image: {state.image_path}")
    print("=======================================================")

    # 1. Initialize your teammate's agent class safely
    # It auto-detects CUDA/CPU internally.
    try:
        agent = SemanticContextAgent()
    except Exception as e:
        print(f"[Orchestrator Error] Failed to initialize Semantic Agent: {e}")
        # Return fallback gracefully so the entire graph doesn't crash if CUDA/dependencies fail
        fallback_analysis = AgentAnalysis(
            agent_name="Semantic & Context Agent",
            confidence_score=0.0,
            findings=[f"Initialization failed: {str(e)}"]
        )
        updated_responses = state.agent_responses.copy()
        updated_responses["semantic_context"] = fallback_analysis
        return {"agent_responses": updated_responses}

    # 2. Call the structural execution method matching their dispatch contract
    try:
        # We run the full pipeline, including the internal SLM explanation and critic review
        raw_result = agent.investigate(
            image_path=state.image_path, 
            run_critic=True
        )
        
        findings_data = raw_result.get("findings", {})
        explanation = raw_result.get("explanation", "")
        critic_data = raw_result.get("critic", {})

        # 3. Extract and map findings directly to our unified Pipeline Schema
        analysis = AgentAnalysis(
            agent_name="Semantic & Context Agent",
            confidence_score=float(findings_data.get("overall_confidence", 0.0)),
            findings=[
                f"Top Scene Label: {findings_data.get('clip_top_label')}",
                f"Lighting Variance Score: {findings_data.get('lighting_variance_score')}",
                f"Shadow Consistency Flag: {findings_data.get('shadow_consistency_flag')}",
                f"Robustness Check: {findings_data.get('robustness_notes')}"
            ],
            raw_output={
                "explanation": explanation,
                "critic_verdict": critic_data.get("verdict"),
                "critic_notes": critic_data.get("notes"),
                "ocr_text": findings_data.get("ocr_text_detected"),
                "recommend_human_review": findings_data.get("recommend_human_review"),
                "limitations": findings_data.get("limitations"),
                "evidence_image_path": findings_data.get("evidence_image_path")
            }
        )
        
        # 4. Update the global state dictionary tracking teammate responses
        updated_responses = state.agent_responses.copy()
        updated_responses["semantic_context"] = analysis

        # 5. Append compliance logs to the global debate framework
        new_logs = [
            f"[Semantic Agent Pass] Verdict: {critic_data.get('verdict')}. Review Required: {findings_data.get('recommend_human_review')}"
        ]

        # Merge extracted metadata into consolidated evidence mapping
        updated_evidence = state.consolidated_evidence.copy()
        updated_evidence["semantic_data"] = findings_data

        return {
            "agent_responses": updated_responses,
            "debate_logs": new_logs,
            "consolidated_evidence": updated_evidence
        }

    except Exception as e:
        print(f"[Orchestrator Error] Execution failed during agent run: {e}")
        error_analysis = AgentAnalysis(
            agent_name="Semantic & Context Agent",
            confidence_score=0.0,
            findings=[f"Execution failed: {str(e)}"]
        )
        updated_responses = state.agent_responses.copy()
        updated_responses["semantic_context"] = error_analysis
        return {"agent_responses": updated_responses}