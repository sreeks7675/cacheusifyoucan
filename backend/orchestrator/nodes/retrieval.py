# backend/orchestrator/nodes/retrieval.py
import os

from backend.orchestrator.state import PipelineState, AgentAnalysis

def retrieval_node(state: PipelineState) -> dict:
    print(f"\n--- [ORCHESTRATOR] Querying Vector Store RAG Layer for Session: {state.session_id} ---")
    
    try:
        # Import lazily so graph compilation survives optional/deep dependency issues.
        from backend.shared.vector_api.app.rag.retriever import search_knowledge

        query_string = os.path.basename(state.image_path)
        if state.routing_rationale:
            query_string += f" {state.routing_rationale}"

        historical_matches = search_knowledge(
            query_text=query_string,
            top_k=3,
            category=None,
        )
        
        # 3. Format into standard pipeline responses
        analysis = AgentAnalysis(
            agent_name="Retrieval Agent",
            confidence_score=0.90 if historical_matches else 0.50,
            findings=[f"Found {len(historical_matches)} context matches in database repository."],
            raw_output={"vector_matches": historical_matches},
        )
        
        updated_responses = state.agent_responses.copy()
        updated_responses["retrieval"] = analysis
        
        updated_evidence = state.consolidated_evidence.copy()
        updated_evidence["matches"] = historical_matches
        
        return {
            "agent_responses": updated_responses,
            "consolidated_evidence": updated_evidence
        }
        
    except Exception as e:
        print(f"[Orchestrator Error] RAG Retrieval pass failed: {e}")
        error_analysis = AgentAnalysis(
            agent_name="Retrieval Agent",
            confidence_score=0.0,
            findings=[f"RAG node exception: {str(e)}"]
        )
        updated_responses = state.agent_responses.copy()
        updated_responses["retrieval"] = error_analysis
        return {"agent_responses": updated_responses}