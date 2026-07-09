# backend/orchestrator/nodes/retrieval.py
from backend.orchestrator.state import PipelineState, AgentAnalysis
# Import the real operational RAG modules from your vector_api subtree
from backend.shared.vector_api.app.rag.retriever import VectorRetriever
from backend.shared.vector_api.app.rag.vector_store import VectorStoreClient

def retrieval_node(state: PipelineState) -> dict:
    print(f"\n--- [ORCHESTRATOR] Querying Vector Store RAG Layer for Session: {state.session_id} ---")
    
    try:
        # 1. Instantiate their database connection layers natively
        store_client = VectorStoreClient()
        retriever = VectorRetriever(store=store_client)
        
        # 2. Run a similarity lookup against historical deepfake data using the current image path
        historical_matches = retriever.find_similar_cases(image_source=state.image_path)
        
        # 3. Format into standard pipeline responses
        analysis = AgentAnalysis(
            agent_name="Retrieval Agent",
            confidence_score=0.90 if historical_matches else 0.50,
            findings=[f"Found {len(historical_matches)} context matches in database repository."]
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