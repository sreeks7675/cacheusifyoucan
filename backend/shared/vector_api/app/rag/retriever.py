# backend/orchestrator/nodes/retrieval.py
import os
from backend.orchestrator.state import PipelineState, AgentAnalysis
# Grounded Import: Importing the real search function directly from your teammate's file
from backend.shared.vector_api.app.rag.retriever import search_knowledge

def retrieval_node(state: PipelineState) -> dict:
    print(f"\n--- [ORCHESTRATOR] Running ChromaDB Search via search_knowledge ---")
    
    try:
        # 1. Use the image file name or the planner's routing rationale as the semantic query string
        query_string = os.path.basename(state.image_path)
        if state.routing_rationale:
            query_string += f" {state.routing_rationale}"
            
        # 2. Call the real search_knowledge function natively
        # It handles querying the vector_store module and parsing distances internally
        matches = search_knowledge(
            query_text=query_string,
            top_k=3,
            category=None  # Can be filtered by KnowledgeCategory enum if required later
        )
        
        # 3. Process the explicit dictionary array keys returned by your teammate's function
        findings_list = []
        for match in matches:
            title = match.get("title", "Untitled Document")
            distance = match.get("distance", 0.0)
            findings_list.append(f"Semantic match found in '{title}' (Distance Vector: {round(distance, 4)})")
            
        if not findings_list:
            findings_list.append("No identical manipulation historical records or related cases matched in ChromaDB.")
            
        # 4. Map the data structure safely back to our LangGraph state contract
        analysis = AgentAnalysis(
            agent_name="Retrieval Agent",
            confidence_score=0.90 if matches else 0.50,
            findings=findings_list,
            raw_output={"vector_matches": matches}
        )
        
        updated_responses = state.agent_responses.copy()
        updated_responses["retrieval"] = analysis
        
        updated_evidence = state.consolidated_evidence.copy()
        updated_evidence["matches"] = matches
        
        return {
            "agent_responses": updated_responses,
            "consolidated_evidence": updated_evidence
        }
        
    except Exception as e:
        print(f"[Orchestrator Error] Native search_knowledge execution failed: {e}")
        error_analysis = AgentAnalysis(
            agent_name="Retrieval Agent",
            confidence_score=0.0,
            findings=[f"Vector search failed: {str(e)}"]
        )
        updated_responses = state.agent_responses.copy()
        updated_responses["retrieval"] = error_analysis
        return {"agent_responses": updated_responses}