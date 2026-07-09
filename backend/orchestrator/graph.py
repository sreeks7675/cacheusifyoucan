# backend/orchestrator/graph.py
from langgraph.graph import StateGraph, END
from backend.orchestrator.state import PipelineState

# Import node execution functions
from backend.orchestrator.nodes.preprocess import preprocess_node
from backend.orchestrator.nodes.planner import planner_node
from backend.orchestrator.nodes.forensic import forensic_node
from backend.orchestrator.nodes.semantic_context import semantic_context_node
from backend.orchestrator.nodes.retrieval import retrieval_node
from backend.orchestrator.nodes.fusion_debate import fusion_debate_node
from backend.orchestrator.nodes.decision import decision_node
from backend.orchestrator.nodes.report import report_node

def should_route_agents(state: PipelineState) -> list[str]:
    """
    Conditional routing logic. Inspects the planner's selected targets 
    and spawns parallel worker paths.
    """
    destinations = []
    if "forensic" in state.selected_agents:
        destinations.append("forensic_agent_node")
    if "semantic_context" in state.selected_agents:
        destinations.append("semantic_context_agent_node")
    if "retrieval" in state.selected_agents:
        destinations.append("retrieval_agent_node")
        
    # Fallback to avoid breaking the graph if the planner selects nothing
    if not destinations:
        return ["fusion_debate_node"]
    return destinations

# Initialize the StateGraph using our Pydantic schema
workflow = StateGraph(PipelineState)

# Add all structural nodes to the graph
workflow.add_node("preprocess_node", preprocess_node)
workflow.add_node("planner_node", planner_node)
workflow.add_node("forensic_agent_node", forensic_node)
workflow.add_node("semantic_context_agent_node", semantic_context_node)
workflow.add_node("retrieval_agent_node", retrieval_node)
workflow.add_node("fusion_debate_node", fusion_debate_node)
workflow.add_node("decision_node", decision_node)
workflow.add_node("report_node", report_node)

# Set Entry Point
workflow.set_entry_point("preprocess_node")

# Linear Pipeline Start
workflow.add_edge("preprocess_node", "planner_node")

# Conditional Split (The Forking Mechanism)
workflow.add_conditional_edges(
    "planner_node",
    should_route_agents,
    {
        "forensic_agent_node": "forensic_agent_node",
        "semantic_context_agent_node": "semantic_context_agent_node",
        "retrieval_agent_node": "retrieval_agent_node",
        "fusion_debate_node": "fusion_debate_node" # direct pass option
    }
)

# Join Joins back to the Fusion Debate node
workflow.add_edge("forensic_agent_node", "fusion_debate_node")
workflow.add_edge("semantic_context_agent_node", "fusion_debate_node")
workflow.add_edge("retrieval_agent_node", "fusion_debate_node")

# Linear Path to Conclusion
workflow.add_edge("fusion_debate_node", "decision_node")
workflow.add_edge("decision_node", "report_node")
workflow.add_edge("report_node", END)

# Compile the execution graph
app = workflow.compile()