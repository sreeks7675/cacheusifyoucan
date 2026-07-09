# backend/orchestrator/state.py
from typing import Dict, List, Any, Optional, Annotated
from pydantic import BaseModel, Field
import operator


def _merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged

class AgentAnalysis(BaseModel):
    agent_name: str
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    findings: List[str] = Field(default_factory=list)
    raw_output: Dict[str, Any] = Field(default_factory=dict)

class PipelineState(BaseModel):
    # Initial Session Information (RBAC & Payload from Frontend)
    user_id: str
    session_id: str
    image_path: str
    
    # Preprocessing Output
    image_metadata: Dict[str, Any] = Field(default_factory=dict)
    fft_analysis_done: bool = False
    
    # Planner Decisions
    selected_agents: List[str] = Field(
        default_factory=list, 
        description="Determined by planner. Options: ['forensic', 'semantic_context', 'retrieval']"
    )
    routing_rationale: str = ""
    
    # Parallel Agents Collective Output
    agent_responses: Annotated[Dict[str, AgentAnalysis], _merge_dicts] = Field(default_factory=dict)
    
    # Evidence Fusion & Debate State
    debate_logs: Annotated[List[str], operator.add] = Field(default_factory=list)
    consolidated_evidence: Annotated[Dict[str, Any], _merge_dicts] = Field(default_factory=dict)
    
    # Final Decision & Audit Trail (The 'Brownie Points' Compliance Checklist)
    final_decision: Dict[str, Any] = Field(default_factory=dict)
    report_path: Optional[str] = None