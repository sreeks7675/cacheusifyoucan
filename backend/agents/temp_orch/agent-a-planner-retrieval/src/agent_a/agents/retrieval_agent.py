from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

from agent_a.backbone.slm_client import SLMClient
from agent_a.schemas.task_plan import (
    RETRIEVAL_CATEGORY_TO_QUERY_TYPE,
    IssueCategory,
    QueryPlan,
    RetrievalMatch,
    RetrievalResult,
)
from agent_a.tools.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "retrieval_system_prompt.txt"


class RetrievalAgent:
    """Agent 4 — Retrieval & Comparison.

    Crafts search queries (reverse image, generator registry, web) via the SLM,
    then executes them against the vector store and returns ranked matches.
    See pipeline_training_and_orchestration.md Section 2 for what the vector
    store contains for this agent.
    """

    def __init__(
        self,
        llm_client: SLMClient,
        vector_store: VectorStoreClient,
        system_prompt: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.system_prompt = system_prompt or self._load_default_prompt()

    def investigate(
        self,
        forensic_findings: str,
        semantic_findings: str = "",
        trigger_categories: Optional[Sequence[IssueCategory]] = None,
    ) -> RetrievalResult:
        """`trigger_categories` are the retrieval-mapped IssueCategory values the
        Planner flagged (e.g. GENERATOR_MODEL_MATCH). Passing them keeps this
        agent's queries in consensus with *why* it was called in the first
        place, rather than re-deriving intent from findings text alone."""
        if not forensic_findings or not forensic_findings.strip():
            raise ValueError("forensic_findings must be non-empty — retrieval needs something to corroborate")

        query_plan = self._craft_queries(forensic_findings, semantic_findings, trigger_categories)

        matches: List[RetrievalMatch] = []
        for query in query_plan.queries_used:
            matches.extend(self.vector_store.search(query.query_text, top_k=query.top_k))

        return RetrievalResult(
            queries_used=query_plan.queries_used,
            matches=matches,
            summary=self._summarize(matches),
        )

    def _craft_queries(
        self,
        forensic_findings: str,
        semantic_findings: str,
        trigger_categories: Optional[Sequence[IssueCategory]],
    ) -> QueryPlan:
        hint = self._query_type_hint(trigger_categories)
        user_prompt = (
            f"Forensic findings so far: {forensic_findings}\n"
            f"Semantic findings so far: {semantic_findings or 'none'}\n"
            f"Planner flagged this case for retrieval because: {hint}\n"
            "Craft the minimum set of retrieval queries needed to corroborate or refute these findings."
        )
        return self.llm_client.generate_structured(self.system_prompt, user_prompt, QueryPlan)

    @staticmethod
    def _query_type_hint(trigger_categories: Optional[Sequence[IssueCategory]]) -> str:
        if not trigger_categories:
            return "not specified by planner"
        suggested = {
            RETRIEVAL_CATEGORY_TO_QUERY_TYPE[c].value
            for c in trigger_categories
            if c in RETRIEVAL_CATEGORY_TO_QUERY_TYPE
        }
        return f"categories={[c.value for c in trigger_categories]}, suggested query types={sorted(suggested)}"

    @staticmethod
    def _summarize(matches: List[RetrievalMatch]) -> str:
        if not matches:
            return "No corroborating or contradicting matches found."
        fake_hits = [m for m in matches if m.label == "fake" and m.similarity >= 0.75]
        if fake_hits:
            best = max(fake_hits, key=lambda m: m.similarity)
            return (
                f"{len(fake_hits)} high-similarity match(es) to known fakes; "
                f"best match: {best.source} ({best.similarity:.2f})"
            )
        return f"{len(matches)} match(es) found, none strongly indicative of a known fake."

    @staticmethod
    def _load_default_prompt() -> str:
        if _PROMPT_PATH.exists():
            return _PROMPT_PATH.read_text()
        logger.warning("Default retrieval prompt file not found at %s; using inline fallback.", _PROMPT_PATH)
        return (
            "You are the Retrieval & Comparison Agent in a deepfake forensic system. "
            "Given forensic and semantic findings, craft precise reverse-image, "
            "generator-registry, or web-search queries — only as many as needed."
        )
