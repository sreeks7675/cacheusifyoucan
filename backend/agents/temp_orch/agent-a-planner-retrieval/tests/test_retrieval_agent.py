import pytest

from agent_a.agents.retrieval_agent import RetrievalAgent
from agent_a.schemas.task_plan import IssueCategory, RetrievalMatch


class FakeVectorStore:
    def __init__(self, hits_by_query=None):
        self.hits_by_query = hits_by_query or {}
        self.searched = []

    def search(self, query_text, top_k=5):
        self.searched.append(query_text)
        return self.hits_by_query.get(query_text, [])


def test_retrieval_agent_crafts_queries_and_searches(fake_slm_client):
    query_response = {
        "queries_used": [
            {"query_text": "diffusion artifact match", "query_type": "generator_match", "top_k": 5},
        ],
    }
    client = fake_slm_client([query_response])
    hits = {
        "diffusion artifact match": [
            RetrievalMatch(source="known_fakes_db", similarity=0.91, label="fake", metadata={}),
        ]
    }
    store = FakeVectorStore(hits_by_query=hits)
    agent = RetrievalAgent(llm_client=client, vector_store=store, system_prompt="test prompt")

    result = agent.investigate(forensic_findings="frequency-domain diffusion signature")

    assert store.searched == ["diffusion artifact match"]
    assert len(result.matches) == 1
    assert "high-similarity match" in result.summary


def test_retrieval_agent_handles_no_matches(fake_slm_client):
    query_response = {
        "queries_used": [{"query_text": "no hits query", "query_type": "web_search", "top_k": 3}]
    }
    client = fake_slm_client([query_response])
    store = FakeVectorStore()
    agent = RetrievalAgent(llm_client=client, vector_store=store, system_prompt="test prompt")

    result = agent.investigate(forensic_findings="inconclusive")

    assert result.matches == []
    assert "No corroborating" in result.summary


def test_retrieval_agent_rejects_empty_forensic_findings(fake_slm_client):
    client = fake_slm_client([])
    store = FakeVectorStore()
    agent = RetrievalAgent(llm_client=client, vector_store=store, system_prompt="test prompt")

    with pytest.raises(ValueError):
        agent.investigate(forensic_findings="")


def test_retrieval_agent_multiple_queries_aggregate_matches(fake_slm_client):
    query_response = {
        "queries_used": [
            {"query_text": "query one", "query_type": "generator_match", "top_k": 5},
            {"query_text": "query two", "query_type": "reverse_image", "top_k": 5},
        ],
    }
    client = fake_slm_client([query_response])
    hits = {
        "query one": [RetrievalMatch(source="db_a", similarity=0.4, label="unknown", metadata={})],
        "query two": [RetrievalMatch(source="db_b", similarity=0.5, label="real", metadata={})],
    }
    store = FakeVectorStore(hits_by_query=hits)
    agent = RetrievalAgent(llm_client=client, vector_store=store, system_prompt="test prompt")

    result = agent.investigate(forensic_findings="some findings")

    assert len(result.matches) == 2
    assert store.searched == ["query one", "query two"]


def test_retrieval_agent_passes_planner_trigger_categories_as_hint(fake_slm_client):
    """Consensus check: the categories the Planner flagged should reach the
    prompt used to craft queries, not get silently dropped."""
    query_response = {
        "queries_used": [{"query_text": "generator style match", "query_type": "generator_match", "top_k": 5}]
    }
    client = fake_slm_client([query_response])
    store = FakeVectorStore()
    agent = RetrievalAgent(llm_client=client, vector_store=store, system_prompt="test prompt")

    agent.investigate(
        forensic_findings="looks like known diffusion style",
        trigger_categories=[IssueCategory.GENERATOR_MODEL_MATCH],
    )

    _, user_prompt, _, _ = client.calls[0]
    assert "generator_model_match" in user_prompt
    assert "generator_match" in user_prompt  # the suggested query type derived from the category
