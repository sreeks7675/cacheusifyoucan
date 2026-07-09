"""Runs the Planner -> Retrieval flow locally with a scripted fake SLM client and
a real (tiny, generated) image, so you can sanity-check the agent wiring without
a GPU, the real vision backbone, or the real vector store.

Once the real Adapter A checkpoint exists, swap FakeSLMClient for:
    RealSLMClient(base_model_name=..., adapter_a_path="checkpoints/adapter_a")

Run: PYTHONPATH=src python scripts/run_local_demo.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from conftest import FakeSLMClient  # reuse the same fake used in tests

from agent_a.agents.planner_agent import PlannerAgent
from agent_a.agents.retrieval_agent import RetrievalAgent
from agent_a.schemas.task_plan import IssueCategory, RetrievalMatch


class FakeVectorStore:
    def search(self, query_text, top_k=5):
        return [RetrievalMatch(source="demo_db", similarity=0.88, label="fake", metadata={"query": query_text})]


def main() -> None:
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "demo.png"
        Image.new("RGB", (64, 64), color=(90, 90, 90)).save(image_path)

        planner_client = FakeSLMClient(
            [
                {
                    "detected_issues": [
                        {
                            "category": "lighting_shadow_consistency",
                            "note": "shadow direction inconsistent with light source",
                            "region": "background",
                            "confidence": 0.75,
                        },
                        {
                            "category": "generator_model_match",
                            "note": "overall look resembles known diffusion output",
                            "region": None,
                            "confidence": 0.6,
                        },
                    ],
                    "agents_to_call": ["semantic_context", "retrieval_comparison"],
                    "parallelizable": [["semantic_context", "retrieval_comparison"]],
                    "reasoning": "Lighting flag routes to semantic context; generator-style "
                    "resemblance routes to retrieval for corroboration.",
                }
            ]
        )
        planner = PlannerAgent(llm_client=planner_client)
        plan = planner.plan(image=image_path)
        print("Planner looked at the actual image and decided to call:", plan.agents_to_call)
        print("Reasoning:", plan.reasoning)

        retrieval_client = FakeSLMClient(
            [
                {
                    "queries_used": [
                        {"query_text": "diffusion style match", "query_type": "generator_match", "top_k": 5}
                    ]
                }
            ]
        )
        retrieval = RetrievalAgent(llm_client=retrieval_client, vector_store=FakeVectorStore())
        result = retrieval.investigate(
            forensic_findings="no strong pixel-level artifact, but overall look is suspicious",
            trigger_categories=[IssueCategory.GENERATOR_MODEL_MATCH],
        )
        print("\nRetrieval summary:", result.summary)


if __name__ == "__main__":
    main()
