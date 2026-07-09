import pytest

from agent_a.agents.planner_agent import PlannerAgent


def test_planner_produces_task_plan_from_image(fake_slm_client, sample_image_path):
    response = {
        "detected_issues": [
            {
                "category": "lighting_shadow_consistency",
                "note": "shadow direction inconsistent with light source",
                "region": "background",
                "confidence": 0.8,
            }
        ],
        "agents_to_call": ["semantic_context"],
        "parallelizable": [["semantic_context"]],
        "reasoning": "Lighting/shadow flag routes to semantic context review.",
    }
    client = fake_slm_client([response])
    agent = PlannerAgent(llm_client=client, system_prompt="test prompt")

    plan = agent.plan(image=sample_image_path)

    assert plan.agents_to_call == ["semantic_context"]
    assert len(client.calls) == 1
    # confirm the image was actually passed through to the LLM client, not dropped
    _, _, _, passed_image = client.calls[0]
    assert passed_image == sample_image_path


def test_planner_rejects_missing_image(fake_slm_client):
    client = fake_slm_client([])
    agent = PlannerAgent(llm_client=client, system_prompt="test prompt")
    with pytest.raises(ValueError):
        agent.plan(image=None)


def test_planner_handles_image_with_no_exif_gracefully(fake_slm_client, sample_image_path):
    response = {
        "agents_to_call": ["forensic_analysis"],
        "reasoning": "no metadata, but visual artifact present",
    }
    client = fake_slm_client([response])
    agent = PlannerAgent(llm_client=client, system_prompt="test prompt")

    plan = agent.plan(image=sample_image_path)

    assert plan.agents_to_call == ["forensic_analysis"]
    _, user_prompt, _, _ = client.calls[0]
    assert "no EXIF data found" in user_prompt


def test_planner_uses_default_prompt_file_when_none_given(fake_slm_client, sample_image_path):
    response = {
        "agents_to_call": ["forensic_analysis"],
        "reasoning": "default prompt path exercised",
    }
    client = fake_slm_client([response])
    agent = PlannerAgent(llm_client=client)  # no system_prompt passed

    assert "Investigation Planner Agent" in agent.system_prompt
    agent.plan(image=sample_image_path)
    assert len(client.calls) == 1


def test_planner_metadata_extraction_handles_non_path_image(fake_slm_client):
    """In-memory images (e.g. bytes) skip EXIF reading rather than crashing."""
    response = {"agents_to_call": ["forensic_analysis"], "reasoning": "in-memory image case"}
    client = fake_slm_client([response])
    agent = PlannerAgent(llm_client=client, system_prompt="test prompt")

    plan = agent.plan(image=b"\x89PNGfakebytesfortest")

    assert plan.agents_to_call == ["forensic_analysis"]
    _, user_prompt, _, _ = client.calls[0]
    assert "not available (in-memory image" in user_prompt
