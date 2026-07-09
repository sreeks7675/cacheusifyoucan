import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from agent_a.backbone.slm_client import SLMClient


class FakeSLMClient(SLMClient):
    """Returns pre-programmed responses in order. Lets tests exercise agent logic
    without loading any real model, GPU, torch, or vision-processor install."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_structured(self, system_prompt, user_prompt, schema, image=None, max_retries=2):
        self.calls.append((system_prompt, user_prompt, schema, image))
        if not self._responses:
            raise RuntimeError("FakeSLMClient has no more programmed responses")
        next_response = self._responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return schema.model_validate(next_response)


@pytest.fixture
def fake_slm_client():
    return FakeSLMClient


@pytest.fixture
def sample_image_path(tmp_path):
    """A tiny real PNG on disk, so PlannerAgent's metadata extraction step has an
    actual file to read (EXIF will be empty — that's a valid, testable case)."""
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(path)
    return path
