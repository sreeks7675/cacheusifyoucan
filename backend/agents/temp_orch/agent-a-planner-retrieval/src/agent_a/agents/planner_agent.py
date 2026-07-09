from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from agent_a.backbone.slm_client import ImageInput, SLMClient
from agent_a.schemas.task_plan import TaskPlan

logger = logging.getLogger(__name__)
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "planner_system_prompt.txt"


class PlannerAgent:
    """Agent 1 — Investigation Planner.

    Looks at the image directly (no pre-summarized text context) and does a
    shallow, jack-of-all-trades scan across every category the deeper agents
    specialize in — never running the deep analysis itself, only deciding
    which of Forensic / Semantic / Retrieval are warranted. Evidence Fusion &
    Debate is a fixed step the orchestrator runs after these three, not a
    planner choice — see PlannerDispatchableAgent in schemas/task_plan.py.
    """

    def __init__(self, llm_client: SLMClient, system_prompt: Optional[str] = None):
        self.llm_client = llm_client
        self.system_prompt = system_prompt or self._load_default_prompt()

    def plan(
        self,
        image: ImageInput,
        user_request: str = "Investigate this image for manipulation.",
    ) -> TaskPlan:
        if image is None:
            raise ValueError("image must be provided — the planner reasons over the image directly")

        metadata_summary = self._extract_basic_metadata(image)
        user_prompt = (
            f"User request: {user_request}\n"
            f"Basic file metadata (cheap, deterministic — not a deep forensic pass): {metadata_summary}\n"
            "Look at the image yourself. For each category you notice something worth flagging, "
            "record it with a short note and rough confidence. Then decide which agent(s) to call."
        )
        plan = self.llm_client.generate_structured(self.system_prompt, user_prompt, TaskPlan, image=image)
        return plan

    @staticmethod
    def _extract_basic_metadata(image: ImageInput) -> str:
        """Cheap, deterministic EXIF/file read — deliberately NOT deep forensic
        metadata analysis (that's still Agent 2's job at a much closer level).
        This is one more surface-level signal alongside the visual scan, not a
        replacement for it. Safe to fail silently: absence of metadata is itself
        a valid (and mildly suspicious) signal."""
        if not isinstance(image, (str, Path)):
            return "not available (in-memory image, no file metadata to read)"
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            with Image.open(image) as img:
                exif = img.getexif()
                if not exif:
                    return "no EXIF data found"
                tags = {TAGS.get(k, k): v for k, v in exif.items()}
                interesting = {k: tags[k] for k in ("Software", "Make", "Model", "DateTime") if k in tags}
                return str(interesting) if interesting else "EXIF present but no notable fields"
        except Exception as exc:  # noqa: BLE001 — metadata read must never block the plan
            logger.warning("Metadata extraction failed (%s); continuing without it.", exc)
            return "unavailable (read error)"

    @staticmethod
    def _load_default_prompt() -> str:
        if _PROMPT_PATH.exists():
            return _PROMPT_PATH.read_text()
        logger.warning("Default planner prompt file not found at %s; using inline fallback.", _PROMPT_PATH)
        return (
            "You are the Investigation Planner Agent in a deepfake forensic system. "
            "Look at the image directly and flag surface-level issues across all "
            "categories, then decide which agents to call. Only call agents that "
            "are actually warranted by the evidence. Always include your reasoning."
        )
