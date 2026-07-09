"""Shared SLM backbone client.

Agents depend only on the `SLMClient` interface, never on torch/transformers/peft
directly. That's what makes the Planner and Retrieval agents unit-testable without
a GPU: tests inject a fake client (see tests/conftest.py) instead of RealSLMClient.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# Accept a path, raw bytes, or an already-loaded PIL.Image — agents shouldn't
# have to know which the caller has on hand.
ImageInput = Union[str, Path, bytes, Any]


class SLMClient(ABC):
    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        image: Optional[ImageInput] = None,
        max_retries: int = 2,
    ) -> T:
        """Generate model output constrained to `schema`, retrying on invalid JSON.

        `image` is optional — the Retrieval Agent never passes one (it reasons over
        text findings only); the Planner Agent always does. A text-only backend
        should raise if given an image it can't handle; a vision-capable backend
        (see RealSLMClient) uses the same processor for both cases.
        """
        raise NotImplementedError


class RealSLMClient(SLMClient):
    """Loads the shared base model once and hot-swaps the Adapter A LoRA weights.

    Backbone must be a small **vision-language** model (e.g. Qwen2-VL-7B-Instruct
    or Phi-3.5-vision-instruct) since the Planner needs to see the image directly —
    see configs/adapter_a.yaml. The same backbone handles the Retrieval Agent's
    text-only calls fine; VLMs' underlying LLM is still a full language model when
    no image is passed.

    Heavy ML dependencies (torch, transformers, peft) are imported lazily inside
    `_ensure_loaded`, so importing this module — and therefore importing the agent
    classes that depend on it — never requires a GPU or those packages to be
    installed. Only calling `generate_structured` for real does.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_a_path: str,
        device: str = "cuda",
        load_in_4bit: bool = True,
    ):
        self.base_model_name = base_model_name
        self.adapter_a_path = adapter_a_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self._processor = AutoProcessor.from_pretrained(self.base_model_name)
        base = AutoModelForImageTextToText.from_pretrained(
            self.base_model_name,
            quantization_config=quant_config,
            device_map=self.device,
        )
        self._model = PeftModel.from_pretrained(base, self.adapter_a_path)
        self._model.eval()
        logger.info("Loaded backbone %s + Adapter A from %s", self.base_model_name, self.adapter_a_path)

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        image: Optional[ImageInput] = None,
        max_retries: int = 2,
    ) -> T:
        self._ensure_loaded()
        import torch

        pil_image = _to_pil_image(image) if image is not None else None
        messages = self._build_messages(system_prompt, user_prompt, schema, has_image=pil_image is not None)
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            inputs = self._processor.apply_chat_template(
                messages,
                images=[pil_image] if pil_image is not None else None,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=attempt > 0,  # first attempt greedy, retries add diversity
                )
            raw = self._processor.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            try:
                data = json.loads(_extract_json(raw))
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning("Attempt %d: invalid structured output (%s), retrying", attempt + 1, exc)

        raise RuntimeError(
            f"SLM failed to produce valid {schema.__name__} after {max_retries + 1} attempts"
        ) from last_error

    @staticmethod
    def _build_messages(system_prompt: str, user_prompt: str, schema: Type[BaseModel], has_image: bool) -> list:
        schema_json = schema.model_json_schema()
        instructions = (
            f"{system_prompt}\n\nRespond ONLY with JSON matching this schema:\n{json.dumps(schema_json)}"
        )
        user_content = [{"type": "text", "text": user_prompt}]
        if has_image:
            user_content.insert(0, {"type": "image"})
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content},
        ]


def _to_pil_image(image: ImageInput):
    """Normalizes str/Path/bytes/PIL.Image into a PIL.Image the processor can use.

    Routes through agent_a.tools.image_preprocessing.preprocess_for_planner so
    inference-time images get the same EXIF-orientation-fix / bounded-resize /
    no-recompression treatment as training-time images. Skipping this here would
    create train/inference skew — the Planner would be evaluated on differently
    distributed images than it was fine-tuned on.
    """
    from agent_a.tools.image_preprocessing import preprocess_for_planner

    result = preprocess_for_planner(image)
    for w in result.warnings:
        logger.warning("image preprocessing: %s", w)
    return result.image


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no JSON object found in model output", text, 0)
    return text[start : end + 1]


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no JSON object found in model output", text, 0)
    return text[start : end + 1]
