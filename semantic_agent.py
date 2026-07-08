"""
Semantic & Context Agent (Agent 3)
-----------------------------------
Role (per design_document.md / EADIS slide):
  - Object & scene consistency
  - CLIP / VLM semantic check
  - Text in image (OCR)
  - Context vs. real-world plausibility
  - Lighting & shadow consistency

Pattern: cheap deterministic/vision checks -> structured JSON findings ->
hand to shared SLM backbone (Adapter B role) with a tight system prompt to
turn scores into a human-readable finding. No LoRA yet -> prompt-based
fallback, per the design doc's timeline-risk mitigation.

Swap the CONFIG block below once the planner's real dispatch contract and
the Ollama/vLLM endpoint are confirmed with the team.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor

import requests


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi4-mini"

SCENE_LABELS = [
    "an indoor scene", "an outdoor scene", "a studio portrait",
    "a natural daylight photo", "an artificially lit photo",
    "a photo with inconsistent shadows", "a photo with consistent lighting",
]


@dataclass
class SemanticFindings:
    clip_top_label: str
    clip_confidence: float
    ocr_text_detected: str
    ocr_suspicious: bool
    lighting_variance_score: float
    shadow_consistency_flag: bool
    raw_notes: Optional[str] = None


class SemanticContextAgent:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    def _clip_scene_check(self, image: Image.Image) -> tuple[str, float]:
        inputs = self.clip_processor(
            text=SCENE_LABELS, images=image, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_idx = int(probs.argmax())
        return SCENE_LABELS[best_idx], float(probs[best_idx])

    def _ocr_check(self, cv_image: np.ndarray) -> tuple[str, bool]:
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(rgb_image).strip()
        if text:
            alnum_ratio = sum(c.isalnum() or c.isspace() for c in text) / max(len(text), 1)
            suspicious = alnum_ratio < 0.6
        else:
            suspicious = False
        return text, suspicious

    def _lighting_shadow_check(self, cv_image: np.ndarray) -> tuple[float, bool]:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        quads = [
            gray[0:h // 2, 0:w // 2], gray[0:h // 2, w // 2:w],
            gray[h // 2:h, 0:w // 2], gray[h // 2:h, w // 2:w],
        ]
        means = [float(q.mean()) for q in quads]
        variance_score = float(np.var(means))
        shadow_flag = variance_score > 400.0
        return variance_score, shadow_flag

    def run_checks(self, image_path: str) -> SemanticFindings:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        cv_image = cv2.imread(image_path)
        if cv_image is None:
            raise ValueError(
                f"cv2 could not read '{image_path}'. File may be corrupt, "
                f"an unsupported format, or a truncated download."
            )

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"PIL could not open '{image_path}': {e}")

        clip_label, clip_conf = self._clip_scene_check(pil_image)
        ocr_text, ocr_suspicious = self._ocr_check(cv_image)
        lighting_var, shadow_flag = self._lighting_shadow_check(cv_image)

        return SemanticFindings(
            clip_top_label=clip_label,
            clip_confidence=round(clip_conf, 4),
            ocr_text_detected=ocr_text,
            ocr_suspicious=ocr_suspicious,
            lighting_variance_score=round(lighting_var, 2),
            shadow_consistency_flag=shadow_flag,
        )

    def explain(self, findings: SemanticFindings) -> str:
        system_prompt = (
            "You are the Semantic & Context forensic agent in a deepfake "
            "investigation pipeline. You are given structured numeric/text "
            "findings from vision checks (CLIP scene classification, OCR, "
            "lighting/shadow variance). Write a concise, specific forensic "
            "finding (2-4 sentences) describing what the evidence suggests "
            "about scene consistency, text plausibility, and lighting. "
            "Do not invent detail not present in the findings. Use forensic "
            "terminology (e.g. 'consistent with', 'anomalous', 'implausible')."
        )
        user_prompt = f"Findings:\n{json.dumps(asdict(findings), indent=2)}"

        response = requests.post(
            OLLAMA_ENDPOINT,
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def investigate(self, image_path: str) -> dict:
        findings = self.run_checks(image_path)
        explanation = self.explain(findings)
        return {
            "agent": "semantic_context",
            "findings": asdict(findings),
            "explanation": explanation,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python semantic_agent.py <image_path>")
        sys.exit(1)

    agent = SemanticContextAgent()
    result = agent.investigate(sys.argv[1])
    print(json.dumps(result, indent=2))
