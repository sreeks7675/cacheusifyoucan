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

Brownie-point coverage added in this version (all scoped to what a cheap,
deterministic, no-training agent can reasonably do):
  - Confidence & Uncertainty Estimation: overall_confidence + uncertainty_flag
  - Failure Analysis & Self-Reflection: recommend_human_review + limitations
  - Evidence-Based Visual Reasoning: annotated evidence image (lighting
    quadrant highlight + OCR text boxes) saved to disk
  - Critic / Reviewer Models: a second, cheap SLM pass that checks the
    explanation against the findings for unsupported claims
  - Robustness to Unseen Manipulations: re-run checks on a JPEG-recompressed
    in-memory copy of the image and flag if results are unstable

NOTE: This agent is dataset-agnostic — it does not assume the image it's
given is real or fake. It just runs checks and reports findings. Ground
truth (real/fake) is only used by the *test harness* for evaluation, and
is passed through here purely as an optional label for bookkeeping.

Swap the CONFIG block below once the planner's real dispatch contract and
the Ollama/vLLM endpoint are confirmed with the team.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor

import requests


# ---------------------------------------------------------------------------
# CONFIG - adjust to match actual infra (Muthu's SLM backbone serving setup)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi4-mini"  # or llama3.1:8b / qwen2.5:7b - whatever backbone is served

EVIDENCE_OUTPUT_DIR = "evidence_output"

# Candidate scene/object labels for zero-shot plausibility check.
SCENE_LABELS = [
    "an indoor scene", "an outdoor scene", "a studio portrait",
    "a natural daylight photo", "an artificially lit photo",
    "a photo with inconsistent shadows", "a photo with consistent lighting",
]

# Thresholds - tune against labeled real/fake examples.
CLIP_LOW_CONFIDENCE_THRESHOLD = 0.40
LIGHTING_VARIANCE_THRESHOLD = 400.0
ROBUSTNESS_JPEG_QUALITY = 50
ROBUSTNESS_VARIANCE_DELTA_THRESHOLD = 150.0


@dataclass
class SemanticFindings:
    clip_top_label: str
    clip_confidence: float
    ocr_text_detected: str
    ocr_suspicious: bool
    lighting_variance_score: float
    shadow_consistency_flag: bool

    # --- Confidence & Uncertainty Estimation ---
    overall_confidence: float = 0.0
    uncertainty_flag: bool = False

    # --- Failure Analysis & Self-Reflection ---
    recommend_human_review: bool = False
    limitations: Optional[str] = None

    # --- Evidence-Based Visual Reasoning ---
    evidence_image_path: Optional[str] = None

    # --- Robustness to Unseen Manipulations ---
    robustness_stable: Optional[bool] = None
    robustness_notes: Optional[str] = None

    raw_notes: Optional[str] = None


class SemanticContextAgent:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        os.makedirs(EVIDENCE_OUTPUT_DIR, exist_ok=True)

    # -- Step 1: cheap deterministic/vision checks --------------------------

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
        # pytesseract expects RGB (or grayscale); cv2 loads BGR by default.
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(rgb_image).strip()
        if text:
            alnum_ratio = sum(c.isalnum() or c.isspace() for c in text) / max(len(text), 1)
            suspicious = alnum_ratio < 0.6
        else:
            suspicious = False
        return text, suspicious

    def _lighting_shadow_check(self, cv_image: np.ndarray) -> tuple[float, bool, list[float]]:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        quads = [
            gray[0:h // 2, 0:w // 2], gray[0:h // 2, w // 2:w],
            gray[h // 2:h, 0:w // 2], gray[h // 2:h, w // 2:w],
        ]
        means = [float(q.mean()) for q in quads]
        variance_score = float(np.var(means))
        shadow_flag = variance_score > LIGHTING_VARIANCE_THRESHOLD
        return variance_score, shadow_flag, means

    # -- Brownie point: Evidence-Based Visual Reasoning ----------------------

    def _generate_evidence_image(self, image_path: str, cv_image: np.ndarray,
                                  quad_means: list[float]) -> Optional[str]:
        """
        Draw a highlight box around the quadrant with the most anomalous
        brightness (proxy for "where the lighting inconsistency evidence
        is"), plus boxes around any OCR-detected text regions. Saves an
        annotated copy so the report/UI has something visual to show,
        per the "Visual Evidence (Highlighted Regions)" output requirement.
        """
        try:
            annotated = cv_image.copy()
            h, w = cv_image.shape[:2]
            quad_boxes = [
                (0, 0, w // 2, h // 2),
                (w // 2, 0, w, h // 2),
                (0, h // 2, w // 2, h),
                (w // 2, h // 2, w, h),
            ]
            avg = sum(quad_means) / len(quad_means)
            most_anomalous_idx = int(np.argmax([abs(m - avg) for m in quad_means]))
            x1, y1, x2, y2 = quad_boxes[most_anomalous_idx]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, "lighting anomaly", (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # OCR text bounding boxes (green)
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)
            for i, conf in enumerate(data.get("conf", [])):
                try:
                    conf_val = float(conf)
                except (TypeError, ValueError):
                    continue
                if conf_val > 30 and data["text"][i].strip():
                    tx, ty, tw, th = (data["left"][i], data["top"][i],
                                       data["width"][i], data["height"][i])
                    cv2.rectangle(annotated, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(EVIDENCE_OUTPUT_DIR, f"{base_name}_evidence.jpg")
            cv2.imwrite(out_path, annotated)
            return out_path
        except Exception:
            # Evidence image is a bonus, not a hard dependency -- never let
            # this break the core findings pipeline.
            return None

    # -- Brownie point: Robustness to Unseen Manipulations -------------------

    def _robustness_check(self, cv_image: np.ndarray, clip_label: str,
                           lighting_var: float) -> tuple[bool, str]:
        """
        Re-run the cheap checks on a JPEG-recompressed in-memory copy of the
        image and compare results to the original. If the scene label
        flips or the lighting variance swings a lot under a routine
        compression pass, that's a signal this image's "evidence" isn't
        robust -- worth surfacing rather than reporting blind confidence.
        """
        try:
            success, encoded = cv2.imencode(
                ".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, ROBUSTNESS_JPEG_QUALITY]
            )
            if not success:
                return True, "Robustness check skipped (re-encode failed)."

            recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            pil_recompressed = Image.fromarray(cv2.cvtColor(recompressed, cv2.COLOR_BGR2RGB))

            re_label, _ = self._clip_scene_check(pil_recompressed)
            re_var, _, _ = self._lighting_shadow_check(recompressed)

            label_stable = (re_label == clip_label)
            var_delta = abs(re_var - lighting_var)
            var_stable = var_delta <= ROBUSTNESS_VARIANCE_DELTA_THRESHOLD

            stable = label_stable and var_stable
            notes = (
                f"scene label {'stable' if label_stable else f'changed to \"{re_label}\"'} "
                f"under JPEG q={ROBUSTNESS_JPEG_QUALITY} recompression; "
                f"lighting variance delta={round(var_delta, 2)} "
                f"({'within' if var_stable else 'exceeds'} threshold "
                f"{ROBUSTNESS_VARIANCE_DELTA_THRESHOLD})."
            )
            return stable, notes
        except Exception as e:
            return True, f"Robustness check skipped due to error: {e}"

    # -- Brownie point: Confidence & Uncertainty / Self-Reflection -----------

    def _assess_confidence_and_limitations(
        self, clip_conf: float, ocr_suspicious: bool, shadow_flag: bool,
        robustness_stable: Optional[bool]
    ) -> tuple[float, bool, bool, str]:
        """
        Combine per-check signals into one overall confidence score and
        decide whether this case should be flagged for human review, with
        a plain-language note on why the agent is or isn't confident.
        """
        overall_confidence = clip_conf
        reasons = []

        uncertainty_flag = clip_conf < CLIP_LOW_CONFIDENCE_THRESHOLD
        if uncertainty_flag:
            reasons.append(
                f"CLIP scene classification confidence is low ({clip_conf})."
            )

        conflicting_signals = ocr_suspicious and shadow_flag
        if conflicting_signals:
            reasons.append(
                "Both OCR-suspicious text and lighting-inconsistency flags "
                "fired together, which this agent cannot itself adjudicate."
            )
            overall_confidence *= 0.7

        if robustness_stable is False:
            reasons.append(
                "Findings were not stable under a routine JPEG recompression "
                "test, so this evidence may not generalize."
            )
            overall_confidence *= 0.7

        recommend_human_review = uncertainty_flag or conflicting_signals or (robustness_stable is False)

        limitations = (
            "No limitations flagged; checks were consistent and confident."
            if not reasons else
            "Limitations: " + " ".join(reasons) +
            " This agent only performs scene/OCR/lighting heuristics -- "
            "it does not verify facial biometrics, frequency-domain "
            "artifacts, or source attribution, which are other agents' jobs."
        )

        return round(float(overall_confidence), 4), uncertainty_flag, recommend_human_review, limitations

    def run_checks(self, image_path: str) -> SemanticFindings:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        cv_image = cv2.imread(image_path)
        if cv_image is None:
            raise ValueError(
                f"cv2 could not read '{image_path}'. File may be corrupt, "
                f"an unsupported format, or a truncated download. "
                f"(.webp support depends on your OpenCV build.)"
            )

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"PIL could not open '{image_path}': {e}")

        clip_label, clip_conf = self._clip_scene_check(pil_image)
        ocr_text, ocr_suspicious = self._ocr_check(cv_image)
        lighting_var, shadow_flag, quad_means = self._lighting_shadow_check(cv_image)

        robustness_stable, robustness_notes = self._robustness_check(
            cv_image, clip_label, lighting_var
        )

        overall_confidence, uncertainty_flag, recommend_human_review, limitations = (
            self._assess_confidence_and_limitations(
                clip_conf, ocr_suspicious, shadow_flag, robustness_stable
            )
        )

        evidence_image_path = self._generate_evidence_image(image_path, cv_image, quad_means)

        return SemanticFindings(
            clip_top_label=clip_label,
            clip_confidence=round(clip_conf, 4),
            ocr_text_detected=ocr_text,
            ocr_suspicious=ocr_suspicious,
            lighting_variance_score=round(lighting_var, 2),
            shadow_consistency_flag=shadow_flag,
            overall_confidence=overall_confidence,
            uncertainty_flag=uncertainty_flag,
            recommend_human_review=recommend_human_review,
            limitations=limitations,
            evidence_image_path=evidence_image_path,
            robustness_stable=robustness_stable,
            robustness_notes=robustness_notes,
        )

    # -- Step 2: hand findings to SLM backbone for explanation ---------------

    def explain(self, findings: SemanticFindings) -> str:
        """
        Prompt-based fallback (Adapter B role) - turns structured findings into
        a human-readable finding. Swap for a LoRA-tuned call later if Session 3
        finishes early and real fine-tuning happens.
        """
        system_prompt = (
            "You are the Semantic & Context forensic agent in a deepfake "
            "investigation pipeline. You are given structured numeric/text "
            "findings from vision checks (CLIP scene classification, OCR, "
            "lighting/shadow variance, a robustness re-check, and a "
            "confidence/limitations assessment). Write a concise, specific "
            "forensic finding (3-5 sentences) describing what the evidence "
            "suggests about scene consistency, text plausibility, lighting, "
            "and overall reliability of this evidence. Explicitly mention if "
            "human review is recommended and why. Do not invent detail not "
            "present in the findings. Use forensic terminology (e.g. "
            "'consistent with', 'anomalous', 'implausible', 'inconclusive')."
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

    # -- Brownie point: Critic / Reviewer Model -------------------------------

    def critique(self, findings: SemanticFindings, explanation: str) -> dict:
        """
        A second, cheap SLM pass that reviews the generated explanation
        against the raw findings and flags unsupported claims -- a minimal
        critic/reviewer step, distinct from the fusion/debate agent that
        will later reconcile evidence across all 8 agents.
        """
        critic_prompt = (
            "You are a critic reviewing another AI agent's forensic finding "
            "for a deepfake investigation. You are given the RAW FINDINGS "
            "(ground truth data) and the EXPLANATION the agent wrote from "
            "them. Check only whether the explanation is fully supported by "
            "the raw findings -- flag any claim that isn't backed by the "
            "data, or any important finding the explanation ignored. "
            "Respond in EXACTLY this format:\n"
            "VERDICT: PASS or FLAG\n"
            "NOTES: <one or two sentences>"
        )
        user_prompt = (
            f"RAW FINDINGS:\n{json.dumps(asdict(findings), indent=2)}\n\n"
            f"EXPLANATION:\n{explanation}"
        )

        try:
            response = requests.post(
                OLLAMA_ENDPOINT,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{critic_prompt}\n\n{user_prompt}",
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()

            verdict = "UNKNOWN"
            notes = text
            for line in text.splitlines():
                if line.upper().startswith("VERDICT:"):
                    verdict = line.split(":", 1)[1].strip().upper()
                elif line.upper().startswith("NOTES:"):
                    notes = line.split(":", 1)[1].strip()

            return {"verdict": verdict, "notes": notes}
        except Exception as e:
            return {"verdict": "SKIPPED", "notes": f"Critic pass failed: {e}"}

    # -- Entry point matching planner's dispatch contract ---------------------

    def investigate(self, image_path: str, true_label: Optional[str] = None,
                     run_critic: bool = True) -> dict:
        """
        Run the full pipeline on a single image.

        true_label: optional ground-truth tag ("real" / "fake" / None).
            This agent makes NO real/fake decision itself and does not use
            this value in any check -- it's passed through purely so a test
            harness (or the planner, during eval) can attach ground truth
            to the output for later comparison. Safe to omit entirely in
            production/inference where ground truth isn't known.
        run_critic: set False to skip the extra critic LLM call (e.g. to
            save time/tokens while iterating).
        """
        findings = self.run_checks(image_path)
        explanation = self.explain(findings)

        result = {
            "agent": "semantic_context",
            "image": image_path,
            "findings": asdict(findings),
            "explanation": explanation,
        }

        if run_critic:
            result["critic"] = self.critique(findings, explanation)

        if true_label is not None:
            result["true_label"] = true_label

        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        print("Usage: python semantic_agent.py <image_path> [real|fake]")
        sys.exit(1)

    label = sys.argv[2] if len(sys.argv) == 3 else None

    agent = SemanticContextAgent()
    result = agent.investigate(sys.argv[1], true_label=label)
    print(json.dumps(result, indent=2))
