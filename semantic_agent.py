"""
Semantic & Context Agent (Agent 3)
-----------------------------------
Role (per design_document.md / EADIS slide):
  - Object & scene consistency
  - CLIP / VLM semantic check
  - Text in image (OCR)
  - Context vs. real-world plausibility
  - Lighting & shadow consistency

Output contract (per "Table 1 - What the Report Agent needs FROM each agent",
Semantic row):
    verdict, confidence, semantic_conflicts[] {type, description, region_bbox,
    severity}, clip_consistency_score, annotated overlay image (optional)

conflict "type" is constrained to the fixed enum handed down by the team:
    lighting | reflection | object_relation | scene_coherence | lip_sync

IMPORTANT / HONEST LIMITATION: that enum has no dedicated "text/OCR" bucket.
OCR-implausibility findings are therefore filed under "scene_coherence" -
see `limitations` in the output for this and other known gaps (reflection,
object_relation, lip_sync are not implemented in this version - stubbed for
future work per the design doc's timeline-risk mitigation).

This agent is dataset-agnostic - it does not assume the image it's given is
real or fake. It just runs checks and reports findings. `true_label` is only
ever used by the *test harness* for bookkeeping/eval, never by the checks
themselves.

Live event stream (per "Table 3 - What the UI needs LIVE from every agent"):
    {"agent": "semantic", "status": "running|completed", "type":
    "thinking|tool|evidence|output", "text": ..., "tool_name": ...,
    "execution_ms": ..., "confidence": ..., "ts": ...}
Semantic-specific UI behavior: conflicts stream in one at a time and the UI
adds a hotspot marker per conflict as it arrives - see `investigate_stream()`.

Swap the CONFIG block below once the planner's real dispatch contract and
the Ollama/vLLM endpoint are confirmed with the team.
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image

# CLIP - use open_clip or transformers CLIP, whichever the team standardized on.
# Using transformers here since it's the lower-friction default.
import torch
from transformers import CLIPModel, CLIPProcessor

import requests


# ---------------------------------------------------------------------------
# CONFIG - adjust to match actual infra (Muthu's SLM backbone serving setup)
# ---------------------------------------------------------------------------
# Per the team's deployment diagram ("TruthLens AI"), the Semantic Agent's
# reasoning backbone is Qwen 7B-VL - a VISION-language model, not a text-only
# one. That means it can look at the image directly, not just read our
# findings as text. explain() below sends both the image and the structured
# findings, so the model can visually corroborate (or push back on) what the
# deterministic checks found, rather than blindly narrating numbers.
#
# Swap OLLAMA_MODEL to whatever tag your team actually pulled/served, e.g.
# "qwen2-vl:7b" or "qwen2.5-vl:7b" - check `ollama list` on the serving box.
# If the backbone is served via vLLM instead of Ollama, swap ENDPOINT/payload
# shape accordingly once that's confirmed with the team.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:7b" # Qwen 7B-VL per the deployment diagram - confirm exact tag with team
AGENT_SUPPORTS_VISION = True  # set False to fall back to text-only findings if VL isn't served yet

# Candidate scene/object labels for zero-shot plausibility check.
# Extend this with domain-relevant labels for your test set (indoor/outdoor,
# lighting conditions, object categories likely to clash in a manipulated image).
SCENE_LABELS = [
    "an indoor scene", "an outdoor scene", "a studio portrait",
    "a natural daylight photo", "an artificially lit photo",
    "a photo with inconsistent shadows", "a photo with consistent lighting",
]

# Fixed conflict-type enum handed down by the team (Table 1 contract).
# Do not add new types here without confirming with whoever owns the
# Report Agent's schema.
CONFLICT_TYPES = {"lighting", "reflection", "object_relation", "scene_coherence", "lip_sync"}

# Tunable thresholds - arbitrary starting points, tune against labeled
# real/fake examples once you have enough of both classes.
CLIP_AMBIGUOUS_THRESHOLD = 0.35     # below this, scene classification is unreliable
LIGHTING_CV_THRESHOLD = 20.0        # coefficient-of-variation %, not raw variance -
                                     # scale/exposure-invariant, fixes false positives
                                     # on bright/dark real photos that raw variance had
SEVERE_CONFLICT_THRESHOLD = 0.6     # conflict severity above which verdict -> inconsistent
CONFIDENCE_LOW_THRESHOLD = 0.5      # below this, flag for human review


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class SemanticConflict:
    type: str                 # one of CONFLICT_TYPES
    description: str
    region_bbox: List[int]    # [x, y, w, h] in pixel coordinates
    severity: float           # 0-1


@dataclass
class SemanticFindings:
    verdict: str                          # "consistent" | "inconsistent" | "uncertain"
    confidence: float                     # 0-1, confidence in the verdict
    semantic_conflicts: List[SemanticConflict]
    clip_consistency_score: float         # = CLIP's top-label confidence
    annotated_overlay_image: Optional[str]  # path, or None if no conflicts to draw
    uncertainty: Dict[str, float]         # {"epistemic": ..., "aleatoric": ...}
    human_review_required: bool
    limitations: List[str]                # explicit gaps / low-confidence caveats
    debug: Dict[str, Any] = field(default_factory=dict)  # raw signals, for engineers only


class SemanticContextAgent:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    # -- low-level checks ----------------------------------------------------

    def _clip_scene_check(self, image: Image.Image) -> tuple[str, float]:
        inputs = self.clip_processor(
            text=SCENE_LABELS, images=image, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_idx = int(probs.argmax())
        return SCENE_LABELS[best_idx], float(probs[best_idx])

    def _ocr_check(self, rgb_image: np.ndarray) -> tuple[str, bool, Optional[List[int]]]:
        text = pytesseract.image_to_string(rgb_image).strip()
        if not text:
            return text, False, None

        # Placeholder heuristic: flag if OCR finds text but it's garbled
        # (mostly non-alphanumeric) - a common artifact of text baked into
        # generated images. Replace with a real dictionary/language check.
        alnum_ratio = sum(c.isalnum() or c.isspace() for c in text) / max(len(text), 1)
        suspicious = alnum_ratio < 0.6

        bbox = None
        if suspicious:
            try:
                data = pytesseract.image_to_data(rgb_image, output_type=Output.DICT)
                xs, ys, xe, ye = [], [], [], []
                for i, word in enumerate(data.get("text", [])):
                    if word.strip():
                        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        xs.append(x); ys.append(y); xe.append(x + w); ye.append(y + h)
                if xs:
                    bbox = [int(min(xs)), int(min(ys)), int(max(xe) - min(xs)), int(max(ye) - min(ys))]
            except Exception:
                bbox = None  # bbox is best-effort; missing bbox doesn't block the finding

        return text, suspicious, bbox

    def _lighting_shadow_check(self, gray: np.ndarray) -> tuple[float, float, bool, List[int]]:
        h, w = gray.shape
        quad_boxes = [
            (0, 0, w // 2, h // 2), (w // 2, 0, w - w // 2, h // 2),
            (0, h // 2, w // 2, h - h // 2), (w // 2, h // 2, w - w // 2, h - h // 2),
        ]
        quads = [gray[y:y + bh, x:x + bw] for (x, y, bw, bh) in quad_boxes]
        means = [float(q.mean()) for q in quads]

        raw_variance = float(np.var(means))
        mean_of_means = float(np.mean(means)) or 1.0  # avoid div-by-zero on solid-black images
        # Coefficient of variation - scale/exposure invariant, unlike raw variance,
        # which was flagging real bright/dark outdoor photos as "inconsistent"
        # purely because they're high-exposure, not because of an actual lighting
        # mismatch. This is still a coarse proxy, not a true illumination model.
        coefficient_of_variation = float(np.std(means) / mean_of_means * 100)

        shadow_flag = coefficient_of_variation > LIGHTING_CV_THRESHOLD
        worst_quad_idx = int(np.argmax(np.abs(np.array(means) - mean_of_means)))
        worst_bbox = list(quad_boxes[worst_quad_idx])

        return raw_variance, coefficient_of_variation, shadow_flag, worst_bbox

    def _estimate_aleatoric_uncertainty(self, gray: np.ndarray) -> float:
        """
        Crude blur/noise proxy: low Laplacian variance = low sharpness = the
        image itself is a worse source of signal (compressed, blurry, noisy),
        independent of what the model thinks. Higher aleatoric = trust the
        pixels less, regardless of how confident CLIP is.
        """
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        aleatoric = 1.0 - min(1.0, laplacian_var / 1000.0)
        return round(max(0.0, aleatoric), 4)

    # -- evidence-based visual reasoning (brownie point) ---------------------

    def _draw_annotated_overlay(
        self, cv_image: np.ndarray, conflicts: List[SemanticConflict], image_path: str
    ) -> Optional[str]:
        if not conflicts:
            return None
        overlay = cv_image.copy()
        for c in conflicts:
            x, y, w, h = c.region_bbox
            color = (0, 0, 255) if c.severity >= 0.7 else (0, 165, 255) if c.severity >= 0.4 else (0, 200, 200)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            label_y = max(y - 8, 14)
            cv2.putText(overlay, c.type, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        base, _ = os.path.splitext(image_path)
        out_path = f"{base}_semantic_annotated.png"
        cv2.imwrite(out_path, overlay)
        return out_path

    # -- critic / reviewer step (brownie point) ------------------------------

    def _critic_review(self, findings: SemanticFindings, explanation: str) -> Dict[str, Any]:
        """
        A second pass where the SLM checks its own (or a peer's) explanation
        against the raw findings only - catches hallucinated detail or a
        tone/severity mismatch before this goes into the final report.
        """
        critic_system = (
            "You are a critic agent reviewing a forensic explanation for a "
            "deepfake investigation pipeline. You are given structured "
            "findings and an explanation written from them. Check ONLY "
            "whether the explanation is fully grounded in the findings "
            "(no invented detail, no unsupported claims) and whether its "
            "tone matches the severity of the findings. Respond with STRICT "
            "JSON only, no prose outside the JSON: "
            '{"agrees": true|false, "notes": "<1-2 sentence critique>"}'
        )
        critic_user = (
            f"Findings:\n{json.dumps(asdict(findings), indent=2)}\n\n"
            f"Explanation to review:\n{explanation}"
        )
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"{critic_system}\n\n{critic_user}",
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        try:
            parsed = json.loads(raw)
            return {"agrees": bool(parsed.get("agrees", True)), "notes": str(parsed.get("notes", ""))}
        except Exception:
            # Critic output wasn't valid JSON - fail open (agrees=True) rather
            # than silently tanking confidence on a parsing fluke, but keep the
            # raw text so a human can see what happened.
            return {"agrees": True, "notes": f"Critic response was not valid JSON: {raw[:200]}"}

    # -- Step 1: cheap deterministic/vision checks --------------------------

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

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        clip_label, clip_conf = self._clip_scene_check(pil_image)
        ocr_text, ocr_suspicious, ocr_bbox = self._ocr_check(rgb_image)
        raw_variance, cv_score, shadow_flag, lighting_bbox = self._lighting_shadow_check(gray)
        aleatoric = self._estimate_aleatoric_uncertainty(gray)

        conflicts: List[SemanticConflict] = []
        limitations: List[str] = [
            "Reflection/specular-consistency check not implemented in this "
            "version - stubbed for future work.",
            "Object-relation consistency check not implemented - would "
            "require detection + relational reasoning (e.g. YOLO + rules).",
            "Lip-sync check is not applicable to single static images; "
            "only relevant for video input.",
            "The shared conflict-type contract has no dedicated 'text/OCR' "
            "category, so OCR-implausibility findings are filed under "
            "'scene_coherence' below.",
            "Lighting/shadow check is a coefficient-of-variation heuristic, "
            "not a true illumination-consistency model - tune "
            "LIGHTING_CV_THRESHOLD against labeled real/fake examples.",
        ]

        if shadow_flag:
            severity = round(min(1.0, cv_score / 50.0), 4)
            conflicts.append(SemanticConflict(
                type="lighting",
                description=(
                    f"Quadrant brightness coefficient of variation ({cv_score:.1f}%) "
                    f"exceeds the {LIGHTING_CV_THRESHOLD:.0f}% threshold, consistent "
                    f"with a lighting/shadow-direction mismatch."
                ),
                region_bbox=lighting_bbox,
                severity=severity,
            ))

        if ocr_suspicious:
            severity = round(1.0 - (sum(c.isalnum() or c.isspace() for c in ocr_text) / max(len(ocr_text), 1)), 4)
            conflicts.append(SemanticConflict(
                type="scene_coherence",
                description=(
                    "OCR detected text in the image that is mostly non-alphanumeric "
                    "or garbled, a pattern often seen in text baked into generated "
                    "images by diffusion models."
                ),
                region_bbox=ocr_bbox if ocr_bbox else [0, 0, w, h],
                severity=severity,
            ))

        if clip_conf < CLIP_AMBIGUOUS_THRESHOLD:
            conflicts.append(SemanticConflict(
                type="scene_coherence",
                description=(
                    f"CLIP zero-shot scene classification is low-confidence "
                    f"(top label '{clip_label}' at {clip_conf:.2f}), meaning the "
                    f"overall scene semantics are ambiguous or internally "
                    f"inconsistent rather than clearly matching one plausible scene."
                ),
                region_bbox=[0, 0, w, h],
                severity=round(1.0 - clip_conf, 4),
            ))

        # -- roll up verdict / confidence / uncertainty ----------------------
        if not conflicts:
            verdict = "consistent"
            confidence = round(clip_conf, 4)
        else:
            max_severity = max(c.severity for c in conflicts)
            verdict = "inconsistent" if max_severity >= SEVERE_CONFLICT_THRESHOLD else "uncertain"
            confidence = round(max(0.05, min(0.99, (clip_conf + (1.0 - max_severity)) / 2)), 4)

        epistemic = round(1.0 - clip_conf, 4)
        human_review_required = (
            verdict == "uncertain"
            or confidence < CONFIDENCE_LOW_THRESHOLD
            or any(c.severity >= SEVERE_CONFLICT_THRESHOLD for c in conflicts)
        )

        overlay_path = self._draw_annotated_overlay(cv_image, conflicts, image_path)

        return SemanticFindings(
            verdict=verdict,
            confidence=confidence,
            semantic_conflicts=conflicts,
            clip_consistency_score=round(clip_conf, 4),
            annotated_overlay_image=overlay_path,
            uncertainty={"epistemic": epistemic, "aleatoric": aleatoric},
            human_review_required=human_review_required,
            limitations=limitations,
            debug={
                "clip_top_label": clip_label,
                "ocr_text_detected": ocr_text,
                "lighting_raw_variance": round(raw_variance, 2),
                "lighting_coefficient_of_variation": round(cv_score, 2),
            },
        )

    # -- Step 2: hand findings (+ image, if VL backbone) to SLM for explanation

    def explain(self, findings: SemanticFindings, image_path: Optional[str] = None) -> str:
        """
        Turns structured findings into a human-readable forensic finding.

        If AGENT_SUPPORTS_VISION is True and image_path is given, the image
        itself is sent alongside the findings (Ollama's vision models accept
        base64 images via the "images" field) - this lets Qwen 7B-VL actually
        look at the picture and corroborate/contradict the deterministic
        checks, instead of just narrating numbers it's told about.

        Falls back to text-only (findings only, no image) if vision isn't
        available yet or image_path isn't provided - keeps this working even
        before the VL backbone is confirmed serving.
        """
        system_prompt = (
            "You are the Semantic & Context forensic agent in a deepfake "
            "investigation pipeline. You are given structured findings from "
            "vision checks (CLIP scene classification, OCR, lighting/shadow "
            "variance) and a list of detected conflicts"
            + (
                ", along with the image itself. Look at the image and confirm "
                "or push back on what the findings claim before writing your "
                "answer - note explicitly if something you see contradicts a "
                "finding."
                if (AGENT_SUPPORTS_VISION and image_path)
                else "."
            )
            + " Write a concise, specific forensic finding (2-4 sentences) "
            "describing what the evidence suggests about scene consistency, "
            "text plausibility, and lighting. Do not invent detail not "
            "present in the findings or image. Use forensic terminology "
            "(e.g. 'consistent with', 'anomalous', 'implausible')."
        )
        user_prompt = f"Findings:\n{json.dumps(asdict(findings), indent=2)}"

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
        }

        if AGENT_SUPPORTS_VISION and image_path:
            try:
                import base64
                with open(image_path, "rb") as f:
                    payload["images"] = [base64.b64encode(f.read()).decode("utf-8")]
            except Exception:
                pass  # fall back silently to text-only if the image can't be read/encoded

        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=90)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    # -- Entry points ---------------------------------------------------------

    def _event(self, status: str, etype: str, text: str = None, tool_name: str = None,
               execution_ms: int = None, confidence: float = None,
               extra: Dict[str, Any] = None) -> Dict[str, Any]:
        e = {
            "agent": "semantic",
            "status": status,
            "type": etype,          # thinking | tool | evidence | output
            "text": text,
            "tool_name": tool_name,
            "execution_ms": execution_ms,
            "confidence": confidence,
            "ts": time.time(),
        }
        if extra:
            e.update(extra)
        return e

    def investigate_stream(self, image_path: str, true_label: Optional[str] = None,
                            skip_explain: bool = False):
        """
        Generator version for live UI streaming (Table 3 contract). Yields
        one event per step; conflicts are yielded one at a time as they're
        found so the UI can drop a hotspot marker in real time. The final
        event has type="output", status="completed", and carries the full
        Report-Agent-shaped result under event["result"].
        """
        yield self._event("running", "thinking",
                           text=f"Starting semantic & context analysis for {os.path.basename(image_path)}")

        t0 = time.time()
        findings = self.run_checks(image_path)
        yield self._event("running", "tool", text="Ran CLIP scene check, OCR, and lighting analysis",
                           tool_name="clip_ocr_lighting", execution_ms=int((time.time() - t0) * 1000),
                           confidence=findings.clip_consistency_score)

        for conflict in findings.semantic_conflicts:
            yield self._event("running", "evidence",
                               text=f"Conflict detected ({conflict.type}): {conflict.description}",
                               confidence=1.0 - conflict.severity,
                               extra={"conflict": asdict(conflict)})

        result = {
            "agent": "semantic",
            "image": image_path,
            "findings": asdict(findings),
        }

        if not skip_explain:
            try:
                explanation = self.explain(findings, image_path=image_path)
                result["explanation"] = explanation
                yield self._event("running", "thinking", text="Drafted explanation, running critic review")
                critic = self._critic_review(findings, explanation)
                result["critic_review"] = critic
                if not critic.get("agrees", True):
                    adjusted = round(max(0.05, findings.confidence - 0.15), 4)
                    result["findings"]["confidence"] = adjusted
                    result["findings"]["human_review_required"] = True
                    result["findings"]["limitations"].append(
                        f"Critic flagged the explanation: {critic.get('notes', '')}"
                    )
            except Exception as e:
                result["explanation"] = None
                result["explain_error"] = str(e)

        if true_label is not None:
            result["true_label"] = true_label

        yield self._event("completed", "output", text="Semantic analysis complete",
                           confidence=result["findings"]["confidence"], extra={"result": result})

    def investigate(self, image_path: str, true_label: Optional[str] = None,
                     skip_explain: bool = False, on_event=None) -> dict:
        """
        Non-streaming convenience wrapper matching the planner's dispatch
        contract. Pass on_event(callback) if you want the live events too
        (e.g. to forward over SSE/WebSocket) while still getting a single
        final dict back.
        """
        result = None
        for event in self.investigate_stream(image_path, true_label=true_label, skip_explain=skip_explain):
            if on_event:
                on_event(event)
            if event["type"] == "output" and event["status"] == "completed":
                result = event["result"]
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
