"""
Semantic & Context Agent (Agent 3)
-----------------------------------
Role (per design_document.md / EADIS slide):
  - Object & scene consistency
  - CLIP / VLM semantic check          -> now VLM-only (Qwen 7B-VL)
  - Text in image (OCR)                -> now read directly by the VLM
  - Context vs. real-world plausibility -> now judged directly by the VLM
  - Lighting & shadow consistency       -> now judged directly by the VLM

CHANGE FROM PREVIOUS VERSION: this agent used to run three *separate*
deterministic/local checks (CLIP zero-shot scene classification,
pytesseract OCR, a cv2 quadrant-brightness heuristic) and only handed the
*results* of those checks to the SLM for narration. Per request, this
version runs EVERYTHING through the vision-language backbone (Qwen 7B-VL,
served via Ollama) instead - a single structured "analyze" call replaces
CLIP + OCR + the cv2 lighting heuristic. cv2/PIL are now only used for
plain image I/O (reading the file, getting dimensions, drawing the overlay)
- never for semantic judgment.

Output contract (per "Table 1 - What the Report Agent needs FROM each agent",
Semantic row) is UNCHANGED:
    verdict, confidence, semantic_conflicts[] {type, description, region_bbox,
    severity}, clip_consistency_score, annotated overlay image (optional)

NOTE on `clip_consistency_score`: the field name is kept as-is because it's
part of the contractual schema handed down for Table 1 - it is no longer
literally a CLIP score, it's now the VLM's own self-reported scene-
consistency confidence. Renaming it would break the Report Agent's parser,
so it stays `clip_consistency_score` with a comment wherever it's set.

conflict "type" is constrained to the fixed enum handed down by the team:
    lighting | reflection | object_relation | scene_coherence | lip_sync

IMPORTANT / HONEST LIMITATION: that enum has no dedicated "text/OCR" bucket.
OCR-implausibility findings are therefore filed under "scene_coherence" -
see `limitations` in the output for this and other known gaps.
Reflection and object_relation ARE now attempted (the VLM is asked
explicitly to look for them) but are still weaker/unvalidated signals since
there's no dedicated detector behind them anymore - just the model's own
judgment. lip_sync is still N/A for single static images.

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
from PIL import Image

import requests


# ---------------------------------------------------------------------------
# CONFIG - adjust to match actual infra (Muthu's SLM backbone serving setup)
# ---------------------------------------------------------------------------
# Everything in this agent now runs through the vision-language backbone -
# Qwen 7B-VL per the deployment diagram ("TruthLens AI") - not just the
# narration step. That means there is no local/offline fallback anymore:
# if the VLM endpoint is down, run_checks() itself fails (handled gracefully,
# see below), not just explain().
#
# Swap OLLAMA_MODEL to whatever tag your team actually pulled/served, e.g.
# "qwen2-vl:7b" or "qwen2.5-vl:7b" - check `ollama list` on the serving box.
# If the backbone is served via vLLM instead of Ollama, swap ENDPOINT/payload
# shape accordingly once that's confirmed with the team.
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:7b"  # Qwen 7B-VL per the deployment diagram - confirm exact tag with team

# Kept for interface compatibility with the planner/report agent - now
# effectively always True, since there is no non-vision path left. Leave it
# here (rather than deleting it) in case a future text-only fallback model
# needs to be wired back in.
AGENT_SUPPORTS_VISION = True

# Fixed conflict-type enum handed down by the team (Table 1 contract).
# Do not add new types here without confirming with whoever owns the
# Report Agent's schema.
CONFLICT_TYPES = {"lighting", "reflection", "object_relation", "scene_coherence", "lip_sync"}

# Tunable thresholds - arbitrary starting points, tune against labeled
# real/fake examples once you have enough of both classes.
SCENE_AMBIGUOUS_THRESHOLD = 0.35    # below this, the VLM's own scene-consistency confidence is unreliable
SEVERE_CONFLICT_THRESHOLD = 0.6     # conflict severity above which verdict -> inconsistent
CONFIDENCE_LOW_THRESHOLD = 0.5      # below this, flag for human review

# The VLM's self-reported "scene consistency confidence" over a single
# forced judgment call is a noisier signal than a genuine detected anomaly
# (VLM-flagged lighting mismatch, VLM-read garbled/impossible text) - plenty
# of ordinary REAL photos can still get a middling confidence just because
# the model hedges, not because anything is actually wrong. So this conflict
# type is capped below SEVERE_CONFLICT_THRESHOLD: low scene-consistency
# confidence alone can push the verdict to "uncertain" but never to
# "inconsistent" on its own - it can only tip things into "inconsistent"
# territory when COMBINED with another real detected conflict that already
# carries higher severity.
SCENE_CONFLICT_MAX_SEVERITY = 0.55

# Whether to derive the "aleatoric" (image-quality-limited) uncertainty
# component from a cheap local blur/noise proxy (cv2 Laplacian variance) or
# ask the VLM to self-report it. Kept as a local deterministic proxy by
# default: it's a *non-semantic* signal ("how much can any observer trust
# these pixels"), it's nearly free (no extra model call), and VLMs are
# generally unreliable at self-reporting calibrated uncertainty about their
# own input quality. Flip to True if you'd rather have Qwen estimate it too
# (folded into the same JSON response as the rest of the analysis).
ALEATORIC_FROM_MODEL = False

# Where to write annotated overlays. None => write next to the source image
# (default). If that write fails (e.g. a read-only mounted dataset folder),
# we fall back to ./semantic_overlays/ automatically - see
# _draw_annotated_overlay(). Set this explicitly to force all overlays into
# one folder regardless.
OVERLAY_OUTPUT_DIR: Optional[str] = None


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
    clip_consistency_score: float         # legacy field name (Table 1 contract) - now Qwen's self-reported scene-consistency confidence
    annotated_overlay_image: Optional[str]  # path, or None if no conflicts to draw
    uncertainty: Dict[str, float]         # {"epistemic": ..., "aleatoric": ...}
    human_review_required: bool
    limitations: List[str]                # explicit gaps / low-confidence caveats
    debug: Dict[str, Any] = field(default_factory=dict)  # raw signals, for engineers only


class SemanticContextAgent:
    def __init__(self):
        # No local model to load anymore - CLIP/pytesseract are gone, every
        # check goes over the wire to the Ollama-served Qwen 7B-VL backbone.
        pass

    # -- low-level helpers (plain I/O only, no semantic judgment) -----------

    @staticmethod
    def _load_image(image_path: str):
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
            Image.open(image_path).convert("RGB")  # just validate PIL can open it too
        except Exception as e:
            raise ValueError(f"PIL could not open '{image_path}': {e}")

        return cv_image

    def _estimate_aleatoric_uncertainty(self, cv_image) -> float:
        """
        Crude blur/noise proxy: low Laplacian variance = low sharpness = the
        image itself is a worse source of signal (compressed, blurry, noisy),
        independent of what the model thinks. Higher aleatoric = trust the
        pixels less, regardless of how confident the VLM is. Deliberately
        kept as a local deterministic signal - see ALEATORIC_FROM_MODEL above.
        """
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        aleatoric = 1.0 - min(1.0, laplacian_var / 1000.0)
        return round(max(0.0, aleatoric), 4)

    # -- JSON extraction (shared by the analysis call and the critic call) --

    @staticmethod
    def _extract_json_object(raw: str) -> Dict[str, Any]:
        """
        LLMs asked for "strict JSON" frequently wrap it in ```json ... ```
        fences or add a stray sentence before/after anyway. A bare
        json.loads(raw) fails on all of that. This strips common fencing
        and, failing that, extracts the first {...} span before parsing.
        """
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"No parseable JSON object found in: {raw[:200]!r}")

    @staticmethod
    def _clamp_bbox(bbox: Any, w: int, h: int) -> List[int]:
        try:
            x, y, bw, bh = [int(v) for v in bbox]
        except Exception:
            return [0, 0, w, h]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        return [x, y, bw, bh]

    # -- Step 1: single VLM call replaces CLIP + OCR + cv2 lighting ---------

    def _call_ollama_vision(self, prompt: str, image_path: str, timeout: int = 90) -> str:
        import base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def _vlm_analyze(self, image_path: str, w: int, h: int) -> Dict[str, Any]:
        """
        Sends the image to Qwen 7B-VL with instructions to do everything
        CLIP + OCR + the cv2 lighting heuristic used to do, and to return
        strict JSON we can parse into the same SemanticFindings shape.
        """
        system_prompt = (
            "You are the Semantic & Context forensic agent in a deepfake "
            "investigation pipeline. Examine the attached image directly - "
            "do not assume it is real or fake, just report what you observe. "
            "Check for: (1) scene coherence - do the objects, setting, and "
            "their relationships make real-world sense together; "
            "(2) reflections - do reflective surfaces (mirrors, glass, water, "
            "eyes) match what should be reflected; (3) lighting and shadows - "
            "are shadow directions and lighting consistent across the whole "
            "image; (4) any text visible in the image - read it and judge "
            "whether it is real, legible, plausible text or garbled/impossible "
            "text (a common artifact of AI-generated images). Lip-sync is not "
            "applicable to a single static image, ignore it.\n\n"
            f"The image is {w}x{h} pixels, origin (0,0) at top-left.\n\n"
            "Respond with STRICT JSON only, no prose outside the JSON, in "
            "exactly this shape:\n"
            "{\n"
            '  "scene_label": "<short description of the overall scene>",\n'
            '  "scene_consistency_confidence": <0.0-1.0, your confidence that '
            "the scene is coherent and internally consistent>,\n"
            '  "ocr_text_detected": "<verbatim text you see in the image, or '
            'empty string if none>",\n'
            '  "conflicts": [\n'
            "    {\n"
            '      "type": "<one of: lighting, reflection, object_relation, '
            'scene_coherence>",\n'
            '      "description": "<specific, concrete description of the '
            'anomaly you observed>",\n'
            '      "region_bbox": [x, y, width, height],\n'
            '      "severity": <0.0-1.0>\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Only include an entry in \"conflicts\" if you genuinely observe "
            "something anomalous - do not invent conflicts to fill the list. "
            "If there is legible text but it looks garbled/nonsensical/AI-"
            'baked-in, file it under type "scene_coherence" (there is no '
            "dedicated text/OCR type in this schema)."
        )

        raw = self._call_ollama_vision(system_prompt, image_path)
        parsed = self._extract_json_object(raw)
        return parsed

    def run_checks(self, image_path: str) -> SemanticFindings:
        cv_image = self._load_image(image_path)
        h, w = cv_image.shape[:2]

        limitations: List[str] = [
            "Lip-sync check is not applicable to single static images; "
            "only relevant for video input.",
            "The shared conflict-type contract has no dedicated 'text/OCR' "
            "category, so OCR-implausibility findings are filed under "
            "'scene_coherence' below.",
            "Scene coherence, reflection, object-relation, lighting/shadow, "
            "and OCR-plausibility are all now judged directly by the "
            "vision-language backbone (Qwen 7B-VL) in a single call rather "
            "than by dedicated CLIP/OCR/cv2 detectors - this is a stronger "
            "holistic signal but a less validated/reproducible one than the "
            "previous deterministic heuristics, and region_bbox values are "
            "the model's own visual estimate rather than pixel-measured.",
        ]

        try:
            analysis = self._vlm_analyze(image_path, w, h)
        except Exception as e:
            # No deterministic fallback exists anymore - fail safe rather
            # than fail loud: return low-confidence findings that force
            # human review instead of raising and killing the whole run.
            limitations.append(f"Vision-language backbone call failed: {e}")
            return SemanticFindings(
                verdict="uncertain",
                confidence=0.05,
                semantic_conflicts=[],
                clip_consistency_score=0.0,
                annotated_overlay_image=None,
                uncertainty={"epistemic": 1.0, "aleatoric": self._estimate_aleatoric_uncertainty(cv_image)},
                human_review_required=True,
                limitations=limitations,
                debug={"vlm_error": str(e)},
            )

        scene_label = str(analysis.get("scene_label", ""))
        scene_conf = analysis.get("scene_consistency_confidence", 0.5)
        try:
            scene_conf = float(scene_conf)
        except Exception:
            scene_conf = 0.5
        scene_conf = max(0.0, min(1.0, scene_conf))

        ocr_text = str(analysis.get("ocr_text_detected", "") or "")

        conflicts: List[SemanticConflict] = []
        for raw_conflict in analysis.get("conflicts", []) or []:
            try:
                ctype = str(raw_conflict.get("type", "")).strip()
                if ctype not in CONFLICT_TYPES or ctype == "lip_sync":
                    # lip_sync excluded for static images; anything else
                    # outside the enum is dropped rather than silently
                    # corrupting the Report Agent's schema.
                    if ctype and ctype not in CONFLICT_TYPES:
                        limitations.append(
                            f"VLM returned an out-of-enum conflict type "
                            f"'{ctype}' - dropped rather than passed through."
                        )
                    continue
                severity = float(raw_conflict.get("severity", 0.5))
                severity = max(0.0, min(1.0, severity))
                bbox = self._clamp_bbox(raw_conflict.get("region_bbox", [0, 0, w, h]), w, h)
                description = str(raw_conflict.get("description", "")).strip() or "No description provided."
                conflicts.append(SemanticConflict(
                    type=ctype, description=description, region_bbox=bbox, severity=severity,
                ))
            except Exception:
                continue  # skip malformed individual conflict entries, don't fail the whole run

        # Low scene-consistency confidence is treated the same way the old
        # low-CLIP-confidence branch was: capped severity, can't alone push
        # the verdict to "inconsistent". See SCENE_CONFLICT_MAX_SEVERITY.
        if scene_conf < SCENE_AMBIGUOUS_THRESHOLD:
            conflicts.append(SemanticConflict(
                type="scene_coherence",
                description=(
                    f"VLM scene-consistency confidence is low (scene "
                    f"'{scene_label}' at {scene_conf:.2f}), meaning the "
                    f"overall scene semantics are ambiguous or internally "
                    f"inconsistent rather than clearly plausible. Note: low "
                    f"scene-consistency confidence alone is a weak signal "
                    f"(common on genuine photos too), so its severity is "
                    f"capped below the 'inconsistent' threshold on its own."
                ),
                region_bbox=[0, 0, w, h],
                severity=round(min(SCENE_CONFLICT_MAX_SEVERITY, 1.0 - scene_conf), 4),
            ))

        # -- roll up verdict / confidence / uncertainty ----------------------
        if not conflicts:
            verdict = "consistent"
            confidence = round(scene_conf, 4)
        else:
            max_severity = max(c.severity for c in conflicts)
            verdict = "inconsistent" if max_severity >= SEVERE_CONFLICT_THRESHOLD else "uncertain"
            confidence = round(max(0.05, min(0.99, (scene_conf + (1.0 - max_severity)) / 2)), 4)

        epistemic = round(1.0 - scene_conf, 4)
        aleatoric = self._estimate_aleatoric_uncertainty(cv_image)  # ALEATORIC_FROM_MODEL toggle honored below
        if ALEATORIC_FROM_MODEL:
            model_aleatoric = analysis.get("aleatoric_confidence")
            try:
                if model_aleatoric is not None:
                    aleatoric = round(max(0.0, min(1.0, 1.0 - float(model_aleatoric))), 4)
            except Exception:
                pass

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
            clip_consistency_score=round(scene_conf, 4),  # legacy field name, see class docstring
            annotated_overlay_image=overlay_path,
            uncertainty={"epistemic": epistemic, "aleatoric": aleatoric},
            human_review_required=human_review_required,
            limitations=limitations,
            debug={
                "vlm_scene_label": scene_label,
                "ocr_text_detected": ocr_text,
                "raw_vlm_analysis": analysis,
            },
        )

    # -- overlay drawing (plain I/O, unchanged) -------------------------------

    def _draw_annotated_overlay(
        self, cv_image, conflicts: List[SemanticConflict], image_path: str
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

        filename = os.path.splitext(os.path.basename(image_path))[0] + "_semantic_annotated.png"
        primary_dir = OVERLAY_OUTPUT_DIR or os.path.dirname(image_path) or "."
        out_path = os.path.join(primary_dir, filename)

        # cv2.imwrite does NOT raise on failure - it just returns False - so
        # a read-only dataset dir (common for shared/staging data) would
        # otherwise silently return a path to a file that was never
        # written. Check the return value and fall back to a writable local
        # directory instead of lying about where the overlay actually is.
        try:
            ok = bool(cv2.imwrite(out_path, overlay))
        except Exception:
            ok = False

        if not ok:
            fallback_dir = os.path.join(os.getcwd(), "semantic_overlays")
            os.makedirs(fallback_dir, exist_ok=True)
            out_path = os.path.join(fallback_dir, filename)
            try:
                ok = bool(cv2.imwrite(out_path, overlay))
            except Exception:
                ok = False

        return out_path if ok else None

    # -- Step 2: hand findings (+ image) to the VLM for a narrative explanation

    def explain(self, findings: SemanticFindings, image_path: Optional[str] = None) -> str:
        """
        Turns structured findings into a human-readable forensic finding.
        Still sends the image alongside the findings (Qwen 7B-VL) so the
        model can visually corroborate/contradict its own earlier structured
        judgment rather than just narrating numbers back at itself.
        """
        system_prompt = (
            "You are the Semantic & Context forensic agent in a deepfake "
            "investigation pipeline. You previously analyzed this image and "
            "produced the structured findings below. Look at the image again "
            "and write a concise, specific forensic finding (2-4 sentences) "
            "describing what the evidence suggests about scene consistency, "
            "text plausibility, and lighting. Do not invent detail not "
            "present in the findings or image. Use forensic terminology "
            "(e.g. 'consistent with', 'anomalous', 'implausible').\n\n"
            f"Findings:\n{json.dumps(asdict(findings), indent=2)}"
        )

        if image_path:
            try:
                return self._call_ollama_vision(system_prompt, image_path)
            except Exception:
                pass  # fall through to text-only below if the image call fails

        # Text-only fallback (no image, or image call failed)
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={"model": OLLAMA_MODEL, "prompt": system_prompt, "stream": False},
            timeout=90,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    # -- critic / reviewer step (unchanged, still Qwen 7B-VL via Ollama) ----

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
            parsed = self._extract_json_object(raw)
            return {"agrees": bool(parsed.get("agrees", True)), "notes": str(parsed.get("notes", ""))}
        except Exception:
            # Critic output wasn't valid/extractable JSON - fail open
            # (agrees=True) rather than silently tanking confidence on a
            # parsing fluke, but keep the raw text so a human can see what
            # happened.
            return {"agrees": True, "notes": f"Critic response was not valid JSON: {raw[:200]}"}

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

        NOTE: skip_explain now only skips the narrative explanation + critic
        review steps. It can NOT skip the initial analysis call, because
        that call (Qwen 7B-VL) is now the only check mechanism - there is no
        local/offline fallback left to fall back to.
        """
        yield self._event("running", "thinking",
                           text=f"Starting semantic & context analysis for {os.path.basename(image_path)}")

        t0 = time.time()
        findings = self.run_checks(image_path)
        yield self._event("running", "tool", text="Ran VLM scene/text/lighting analysis (Qwen 7B-VL)",
                           tool_name="qwen_vl_semantic_analysis", execution_ms=int((time.time() - t0) * 1000),
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
            # Separate try/except per step: a critic-review failure (e.g. a
            # second Ollama call timing out) must not discard an explanation
            # that already succeeded.
            try:
                explanation = self.explain(findings, image_path=image_path)
                result["explanation"] = explanation
            except Exception as e:
                result["explanation"] = None
                result["explain_error"] = str(e)

            if result.get("explanation"):
                try:
                    yield self._event("running", "thinking", text="Drafted explanation, running critic review")
                    critic = self._critic_review(findings, result["explanation"])
                    result["critic_review"] = critic
                    if not critic.get("agrees", True):
                        adjusted = round(max(0.05, findings.confidence - 0.15), 4)
                        result["findings"]["confidence"] = adjusted
                        result["findings"]["human_review_required"] = True
                        result["findings"]["limitations"].append(
                            f"Critic flagged the explanation: {critic.get('notes', '')}"
                        )
                except Exception as e:
                    result["critic_review"] = None
                    result["critic_error"] = str(e)

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
