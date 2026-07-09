"""Preprocessing applied to every image before it reaches the Planner agent
(and before it goes into the training set the Planner is distilled from).

WHY THIS EXISTS AND HAS ITS OWN FILE:
For a normal vision task you'd just resize + convert to RGB. For a forensic
deepfake-detection task, resizing/re-encoding is not free — it can destroy the
exact signals the downstream forensic agent is looking for (JPEG re-compression
artifacts, blending-boundary noise, sensor-noise patterns, upsampling grids).
So preprocessing here has two competing goals that have to be balanced explicitly:

  1. Make the image safe/consistent to feed into the VLM processor
     (bounded size, correct orientation, correct color mode, no corrupt files).
  2. Do NOT destroy forensic evidence in the process
     (no forced re-JPEG at low quality, no blurring, no aggressive downscaling
     that erases the very high-frequency artifacts the forensic agent needs).

Both the training pipeline (`prepare_planner_dataset.py`) and the live agent
(`slm_client.RealSLMClient`) should call `preprocess_for_planner()` so the
Planner sees images at train time and inference time in the same distribution.
Skipping this at inference time (train/inference skew) is one of the most common
silent causes of a fine-tuned VLM adapter under-performing its eval numbers.
"""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, bytes, "PIL.Image.Image"]  # noqa: F821

# Qwen2-VL / Phi-3.5-vision both tolerate large inputs, but VRAM scales with
# token count (roughly image_area / patch_size^2). Cap the long edge instead of
# a fixed square resize — squashing to a fixed aspect ratio warps geometry cues
# the semantic-context agent relies on (object proportions, horizon lines).
MAX_LONG_EDGE = 1536
# Below this, forensic signal (blending boundaries, noise residue) gets too
# sparse to be meaningfully visible even to a human reviewer; flag rather than
# silently proceed.
MIN_LONG_EDGE = 256


@dataclass
class PreprocessResult:
    image: "PIL.Image.Image"  # noqa: F821
    original_size: tuple
    final_size: tuple
    was_resized: bool
    sha256: str  # content hash of the *original* bytes — for dataset dedup/audit trail
    warnings: list


def preprocess_for_planner(image: ImageInput) -> PreprocessResult:
    """The one function both training and inference should call.

    Order matters:
      1. Load bytes once (so we can hash the original before any transform).
      2. Fix EXIF orientation BEFORE anything else — otherwise every later
         geometric judgment (horizon consistency, shadow direction) is on a
         rotated image and will be systematically wrong.
      3. Convert to RGB (strips alpha/palette weirdness the processor chokes on;
         do this after orientation fix, not before — EXIF lives in the original
         mode's metadata).
      4. Only resize if the long edge exceeds MAX_LONG_EDGE, and use LANCZOS
         (best detail preservation of PIL's resamplers) rather than default
         BILINEAR — every bit of high-frequency detail matters here.
      5. Never re-encode to JPEG as part of this step. If the caller needs
         bytes back out, save lossless (PNG) so no new compression artifacts
         are introduced on top of whatever's already in the source image —
         real vs. re-compressed-by-us should not become a spurious signal
         the planner learns to key off.
    """
    from PIL import Image, ImageOps

    warnings: list = []
    raw_bytes = _read_bytes(image)
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()  # force decode now so corrupt files fail here, not mid-batch later
    except Exception as exc:
        raise ValueError(f"Could not decode image (corrupt or unsupported format): {exc}") from exc

    # Step 2: orientation fix from EXIF, before any mode conversion.
    img = ImageOps.exif_transpose(img)

    original_size = img.size

    # Step 3: normalize color mode.
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):
            warnings.append(f"converted from {img.mode} to RGB (alpha/palette channel dropped)")
        img = img.convert("RGB")

    # Step 4: bounded resize, aspect-ratio preserved.
    long_edge = max(img.size)
    was_resized = False
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        was_resized = True
        warnings.append(f"downscaled {original_size} -> {new_size} (long edge cap {MAX_LONG_EDGE}px)")
    elif long_edge < MIN_LONG_EDGE:
        warnings.append(
            f"image long edge is only {long_edge}px (below {MIN_LONG_EDGE}px) — "
            "forensic signal may be too sparse for reliable detection; not blocking, but flag downstream"
        )

    return PreprocessResult(
        image=img,
        original_size=original_size,
        final_size=img.size,
        was_resized=was_resized,
        sha256=sha256,
        warnings=warnings,
    )


def _read_bytes(image: ImageInput) -> bytes:
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if isinstance(image, bytes):
        return image
    # Already a PIL.Image (e.g. loaded upstream) — re-encode losslessly to get
    # bytes for hashing without discarding anything.
    from PIL import Image

    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(f"Unsupported image input type: {type(image)}")
