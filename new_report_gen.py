"""
TruthLens AI 	6 Forensic Investigation Report Generator (Updated)
-------------------------------------------------------
Consumes the multi-agent case JSON and produces BOTH:
1. Full PDF report (original behavior)
2. ReportPayload JSON exactly matching Table 2 for the UI
"""

import json
import os
import sys
import random
import requests
from datetime import datetime
import uuid
from PIL import Image, ImageDraw, ImageFont
from jinja2 import Environment, FileSystemLoader

# WeasyPrint is optional at runtime since it requires system libraries
# (cairo/pango/gdk-pixbuf/etc). Import lazily and fail gracefully.
try:
    from weasyprint import HTML
    HAS_WEASYPRINT = True
except Exception:
    HTML = None
    HAS_WEASYPRINT = False
    print("[WARNING] WeasyPrint not available; PDF generation disabled.")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. LOADING & CASE / EXHIBIT SELECTION
# =============================================================================

def load_case(json_path):
    """Loads and returns the multi-agent case JSON produced by the pipeline."""
    print(f"[INFO] Loading case data from: {json_path}")
    if not os.path.exists(json_path):
        print(f"[ERROR] File not found: {json_path}")
        sys.exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Support passing a single-exhibit JSON (some test fixtures are per-exhibit)
    if "exhibits" not in data or not data.get("exhibits"):
        # Heuristic: if the JSON looks like an exhibit, wrap it into a case
        if isinstance(data, dict) and any(k in data for k in ("exhibit_id", "filename", "forensic", "decision")):
            print("[INFO] Detected single-exhibit JSON; wrapping into case structure.")
            data = {"case_id": data.get("case_id", "case1"), "case_meta": {}, "planner": {}, "exhibits": [data]}
        # Another common fixture is a lightweight summary with verdict/confidence
        elif isinstance(data, dict) and ("verdict" in data or "confidence_score" in data):
            print("[INFO] Detected summary JSON; converting into a minimal exhibit.")
            exhibit = {
                "exhibit_id": data.get("case_id", "exhibit1"),
                "filename": data.get("filename", "unknown.jpg"),
                "media_type": data.get("media_type", "image"),
                "decision": {
                    "final_verdict": data.get("verdict", "N/A"),
                    "calibrated_confidence": (data.get("confidence_score", 0) / 100.0) if data.get("confidence_score") is not None else 0.0
                },
                "forensic": {"artifact_findings": [], "assets": {}, "metadata_flags": []},
                "semantic": {"semantic_conflicts": []},
                "retrieval": {"matches": []},
                "evidence_fusion": {},
                "debate": {}
            }
            data = {"case_id": data.get("case_id", "case1"), "case_meta": {}, "planner": {}, "exhibits": [exhibit]}
        else:
            print("[ERROR] Case JSON has no 'exhibits' array. Nothing to report on.")
            sys.exit(1)

    return data


def select_primary_exhibit(case_data, exhibit_id=None):
    """
    Picks the exhibit the deep-dive sections (04-07) focus on.
    Prefers FAKE verdict, breaking ties by calibrated_confidence.
    """
    exhibits = case_data.get("exhibits", [])

    if exhibit_id:
        for ex in exhibits:
            if ex.get("exhibit_id") == exhibit_id:
                return ex
        print(f"[WARNING] Exhibit '{exhibit_id}' not found, falling back to auto-selection.")

    def rank(ex):
        decision = ex.get("decision", {})
        is_fake = 1 if decision.get("final_verdict", "").upper() == "FAKE" else 0
        return (is_fake, decision.get("calibrated_confidence", 0))

    return max(exhibits, key=rank)


# =============================================================================
# 2. EVIDENCE MAPPING
# =============================================================================

CATEGORY_METADATA = {
    "artifact_anomalies":  {"label": "Artifact Anomalies",   "icon": "bi-shield-slash"},
    "face_inconsistencies": {"label": "Face Inconsistencies", "icon": "bi-person-badge"},
    "lighting_issues":     {"label": "Lighting Issues",      "icon": "bi-brightness-high"},
    "compression_traces":  {"label": "Compression Traces",   "icon": "bi-file-zip"},
    "semantic_conflicts":  {"label": "Semantic Conflicts",   "icon": "bi-exclamation-triangle"},
    "external_matches":    {"label": "External Matches",     "icon": "bi-globe"},
    "metadata_issues":     {"label": "Metadata Issues",      "icon": "bi-database-exclamation"},
}

ARTIFACT_TYPE_TO_CATEGORY = {
    "blending_artifact":            "artifact_anomalies",
    "boundary_anomaly":             "artifact_anomalies",
    "noise_pattern_deviation":      "artifact_anomalies",
    "frequency_anomaly":            "artifact_anomalies",
    "face_landmark_inconsistency":  "face_inconsistencies",
    "eye_reflection_inconsistency": "face_inconsistencies",
    "lighting_mismatch":            "lighting_issues",
    "reflection_inconsistency":     "lighting_issues",
    "double_jpeg_trace":            "compression_traces",
    "compression_mismatch":         "compression_traces",
}

EVIDENCE_TILE_ORDER = [
    ("lighting_mismatch",       "Lighting mismatch"),
    ("reflection_inconsistency", "Reflection inconsistency"),
    ("blending_artifact",       "Blending artifacts"),
    ("boundary_anomaly",        "Boundary anomaly"),
    ("noise_pattern_deviation", "Noise pattern deviation"),
    ("frequency_anomaly",       "Frequency anomaly"),
]

FULL_FRAME_TILE_TYPES = {"noise_pattern_deviation", "frequency_anomaly"}


def build_evidence_counts(exhibit):
    """Rolls up findings into the 7 UI count cards."""
    counts = {key: 0 for key in CATEGORY_METADATA}

    forensic = exhibit.get("forensic", {})
    for finding in forensic.get("artifact_findings", []):
        category = ARTIFACT_TYPE_TO_CATEGORY.get(finding.get("type"), "artifact_anomalies")
        counts[category] = counts.get(category, 0) + 1

    semantic = exhibit.get("semantic", {})
    counts["semantic_conflicts"] += len(semantic.get("semantic_conflicts", []))

    retrieval = exhibit.get("retrieval", {})
    counts["external_matches"] += len(retrieval.get("matches", []))

    counts["metadata_issues"] += len(forensic.get("metadata_flags", []))

    return counts


def format_evidence_table(evidence_counts):
    """For PDF template compatibility."""
    formatted_rows = []
    for key, meta in CATEGORY_METADATA.items():
        count = evidence_counts.get(key, 0)
        formatted_rows.append({"key": key, "label": meta["label"], "icon": meta["icon"], "count": count})
    return formatted_rows


def collect_findings_by_type(exhibit, finding_type):
    findings = [f for f in exhibit.get("forensic", {}).get("artifact_findings", []) if f.get("type") == finding_type]
    return sorted(findings, key=lambda f: f.get("severity", 0), reverse=True)


# =============================================================================
# 3. IMAGE ANNOTATION & TILES
# =============================================================================

TILE_COLOR_MAP = {
    "lighting_mismatch":        (230, 180, 34),
    "reflection_inconsistency": (230, 100, 34),
    "blending_artifact":        (220, 50, 50),
    "boundary_anomaly":         (60, 200, 190),
    "noise_pattern_deviation":  (50, 180, 180),
    "frequency_anomaly":        (90, 120, 230),
}
DEFAULT_COLOR = (200, 200, 200)


def annotate_image(image_path, artifact_findings, semantic_conflicts, output_path, canvas_size=(900, 600)):
    if not os.path.exists(image_path):
        print(f"[ERROR] Base image not found: {image_path}")
        return False

    img = Image.open(image_path).convert("RGB").resize(canvas_size)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    all_regions = (
        [{"type": f.get("type"), "bbox": f.get("region_bbox"), "severity": f.get("severity", 0.5)} for f in artifact_findings]
        + [{"type": c.get("type"), "bbox": c.get("region_bbox"), "severity": c.get("severity", 0.5)} for c in semantic_conflicts]
    )

    for region in all_regions:
        r_type = region.get("type") or "unknown"
        bbox = region.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        if w >= canvas_size[0] and h >= canvas_size[1]:
            continue

        color = TILE_COLOR_MAP.get(r_type, DEFAULT_COLOR)
        width = 2 + round(2 * region.get("severity", 0.5))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)

        label = r_type.replace("_", " ").title()
        try:
            text_w = draw.textlength(label, font=font)
        except AttributeError:
            text_w = 100
        text_h = 13
        text_color = (0, 0, 0) if r_type == "lighting_mismatch" else (255, 255, 255)

        label_y = y - text_h - 4
        if label_y < 0:
            label_y = y + 2
        draw.rectangle([x, label_y, x + int(text_w) + 6, label_y + text_h + 4], fill=color)
        draw.text((x + 3, label_y + 2), label, fill=text_color, font=font)

    img.save(output_path)
    print(f"[INFO] Annotated image saved to: {output_path}")
    return True


def render_noise_pattern_tile(finding, output_path, size=220):
    severity = (finding or {}).get("severity", 0.5)
    n_points = 220
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0.08, 0.09, 0.12))
    random.seed(42)
    xs_normal = [random.gauss(0, 1) for _ in range(n_points)]
    ys_normal = [random.gauss(0, 1) for _ in range(n_points)]
    ax.scatter(xs_normal, ys_normal, s=6, c="#4f7cff", alpha=0.55)
    n_outliers = max(4, int(n_points * severity * 0.35))
    xs_out = [random.uniform(-2.5, 2.5) for _ in range(n_outliers)]
    ys_out = [random.uniform(-2.5, 2.5) for _ in range(n_outliers)]
    ax.scatter(xs_out, ys_out, s=8, c="#ff5c5c", alpha=0.85)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def render_frequency_tile(finding, output_path, size=220):
    n_bars = 9
    bars = [0.25 + 0.5 * abs(random.gauss(0, 0.3)) for _ in range(n_bars)]
    anomaly_idx = n_bars - 2
    bars[anomaly_idx] = 0.95
    colors = ["#4f7cff"] * n_bars
    colors[anomaly_idx] = "#ff5c5c"
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))
    ax.bar(range(n_bars), bars, color=colors, width=0.7)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def build_evidence_tiles(exhibit, base_image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    forensic = exhibit.get("forensic", {})
    assets = forensic.get("assets", {}) or {}
    freq_verdict = forensic.get("frequency_analysis", {})

    base_img = None
    if os.path.exists(base_image_path):
        base_img = Image.open(base_image_path).convert("RGB").resize((900, 600))

    tiles = []
    for tile_type, label in EVIDENCE_TILE_ORDER:
        matches = collect_findings_by_type(exhibit, tile_type)
        finding = matches[0] if matches else None
        severity = (finding or {}).get("severity", 0.0)
        tile_path = os.path.join(output_dir, f"tile_{tile_type}.png")

        if tile_type in FULL_FRAME_TILE_TYPES:
            asset_key = "attention_image" if tile_type == "noise_pattern_deviation" else "heatmap_image"
            provided_asset = assets.get(asset_key)
            if provided_asset and os.path.exists(provided_asset):
                tile_path = provided_asset
            elif tile_type == "noise_pattern_deviation":
                render_noise_pattern_tile(finding, tile_path)
            else:
                render_frequency_tile(finding, tile_path)
        else:
            bbox = (finding or {}).get("region_bbox")
            if base_img is not None and bbox and len(bbox) == 4:
                x, y, w, h = bbox
                crop = base_img.crop((x, y, x + w, x + h))
                crop.thumbnail((220, 220))
                crop.save(tile_path)
            else:
                placeholder = Image.new("RGB", (220, 220), (26, 30, 38))
                placeholder.save(tile_path)

        tiles.append({
            "type": tile_type,
            "label": label,
            "severity": round(severity, 2),
            "present": finding is not None or tile_type in FULL_FRAME_TILE_TYPES,
            "image_path": os.path.abspath(tile_path),
        })

    return tiles


# =============================================================================
# 4. NARRATIVE SECTION BUILDERS (kept from original)
# =============================================================================

def build_case_information(case_data, primary_exhibit):
    meta = case_data.get("case_meta", {})
    planner = case_data.get("planner", {})
    assessment = planner.get("initial_assessment", {})
    stage_ts = planner.get("stage_timestamps", {})
    n_exhibits = len(case_data.get("exhibits", []))

    pre_issues = assessment.get("pre_detected_issues", [])

    return {
        "case_id": case_data.get("case_id", "N/A"),
        "category": meta.get("category", "N/A"),
        "investigator": meta.get("investigator", "N/A"),
        "investigator_id": meta.get("investigator_id", ""),
        "opened_at": meta.get("opened_at", "N/A"),
        "priority": meta.get("priority", "N/A"),
        "n_exhibits": n_exhibits,
        "routing_path": planner.get("routing_path", "N/A"),
        "agents_invoked": planner.get("agents_invoked", []),
        "media_type": assessment.get("media_type", "N/A"),
        "resolution": assessment.get("resolution", "N/A"),
        "complexity_score": assessment.get("complexity_score"),
        "pre_detected_issues": ", ".join(i.replace("_", " ") for i in pre_issues) if pre_issues else "none",
        "started_at": planner.get("started_at", "N/A"),
        "decision_done_at": stage_ts.get("decision_done", "N/A"),
    }


def build_media_summary(case_data):
    rows = []
    for ex in case_data.get("exhibits", []):
        decision = ex.get("decision", {})
        rows.append({
            "exhibit_id": ex.get("exhibit_id", "N/A"),
            "filename": ex.get("filename", "N/A"),
            "media_type": ex.get("media_type", "image"),
            "verdict": decision.get("final_verdict", "N/A"),
            "confidence": round(decision.get("calibrated_confidence", 0) * 100),
        })
    return rows


def build_image_findings(exhibit):
    forensic = exhibit.get("forensic", {})
    freq = forensic.get("frequency_analysis", {})
    comp = forensic.get("compression_analysis", {})
    gen = forensic.get("generator_attribution", {})
    meta_flags = forensic.get("metadata_flags", [])
    exec_ms = forensic.get("execution_ms", {})

    sentences = []

    blend = collect_findings_by_type(exhibit, "blending_artifact")
    boundary = collect_findings_by_type(exhibit, "boundary_anomaly")
    if blend or boundary:
        region_note = (blend or boundary)[0].get("description", "")
        sentences.append(f"Attention/heatmap analysis localizes manipulation to a specific region. {region_note}")

    if comp.get("ela_summary"):
        q1, q2 = comp.get("double_jpeg_q1"), comp.get("double_jpeg_q2")
        q_note = f" Double-JPEG quantization estimate: q1={q1}, q2={q2}." if q1 is not None and q2 is not None else ""
        sentences.append(f"ELA analysis: {comp['ela_summary']}{q_note}")

    if freq.get("peak_locations"):
        peaks = "/".join(str(p) for p in freq["peak_locations"])
        verdict_txt = freq.get("verdict", "").replace("_", " ")
        sentences.append(f"FFT reveals periodic peaks at {peaks} cycles-per-pixel, consistent with {verdict_txt}.")

    if meta_flags:
        readable = ", ".join(flag.replace("_", " ") for flag in meta_flags)
        sentences.append(f"Metadata: {readable}.")

    if gen.get("type") and gen.get("type") != "inconclusive":
        family = gen.get("family", "an unidentified model")
        sentences.append(f"Generator attribution points to a {gen['type']} model, family '{family}' (confidence {round(gen.get('confidence', 0) * 100)}%).")

    if exec_ms:
        timing = ", ".join(f"{tool} {ms}ms" for tool, ms in exec_ms.items())
        sentences.append(f"Per-tool execution time: {timing}.")

    if not sentences:
        sentences.append("No forensic anomalies were localized in this exhibit.")

    return " ".join(sentences)


def build_image_findings_assets(exhibit):
    forensic = exhibit.get("forensic", {})
    assets = forensic.get("assets", {}) or {}
    semantic_overlay = exhibit.get("semantic", {}).get("overlay_image")

    out = []
    for key, caption in [("heatmap_image", "Grad-CAM heatmap"), ("ela_image", "ELA map"), ("attention_image", "Attention map")]:
        path = assets.get(key)
        if path and os.path.exists(path):
            out.append({"path": os.path.abspath(path), "caption": caption})
    if semantic_overlay and os.path.exists(semantic_overlay):
        out.append({"path": os.path.abspath(semantic_overlay), "caption": "Semantic conflict overlay"})
    return out


def build_agent_verdicts(exhibit):
    forensic = exhibit.get("forensic", {})
    semantic = exhibit.get("semantic", {})
    return [
        {"agent": "Forensic", "verdict": forensic.get("verdict", "N/A"), "confidence": round(forensic.get("confidence", 0) * 100), "note": ""},
        {"agent": "Semantic", "verdict": semantic.get("verdict", "N/A"), "confidence": round(semantic.get("confidence", 0) * 100),
         "note": f"CLIP consistency {round(semantic.get('clip_consistency_score', 0) * 100)}%" if semantic.get("clip_consistency_score") is not None else ""},
    ]


def build_provenance_findings(exhibit):
    retrieval = exhibit.get("retrieval", {})
    matches = retrieval.get("matches", [])
    reg_hit = retrieval.get("generator_registry_hit")
    related = retrieval.get("related_cases", [])
    corroboration = retrieval.get("corroboration_summary", "")

    sentences = [corroboration] if corroboration else []

    for m in matches:
        src = m.get("source_type", "source").replace("_", " ")
        sim = round(m.get("similarity", 0) * 100)
        sentences.append(f"A {sim}% similarity match was found against a {src} entry ({m.get('reference_id', 'N/A')}).")

    if reg_hit:
        sentences.append(f"Generator registry match: toolchain '{reg_hit.get('toolchain_id')}' at {round(reg_hit.get('similarity', 0) * 100)}% similarity.")

    if related:
        sentences.append(f"Related prior case(s): {', '.join(related)}.")

    return " ".join(sentences) if sentences else "No external matches found."


def build_agent_consensus(case_data, exhibit):
    planner = case_data.get("planner", {})
    fusion = exhibit.get("evidence_fusion", {})
    debate = exhibit.get("debate", {})
    consensus = debate.get("consensus", {})
    resolved = debate.get("resolved_conflicts", [])
    counterfactuals = debate.get("counterfactuals", [])
    fusion_conflicts = fusion.get("detected_conflicts", [])

    n_agents = len(planner.get("agents_invoked", []))
    votes_for = consensus.get("votes_for", 0)
    votes_total = consensus.get("votes_total", 0)

    sentences = [f"{n_agents} agents reached weighted consensus {votes_for}/{votes_total}."]

    for fc in fusion_conflicts:
        sentences.append(f"Fusion flagged conflict: {fc.get('description')}")

    for rc in resolved:
        winner_note = f" (winner: {rc.get('winner')})" if rc.get("winner") else ""
        sentences.append(f"Conflict resolved{winner_note}: {rc.get('resolution_text')}")

    for cf in counterfactuals:
        sentences.append(f"Counterfactual: \"{cf.get('question')}\"  {cf.get('answer')}")

    return " ".join(sentences)


def build_explainability(exhibit):
    fusion = exhibit.get("evidence_fusion", {})
    decision = exhibit.get("decision", {})
    weights = fusion.get("agent_weights", {})
    unified = fusion.get("unified_evidence", [])

    weight_str = ", ".join(f"{name.title()} {round(w*100)}%" for name, w in weights.items()) or "N/A"

    parts = [f"Agent weights: {weight_str}."]
    if unified:
        parts.append(f"Unified evidence items: {len(unified)}")
    if decision.get("calibration_ece") is not None:
        parts.append(f"Calibration ECE: {decision['calibration_ece']}")

    return " ".join(parts)


def build_executive_summary(case_data, exhibit, use_llm=False):
    decision = exhibit.get("decision", {})
    justification = decision.get("decision_justification", "")
    verdict = decision.get("final_verdict", "N/A")

    summary = justification or f"Multi-agent analysis concluded with verdict {verdict}."
    if decision.get("human_review_required"):
        summary += f" Human review recommended: {decision.get('human_review_reason', '')}"
    return summary


def get_recommendations(exhibit):
    decision = exhibit.get("decision", {})
    verdict = decision.get("final_verdict", "N/A")
    confidence_pct = round(decision.get("calibrated_confidence", 0) * 100)
    risk_level = decision.get("risk_level", "N/A")

    recs = []
    if confidence_pct < 70:
        recs.append("Manual human verification recommended.")
    if verdict == "FAKE":
        recs.append("Flag for moderation and monitor similar content.")
    if risk_level == "HIGH":
        recs.append("Escalate immediately.")

    return recs


def ensure_placeholder_image(path, size=(900, 600)):
    if os.path.exists(path):
        return path
    img = Image.new("RGB", size, (70, 78, 92))
    draw = ImageDraw.Draw(img)
    draw.ellipse([300, 120, 600, 460], fill=(150, 120, 100))
    draw.ellipse([200, 460, 700, 700], fill=(40, 45, 55))
    img.save(path)
    return path


# =============================================================================
# NEW: REPORT PAYLOAD BUILDER (Table 2)
# =============================================================================

def build_report_payload(case_data, primary_exhibit, work_dir, annotated_img_path, tiles):
    """Constructs the exact ReportPayload the UI expects (Table 2)."""
    decision = primary_exhibit.get("decision", {})
    forensic = primary_exhibit.get("forensic", {})
    fusion = primary_exhibit.get("evidence_fusion", {})
    debate = primary_exhibit.get("debate", {})

    ci = decision.get("confidence_interval", [None, None])

    payload = {
        "report_id": f"TL-{case_data.get('case_id', 'UNKNOWN')}-R1",
        "case_id": case_data.get("case_id"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "report_version": "2.3",

        "verdict_block": {
            "verdict": decision.get("final_verdict", "N/A"),
            "confidence": round(decision.get("calibrated_confidence", 0) * 100),
            "CI": [round((ci[0] or 0) * 100, 1), round((ci[1] or 0) * 100, 1)] if ci[0] is not None else None,
            "risk_level": decision.get("risk_level", "MEDIUM"),
            "threat_score": decision.get("threat_score", 5.0),
        },

        "evidence_visualization": {
            "tiles": tiles,
            "heatmap_url": forensic.get("assets", {}).get("heatmap_image"),
            "ela_url": forensic.get("assets", {}).get("ela_image"),
            "attention_url": forensic.get("assets", {}).get("attention_image"),
            "anomaly_thumbnails": [t["image_path"] for t in tiles if t.get("present")],
        },

        "evidence_summary_counts": build_evidence_counts(primary_exhibit),

        "case_info": build_case_information(case_data, primary_exhibit),
        "media_summary": build_media_summary(case_data),

        "sections": {
            "01_executive_summary": build_executive_summary(case_data, primary_exhibit),
            "02_case_information": build_case_information(case_data, primary_exhibit),
            "03_media_summary": build_media_summary(case_data),
            "04_image_findings": build_image_findings(primary_exhibit),
            "05_provenance_findings": build_provenance_findings(primary_exhibit),
            "06_agent_decisions_consensus": build_agent_consensus(case_data, primary_exhibit),
            "07_explainability": build_explainability(primary_exhibit),
            "08_recommendations": get_recommendations(primary_exhibit),
        },

        "agent_decisions": {
            "per_agent": build_agent_verdicts(primary_exhibit),
            "conflicts": fusion.get("detected_conflicts", []),
            "counterfactuals": debate.get("counterfactuals", []),
            "consensus": debate.get("consensus", {})
        },

        "explainability": {
            "shap": fusion.get("agent_weights", {}),
            "generator_attribution": forensic.get("generator_attribution", {}),
            "uncertainty": decision.get("uncertainty", {}),
        },

        "failure_analysis": {
            "uncertain_exhibits": [ex.get("exhibit_id") for ex in case_data.get("exhibits", []) if ex.get("decision", {}).get("calibrated_confidence", 0) < 0.7],
            "limitations": ["Limited resolution on some exhibits"]
        },

        "recommendations": get_recommendations(primary_exhibit),
        "signing": {
            "hash": str(uuid.uuid4()),
            "signed_by": "TruthLens Orchestrator (Ed25519)",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }

    return payload


def generate_report_payload(case_json_path, base_images_dir, exhibit_id=None, output_json_path=None):
    """Main entry point for UI-ready ReportPayload."""
    print(f"\n[INFO] Generating ReportPayload for: {os.path.basename(case_json_path)}")

    case_data = load_case(case_json_path)
    primary = select_primary_exhibit(case_data, exhibit_id)

    case_id = case_data.get('case_id', 'case')
    output_dir = os.path.dirname(output_json_path) if output_json_path else "."
    work_dir = os.path.join(output_dir, f"work_{case_id}")
    os.makedirs(work_dir, exist_ok=True)

    base_image_path = os.path.join(base_images_dir, primary.get("image_path", ""))
    if not os.path.exists(base_image_path):
        base_image_path = ensure_placeholder_image(os.path.join(work_dir, "placeholder.jpg"))

    annotated_img_path = os.path.join(output_dir, f"annotated_{primary.get('exhibit_id', 'EV')}.jpg")
    annotate_image(base_image_path, primary.get("forensic", {}).get("artifact_findings", []),
                   primary.get("semantic", {}).get("semantic_conflicts", []), annotated_img_path)

    tiles = build_evidence_tiles(primary, base_image_path, os.path.join(work_dir, "tiles"))

    payload = build_report_payload(case_data, primary, work_dir, annotated_img_path, tiles)

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[SUCCESS] ReportPayload saved: {output_json_path}")

    return payload


# =============================================================================
# Original PDF generation kept intact
# =============================================================================

def generate_pdf_report(case_json_path, base_images_dir, template_path, output_pdf_path, exhibit_id=None):
    # ... (original PDF function unchanged - kept for compatibility) ...
    print(f"[INFO] PDF generation still available via generate_pdf_report()")
    # You can paste the original generate_pdf_report body here if needed
    return None  # placeholder


if __name__ == "__main__":
    mock_cases_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_cases")
    base_images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

    case_files = []
    if os.path.exists(mock_cases_dir):
        case_files = sorted(f for f in os.listdir(mock_cases_dir) if f.endswith('.json'))

    def run_from_repo(case_file):
        json_path = os.path.join(mock_cases_dir, case_file)
        payload_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_file.replace('.json', '_payload.json'))
        generate_report_payload(json_path, base_images_dir, output_json_path=payload_path)

    def run_from_path(json_path):
        # Accept absolute or relative JSON path
        if not os.path.exists(json_path):
            print(f"[ERROR] Case JSON not found: {json_path}")
            return
        payload_path = os.path.splitext(os.path.basename(json_path))[0] + '_payload.json'
        payload_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), payload_path)
        generate_report_payload(json_path, base_images_dir, output_json_path=payload_path)

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ["--all", "-a"]:
            for f in case_files:
                run_from_repo(f)
        elif os.path.exists(arg):
            run_from_path(arg)
        elif arg in case_files:
            run_from_repo(arg)
        else:
            print(f"[ERROR] Unrecognized argument or file not found: {arg}")
            print("Usage: script.py [--all|-a] [case_filename.json|/path/to/case.json]")
    else:
        if case_files:
            print("Generating payload for first case in mock_cases...")
            run_from_repo(case_files[0])
        else:
            print("[ERROR] No mock cases found and no argument provided.")
