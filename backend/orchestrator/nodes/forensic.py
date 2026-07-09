from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

from backend.orchestrator.state import AgentAnalysis, PipelineState


FORENSIC_DIR = Path(__file__).resolve().parents[2] / "agents" / "forensic"
_PREDICT_FN: Callable[[str], dict[str, Any]] | None = None
_PREDICT_LOAD_ERROR: Exception | None = None


def _clamp_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score != score:
        return 0.0
    return min(max(score, 0.0), 1.0)


def _load_predictor() -> Callable[[str], dict[str, Any]]:
    global _PREDICT_FN, _PREDICT_LOAD_ERROR

    if _PREDICT_FN is not None:
        return _PREDICT_FN

    if _PREDICT_LOAD_ERROR is not None:
        raise _PREDICT_LOAD_ERROR

    added_path = False
    previous_cwd = os.getcwd()
    try:
        forensic_path = str(FORENSIC_DIR)
        if forensic_path not in sys.path:
            sys.path.insert(0, forensic_path)
            added_path = True

        # The forensic package uses relative checkpoint paths; import from its directory.
        os.chdir(forensic_path)
        predict_module = importlib.import_module("predict")
        _PREDICT_FN = getattr(predict_module, "predict")
        return _PREDICT_FN
    except Exception as exc:  # noqa: BLE001
        _PREDICT_LOAD_ERROR = exc
        raise
    finally:
        os.chdir(previous_cwd)
        if added_path:
            try:
                sys.path.remove(str(FORENSIC_DIR))
            except ValueError:
                pass


def _forensic_fake_score(prediction: str, confidence_01: float) -> float:
    normalized = str(prediction).strip().upper()
    if normalized == "FAKE":
        return confidence_01
    if normalized == "REAL":
        return 1.0 - confidence_01
    return 0.5


def forensic_node(state: PipelineState) -> dict:
    print(f"\n--- [ORCHESTRATOR] Running Forensic Agent for Session: {state.session_id} ---")

    image_path = str(Path(state.image_path).resolve())
    if not Path(image_path).exists():
        error = f"Image path not found: {image_path}"
        analysis = AgentAnalysis(
            agent_name="Forensic Agent",
            confidence_score=0.0,
            findings=[error],
            raw_output={"error": error},
        )
        updated_responses = state.agent_responses.copy()
        updated_responses["forensic"] = analysis
        return {"agent_responses": updated_responses}

    try:
        predict_fn = _load_predictor()

        # Runtime call also needs forensic cwd because the agent writes output/heatmap.jpg.
        previous_cwd = os.getcwd()
        os.chdir(str(FORENSIC_DIR))
        try:
            result = predict_fn(image_path)
        finally:
            os.chdir(previous_cwd)

        confidence_pct = result.get("confidence", 0.0)
        confidence_01 = _clamp_confidence(float(confidence_pct) / 100.0)
        prediction = str(result.get("prediction", "UNCERTAIN")).upper()
        selected_expert = result.get("selected_expert", "unknown")
        artifact_score = _forensic_fake_score(prediction, confidence_01)

        analysis = AgentAnalysis(
            agent_name="Forensic Agent",
            confidence_score=confidence_01,
            findings=[
                f"Prediction: {prediction}",
                f"Selected expert: {selected_expert}",
                f"Model confidence: {confidence_pct}",
            ],
            raw_output={
                **result,
                "verdict": prediction.lower(),
                "artifact_score": artifact_score,
            },
        )

        updated_responses = state.agent_responses.copy()
        updated_responses["forensic"] = analysis

        updated_evidence = state.consolidated_evidence.copy()
        updated_evidence.update(
            {
                "artifact_score": artifact_score,
                "forensic_prediction": prediction,
                "forensic_confidence": confidence_01,
                "forensic_selected_expert": selected_expert,
                "forensic_heatmap": result.get("heatmap"),
            }
        )

        return {
            "agent_responses": updated_responses,
            "consolidated_evidence": updated_evidence,
            "debate_logs": [
                f"[Forensic] {prediction} ({confidence_01:.3f}) via {selected_expert}",
            ],
        }

    except Exception as exc:  # noqa: BLE001
        error = f"Forensic execution failed: {exc}"
        print(f"[Forensic Node Error] {error}")
        analysis = AgentAnalysis(
            agent_name="Forensic Agent",
            confidence_score=0.0,
            findings=[error],
            raw_output={"error": str(exc)},
        )
        updated_responses = state.agent_responses.copy()
        updated_responses["forensic"] = analysis
        return {"agent_responses": updated_responses}