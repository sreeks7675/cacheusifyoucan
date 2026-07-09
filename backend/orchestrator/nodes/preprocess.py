from __future__ import annotations

from pathlib import Path

from backend.orchestrator.state import PipelineState


def preprocess_node(state: PipelineState) -> dict:
    print(f"\n--- [ORCHESTRATOR] Preprocess Stage for Session: {state.session_id} ---")

    image = Path(state.image_path)
    metadata = state.image_metadata.copy()
    metadata["image_path"] = str(image)
    metadata["exists"] = image.exists()
    metadata["suffix"] = image.suffix.lower()

    if image.exists():
        try:
            stat = image.stat()
            metadata["file_size_bytes"] = stat.st_size
        except OSError:
            metadata["file_size_bytes"] = None

    logs = [
        f"[Preprocess] path={image} exists={metadata['exists']} suffix={metadata['suffix'] or 'n/a'}",
    ]

    return {
        "image_metadata": metadata,
        "fft_analysis_done": bool(metadata["exists"]),
        "debate_logs": logs,
    }