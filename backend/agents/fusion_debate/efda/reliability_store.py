"""
Reliability store — the "Past Cases & Feedback" memory from the shared
knowledge layer, scoped to what Agent 5 needs: a trust score per upstream
agent, optionally conditioned on case type (e.g. "face_swap", "meme_edit").

Backed by a simple JSON file so it survives across runs. Swap this for a
real DB (Postgres, Redis, a vector store) in production without changing
the interface used by the rest of the package.
"""

import json
import os
from pathlib import Path

DEFAULT_PRIOR = 0.6
LEARNING_RATE = 0.08  # how fast trust shifts after one piece of feedback


class ReliabilityStore:
    def __init__(self, path: str = "reliability_store.json"):
        self.path = Path(path)
        self._data: dict[str, dict[str, float]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def get(self, agent: str, case_type: str = "default") -> float:
        return self._data.get(agent, {}).get(case_type, DEFAULT_PRIOR)

    def update(self, agent: str, case_type: str, was_correct: bool) -> float:
        """Nudge an agent's trust score after a case is resolved
        (by a human reviewer, or by later corroborating evidence)."""
        current = self.get(agent, case_type)
        target = 1.0 if was_correct else 0.0
        updated = current + LEARNING_RATE * (target - current)
        updated = max(0.05, min(0.98, updated))  # keep it bounded, never 0 or 1
        self._data.setdefault(agent, {})[case_type] = round(updated, 4)
        self._persist()
        return updated

    def _persist(self):
        self.path.write_text(json.dumps(self._data, indent=2))

    def snapshot(self) -> dict:
        return json.loads(json.dumps(self._data))  # deep copy
