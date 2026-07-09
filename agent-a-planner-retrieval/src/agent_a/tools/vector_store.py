"""Vector store access for the Retrieval Agent.

Same lazy-import pattern as slm_client.py: RetrievalAgent depends on the
VectorStoreClient interface only, so it's testable with a fake store and never
needs faiss/sentence-transformers installed to be imported or unit tested.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

from agent_a.schemas.task_plan import RetrievalMatch

logger = logging.getLogger(__name__)


class VectorStoreClient(ABC):
    @abstractmethod
    def search(self, query_text: str, top_k: int = 5) -> List[RetrievalMatch]:
        raise NotImplementedError


class FaissVectorStoreClient(VectorStoreClient):
    """Backs the Retrieval Agent's 'known real/fake samples + generator registry'
    store described in pipeline_training_and_orchestration.md, Section 2.
    """

    def __init__(
        self,
        index_path: str,
        metadata: List[dict],
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_path = index_path
        self.embedding_model_name = embedding_model_name
        self._metadata = metadata
        self._index = None
        self._embedder = None

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        import faiss
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedding_model_name)
        self._index = faiss.read_index(self.index_path)
        logger.info("Loaded FAISS index from %s (%d vectors)", self.index_path, self._index.ntotal)

    def search(self, query_text: str, top_k: int = 5) -> List[RetrievalMatch]:
        self._ensure_loaded()
        query_vec = self._embedder.encode([query_text])
        distances, indices = self._index.search(query_vec, top_k)

        results: List[RetrievalMatch] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            similarity = 1.0 / (1.0 + float(dist))
            results.append(
                RetrievalMatch(
                    source=meta.get("source", "unknown"),
                    similarity=min(max(similarity, 0.0), 1.0),
                    label=meta.get("label", "unknown"),
                    metadata=meta,
                )
            )
        return results
