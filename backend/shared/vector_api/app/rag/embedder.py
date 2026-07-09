"""
Embedding layer for the Shared Knowledge & Memory (RAG) module.

This is the ONE file to swap when you want real semantic embeddings
(sentence-transformers, OpenAI, BGE, E5, etc.) instead of the offline
default. Everything downstream (vector_store, retriever, routes) only
talks to `get_embedding_function()` and never cares which model backs it.

Default: a deterministic, fully-offline `HashingVectorizer`-based
embedder. It needs no internet access and no model download, which
makes the whole knowledge base runnable immediately in any environment
(including CI or a locked-down sandbox). Its semantic quality is lower
than a real neural embedding model - fine for a hackathon demo, but
swap it out (see below) once you have internet access / want stronger
retrieval quality.
"""

from typing import List

from sklearn.feature_extraction.text import HashingVectorizer

from app.config import EMBEDDING_DIM


class OfflineHashingEmbeddingFunction:
    """
    Chroma-compatible embedding function: callable, takes a list of
    strings, returns a list of equal-length float vectors.

    Deterministic and stateless (no training/fit step), so it can embed
    documents incrementally without needing to see the whole corpus
    first - important since documents get ingested one at a time via
    the API.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self._dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm="l2",
        )

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        vectors = self._vectorizer.transform(input)
        return vectors.toarray().tolist()

    def name(self) -> str:
        return "offline-hashing-embedder-v1"


# ---------------------------------------------------------------------------
# To upgrade to real semantic embeddings later, install sentence-transformers
# and swap the factory below, e.g.:
#
#   from chromadb.utils import embedding_functions
#   def get_embedding_function():
#       return embedding_functions.SentenceTransformerEmbeddingFunction(
#           model_name="BAAI/bge-small-en-v1.5"
#       )
#
# Nothing else in the RAG module needs to change - vector_store.py and
# retriever.py only call get_embedding_function().
# ---------------------------------------------------------------------------
def get_embedding_function() -> OfflineHashingEmbeddingFunction:
    return OfflineHashingEmbeddingFunction()
