"""Local embedding generation and transparent semantic search."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from supply_chain_agent.documents import DocumentChunk


MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "fastembed"


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its semantic similarity score."""

    chunk: DocumentChunk
    score: float


class LocalEmbeddingModel:
    """Generate embeddings locally with a small CPU-friendly model."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(cache_dir),
        )

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Convert document passages into one matrix of vectors."""
        return np.asarray(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Convert one search query into a vector."""
        return np.asarray(next(self._model.query_embed(query)), dtype=np.float32)


def cosine_scores(query_vector: np.ndarray, passage_vectors: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between one query and every passage."""
    query_norm = np.linalg.norm(query_vector)
    passage_norms = np.linalg.norm(passage_vectors, axis=1)

    if query_norm == 0 or np.any(passage_norms == 0):
        raise ValueError("Cosine similarity cannot use a zero-length vector")

    dot_products = passage_vectors @ query_vector
    return dot_products / (passage_norms * query_norm)


def semantic_search(
    query: str,
    chunks: list[DocumentChunk],
    model: LocalEmbeddingModel,
    *,
    top_k: int = 3,
) -> list[SearchResult]:
    """Return the chunks whose embeddings are closest to the query embedding."""
    if not chunks:
        return []
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    passage_vectors = model.embed_passages([chunk.text for chunk in chunks])
    query_vector = model.embed_query(query)
    scores = cosine_scores(query_vector, passage_vectors)
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return [
        SearchResult(chunk=chunks[index], score=float(scores[index]))
        for index in ranked_indices
    ]
