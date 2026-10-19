"""Tests for persistent vector storage and retrieval."""

from pathlib import Path

import numpy as np

from supply_chain_agent.documents import DocumentChunk
from supply_chain_agent.vector_store import FaissVectorStore, build_vector_store


class FakeEmbeddingModel:
    """Small deterministic model used only by tests."""

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def test_vector_store_persists_and_retrieves_metadata(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk("closest", "policy.pdf", 1, 0),
        DocumentChunk("unrelated", "policy.pdf", 2, 0),
        DocumentChunk("opposite", "policy.pdf", 3, 0),
    ]
    model = FakeEmbeddingModel()

    build_vector_store(chunks, model, tmp_path)
    store = FaissVectorStore(tmp_path, model)
    results = store.search("test query", top_k=2)

    assert results[0].chunk.text == "closest"
    assert results[0].chunk.page == 1
    assert results[0].score == 1.0
    assert results[1].chunk.text == "unrelated"
