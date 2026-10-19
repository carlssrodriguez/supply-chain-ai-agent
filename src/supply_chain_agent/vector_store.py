"""Persistent FAISS vector index linked to document chunk metadata."""

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from supply_chain_agent.documents import DocumentChunk
from supply_chain_agent.embeddings import MODEL_NAME, LocalEmbeddingModel, SearchResult


INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
MANIFEST_FILENAME = "manifest.json"


def build_vector_store(
    chunks: list[DocumentChunk],
    model: LocalEmbeddingModel,
    output_dir: Path,
) -> None:
    """Embed chunks and persist a cosine-similarity FAISS index plus metadata."""
    if not chunks:
        raise ValueError("Cannot build a vector store without chunks")

    vectors = model.embed_passages([chunk.text for chunk in chunks])
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)

    dimensions = vectors.shape[1]
    index = faiss.IndexFlatIP(dimensions)
    index.add(vectors)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / INDEX_FILENAME))

    (output_dir / CHUNKS_FILENAME).write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2),
        encoding="utf-8",
    )
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "embedding_model": MODEL_NAME,
                "dimensions": dimensions,
                "vector_count": len(chunks),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class FaissVectorStore:
    """Load a persisted FAISS index and retrieve its associated chunks."""

    def __init__(self, store_dir: Path, model: LocalEmbeddingModel) -> None:
        self._model = model
        self._index = faiss.read_index(str(store_dir / INDEX_FILENAME))

        raw_chunks = json.loads(
            (store_dir / CHUNKS_FILENAME).read_text(encoding="utf-8")
        )
        self._chunks = [DocumentChunk(**chunk) for chunk in raw_chunks]

        if self._index.ntotal != len(self._chunks):
            raise ValueError("FAISS vectors and chunk metadata are out of sync")

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        """Return indexed chunks in the same order as their FAISS vectors."""
        return tuple(self._chunks)

    def search(self, query: str, *, top_k: int = 3) -> list[SearchResult]:
        """Find the top-k chunks for one query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self._model.embed_query(query).reshape(1, -1)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_vector)

        result_count = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query_vector, result_count)

        return [
            SearchResult(
                chunk=self._chunks[int(index)],
                score=float(score),
            )
            for score, index in zip(scores[0], indices[0], strict=True)
        ]
