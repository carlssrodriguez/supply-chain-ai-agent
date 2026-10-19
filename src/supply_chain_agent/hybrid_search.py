"""Hybrid retrieval combining dense vectors with lexical BM25 ranking."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from supply_chain_agent.documents import DocumentChunk
from supply_chain_agent.embeddings import SearchResult
from supply_chain_agent.vector_store import FaissVectorStore


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
RRF_CONSTANT = 60


def tokenize(text: str) -> list[str]:
    """Apply the same simple lexical preprocessing to documents and queries."""
    return TOKEN_PATTERN.findall(text.casefold())


def _chunk_key(chunk: DocumentChunk) -> tuple[str, int, int]:
    return chunk.source, chunk.page, chunk.chunk_index


class HybridRetriever:
    """Fuse FAISS semantic ranking and BM25 lexical ranking with RRF."""

    def __init__(self, vector_store: FaissVectorStore) -> None:
        self._vector_store = vector_store
        self._chunks = list(vector_store.chunks)
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self._chunks])

    def search(self, query: str, *, top_k: int = 3) -> list[SearchResult]:
        """Return chunks ranked by reciprocal-rank fusion."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        dense_results = self._vector_store.search(query, top_k=len(self._chunks))
        fused_scores: dict[tuple[str, int, int], float] = {}

        for rank, result in enumerate(dense_results, start=1):
            key = _chunk_key(result.chunk)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1 / (
                RRF_CONSTANT + rank
            )

        lexical_scores = self._bm25.get_scores(tokenize(query))
        lexical_indices = np.argsort(lexical_scores)[::-1]
        lexical_rank = 0
        for index in lexical_indices:
            if lexical_scores[index] <= 0:
                continue
            lexical_rank += 1
            key = _chunk_key(self._chunks[int(index)])
            fused_scores[key] = fused_scores.get(key, 0.0) + 1 / (
                RRF_CONSTANT + lexical_rank
            )

        chunks_by_key = {_chunk_key(chunk): chunk for chunk in self._chunks}
        ranked_keys = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                chunk=chunks_by_key[key],
                score=fused_scores[key],
            )
            for key in ranked_keys
        ]
