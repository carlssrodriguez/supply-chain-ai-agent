"""Tests for hybrid semantic and lexical retrieval."""

from supply_chain_agent.documents import DocumentChunk
from supply_chain_agent.embeddings import SearchResult
from supply_chain_agent.hybrid_search import HybridRetriever, tokenize


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks = (
            DocumentChunk("general warehouse planning", "policy.pdf", 1, 0),
            DocumentChunk("inventory position equals stock", "policy.pdf", 1, 1),
            DocumentChunk("supplier approval", "policy.pdf", 2, 0),
        )

    def search(self, query: str, *, top_k: int):
        dense_order = [self.chunks[0], self.chunks[2], self.chunks[1]]
        return [
            SearchResult(chunk=chunk, score=1.0 / rank)
            for rank, chunk in enumerate(dense_order[:top_k], start=1)
        ]


def test_tokenize_normalizes_case_and_punctuation() -> None:
    assert tokenize("Inventory POSITION, equals stock.") == [
        "inventory",
        "position",
        "equals",
        "stock",
    ]


def test_hybrid_search_promotes_exact_lexical_match() -> None:
    retriever = HybridRetriever(FakeVectorStore())

    results = retriever.search("inventory position", top_k=1)

    assert results[0].chunk.text == "inventory position equals stock"
