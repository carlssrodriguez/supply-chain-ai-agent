"""Tests for retrieval evaluation logic."""

from supply_chain_agent.documents import DocumentChunk
from supply_chain_agent.embeddings import SearchResult
from scripts.evaluate_retrieval import evaluate_question


class FakeStore:
    def search(self, query: str, *, top_k: int):
        return [
            SearchResult(
                DocumentChunk("irrelevant", "policy.pdf", 1, 0),
                score=0.9,
            ),
            SearchResult(
                DocumentChunk("the expected evidence", "policy.pdf", 2, 0),
                score=0.8,
            ),
        ]


def test_evaluate_question_reports_first_evidence_rank() -> None:
    item = {
        "id": "example",
        "question": "Example question?",
        "evidence_contains": "expected evidence",
    }

    result = evaluate_question(item, FakeStore())

    assert result.question_id == "example"
    assert result.rank == 2
    assert result.top_score == 0.9


def test_evaluate_question_accepts_alternative_evidence() -> None:
    item = {
        "id": "example",
        "question": "Example question?",
        "evidence_contains_any": ["missing phrase", "expected evidence"],
    }

    result = evaluate_question(item, FakeStore())

    assert result.rank == 2
