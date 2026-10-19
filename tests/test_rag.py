"""Tests for retrieval-augmented prompt construction."""

from supply_chain_agent.documents import DocumentChunk
from supply_chain_agent.embeddings import SearchResult
from supply_chain_agent.rag import answer_with_rag


class FakeVectorStore:
    def search(self, query: str, *, top_k: int):
        assert query == "Who approves EUR 20,000?"
        assert top_k == 2
        return [
            SearchResult(
                chunk=DocumentChunk(
                    text="EUR 10,000.01 to EUR 50,000: Director of Supply Chain",
                    source="inventory_policy.pdf",
                    page=2,
                    chunk_index=1,
                ),
                score=0.75,
            )
        ]


def test_answer_with_rag_sends_context_and_returns_source(monkeypatch) -> None:
    def fake_generate_answer(user_message: str, *, system_prompt: str) -> str:
        assert "EUR 10,000.01 to EUR 50,000" in user_message
        assert "Who approves EUR 20,000?" in user_message
        assert "using only the supplied context" in system_prompt
        return "The Director of Supply Chain must approve it."

    monkeypatch.setattr(
        "supply_chain_agent.rag.generate_answer",
        fake_generate_answer,
    )

    result = answer_with_rag(
        "Who approves EUR 20,000?",
        FakeVectorStore(),
        top_k=2,
    )

    assert result.answer == "The Director of Supply Chain must approve it."
    assert result.retrieved_sources[0].document == "inventory_policy.pdf"
    assert result.retrieved_sources[0].page == 2
