"""Tests for the HTTP API."""

from fastapi.testclient import TestClient

from supply_chain_agent.api import app
from supply_chain_agent.llm import LLMProviderError


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_returns_llm_answer(monkeypatch) -> None:
    def fake_generate_answer(user_message: str) -> str:
        assert user_message == "What is our current inventory?"
        return "Inventory data is unavailable."

    monkeypatch.setattr(
        "supply_chain_agent.api.generate_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/chat",
        json={"message": "What is our current inventory?"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Inventory data is unavailable."}


def test_chat_rejects_empty_message() -> None:
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_reports_provider_failure(monkeypatch) -> None:
    def fake_provider_failure(user_message: str) -> str:
        raise LLMProviderError("Simulated provider failure")

    monkeypatch.setattr(
        "supply_chain_agent.api.generate_answer",
        fake_provider_failure,
    )

    response = client.post("/chat", json={"message": "Test question"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The language model provider is temporarily unavailable."
    }

