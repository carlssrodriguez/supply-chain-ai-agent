"""Tests for the HTTP API."""

from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient

from supply_chain_agent.approval import ApprovalStore
from supply_chain_agent.api import app
from supply_chain_agent.data_repository import SupplyChainRepository
from supply_chain_agent.llm import LLMProviderError
from supply_chain_agent.rag import RAGAnswer, SourceReference
from supply_chain_agent.replenishment import assess_replenishment


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


async def test_health_check(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


async def test_chat_returns_llm_answer(monkeypatch, client: AsyncClient) -> None:
    def fake_generate_answer(user_message: str) -> str:
        assert user_message == "What is our current inventory?"
        return "Inventory data is unavailable."

    monkeypatch.setattr(
        "supply_chain_agent.api.generate_answer",
        fake_generate_answer,
    )

    response = await client.post(
        "/chat",
        json={"message": "What is our current inventory?"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Inventory data is unavailable."}


async def test_chat_rejects_empty_message(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": ""})

    assert response.status_code == 422


async def test_chat_reports_provider_failure(monkeypatch, client: AsyncClient) -> None:
    def fake_provider_failure(user_message: str) -> str:
        raise LLMProviderError("Simulated provider failure")

    monkeypatch.setattr(
        "supply_chain_agent.api.generate_answer",
        fake_provider_failure,
    )

    response = await client.post("/chat", json={"message": "Test question"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The language model provider is temporarily unavailable."
    }


async def test_agent_chat_returns_tool_grounded_answer(
    monkeypatch,
    client: AsyncClient,
) -> None:
    def fake_answer_with_tools(user_message: str) -> str:
        assert user_message == "What is available stock for SKU-102?"
        return "SKU-102 has 100 available units."

    monkeypatch.setattr(
        "supply_chain_agent.api.answer_with_tools",
        fake_answer_with_tools,
    )

    response = await client.post(
        "/agent/chat",
        json={"message": "What is available stock for SKU-102?"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "SKU-102 has 100 available units."}


async def test_rag_chat_returns_answer_and_retrieved_sources(
    monkeypatch,
    client: AsyncClient,
) -> None:
    def fake_answer_with_rag(question: str, store) -> RAGAnswer:
        assert question == "Who approves EUR 20,000?"
        return RAGAnswer(
            answer="The Director of Supply Chain.",
            retrieved_sources=[
                SourceReference(document="inventory_policy.pdf", page=2)
            ],
        )

    monkeypatch.setattr("supply_chain_agent.api.get_vector_store", lambda: object())
    monkeypatch.setattr(
        "supply_chain_agent.api.answer_with_rag",
        fake_answer_with_rag,
    )

    response = await client.post(
        "/rag/chat",
        json={"message": "Who approves EUR 20,000?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The Director of Supply Chain.",
        "retrieved_sources": [
            {"document": "inventory_policy.pdf", "page": 2}
        ],
    }


async def test_recommendation_api_records_authorized_decision(
    monkeypatch,
    tmp_path,
    client: AsyncClient,
) -> None:
    store = ApprovalStore(tmp_path / "api-approvals.db")
    assessment = assess_replenishment(
        "SKU-104",
        SupplyChainRepository(),
        as_of_date=date(2026, 2, 10),
    )
    monkeypatch.setattr(
        "supply_chain_agent.api.assess_replenishment",
        lambda product_id, repository: assessment,
    )
    monkeypatch.setattr("supply_chain_agent.api.get_approval_store", lambda: store)

    create_response = await client.post(
        "/recommendations",
        json={"product_id": "SKU-104"},
    )
    assert create_response.status_code == 201
    recommendation_id = create_response.json()["recommendation_id"]

    decision_response = await client.post(
        f"/recommendations/{recommendation_id}/decision",
        json={
            "decision": "approved",
            "approver_name": "Alex Morgan",
            "approver_role": "Procurement Manager",
            "rationale": "Current inventory was manually verified.",
        },
    )

    assert decision_response.status_code == 200
    assert decision_response.json()["decision"]["decision"] == "approved"


async def test_recommendation_api_rejects_insufficient_role(
    monkeypatch,
    tmp_path,
    client: AsyncClient,
) -> None:
    store = ApprovalStore(tmp_path / "api-approvals.db")
    assessment = assess_replenishment(
        "SKU-104",
        SupplyChainRepository(),
        as_of_date=date(2026, 2, 10),
    )
    assessment.required_approver = "Director of Supply Chain"
    record = store.create(assessment)
    monkeypatch.setattr("supply_chain_agent.api.get_approval_store", lambda: store)

    response = await client.post(
        f"/recommendations/{record.recommendation_id}/decision",
        json={
            "decision": "approved",
            "approver_name": "Alex Morgan",
            "approver_role": "Procurement Manager",
            "rationale": "Approved.",
        },
    )

    assert response.status_code == 403
    assert "Director of Supply Chain" in response.json()["detail"]


async def test_recommendation_api_prevents_second_decision(
    monkeypatch,
    tmp_path,
    client: AsyncClient,
) -> None:
    store = ApprovalStore(tmp_path / "api-approvals.db")
    assessment = assess_replenishment(
        "SKU-104",
        SupplyChainRepository(),
        as_of_date=date(2026, 2, 10),
    )
    record = store.create(assessment)
    monkeypatch.setattr("supply_chain_agent.api.get_approval_store", lambda: store)
    payload = {
        "decision": "rejected",
        "approver_name": "Alex Morgan",
        "approver_role": "Procurement Manager",
        "rationale": "Refresh the inventory snapshot.",
    }

    assert (await client.post(
        f"/recommendations/{record.recommendation_id}/decision",
        json=payload,
    )).status_code == 200
    second_response = await client.post(
        f"/recommendations/{record.recommendation_id}/decision",
        json=payload,
    )

    assert second_response.status_code == 409
