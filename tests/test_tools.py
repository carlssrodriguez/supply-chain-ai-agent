import pytest
from pydantic import ValidationError

from supply_chain_agent.documents import DocumentChunk
from supply_chain_agent.embeddings import SearchResult
from supply_chain_agent.data_repository import SupplyChainRepository
from supply_chain_agent.tools import (
    TOOL_DEFINITIONS,
    UnknownToolError,
    execute_tool,
)


@pytest.fixture
def repository() -> SupplyChainRepository:
    return SupplyChainRepository()


def test_inventory_tool_returns_available_units(
    repository: SupplyChainRepository,
) -> None:
    result = execute_tool(
        "get_inventory",
        {"product_id": "SKU-102"},
        repository,
    )

    assert isinstance(result, dict)
    assert result["on_hand_units"] == 120
    assert result["allocated_units"] == 20
    assert result["available_units"] == 100


def test_supplier_tool_only_returns_approved_options(
    repository: SupplyChainRepository,
) -> None:
    result = execute_tool(
        "get_approved_suppliers",
        {"product_id": "SKU-105"},
        repository,
    )

    assert isinstance(result, list)
    assert [supplier["supplier_id"] for supplier in result] == ["SUP-007"]


def test_tool_arguments_are_validated(repository: SupplyChainRepository) -> None:
    with pytest.raises(ValidationError):
        execute_tool("get_inventory", {}, repository)


def test_unregistered_tool_cannot_run(repository: SupplyChainRepository) -> None:
    with pytest.raises(UnknownToolError, match="delete_everything"):
        execute_tool("delete_everything", {}, repository)


def test_tool_definitions_contain_json_argument_schema() -> None:
    inventory_definition = TOOL_DEFINITIONS[0]["function"]

    assert inventory_definition["name"] == "get_inventory"
    assert inventory_definition["parameters"]["required"] == ["product_id"]


def test_policy_search_returns_retrieved_evidence(
    repository: SupplyChainRepository,
) -> None:
    class FakeVectorStore:
        def search(self, query: str, *, top_k: int):
            assert query == "Who approves EUR 20,000?"
            assert top_k == 3
            return [
                SearchResult(
                    chunk=DocumentChunk(
                        text="EUR 10,000.01 to EUR 50,000: Director of Supply Chain",
                        source="inventory_policy.pdf",
                        page=2,
                        chunk_index=1,
                    ),
                    score=0.81,
                )
            ]

    result = execute_tool(
        "search_policy",
        {"query": "Who approves EUR 20,000?"},
        repository,
        FakeVectorStore(),
    )

    assert result[0]["page"] == 2
    assert "Director of Supply Chain" in result[0]["text"]


def test_replenishment_tool_returns_validated_draft(
    repository: SupplyChainRepository,
) -> None:
    result = execute_tool(
        "assess_replenishment",
        {"product_id": "SKU-104"},
        repository,
    )

    assert isinstance(result, dict)
    assert result["recommended_order_quantity"] == 500
    assert result["required_approver"] == "Procurement Manager"
    assert result["status"] == "pending_human_review"
    assert result["warnings"]
