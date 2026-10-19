"""Safe tools that an LLM may request but cannot execute directly."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from supply_chain_agent.data_repository import SupplyChainRepository
from supply_chain_agent.replenishment import assess_replenishment


class UnknownToolError(LookupError):
    """Raised when the LLM requests a tool that is not registered."""


class ProductIdArguments(BaseModel):
    product_id: str = Field(
        min_length=1,
        description="Exact product identifier, for example SKU-104.",
    )


class PolicySearchArguments(BaseModel):
    query: str = Field(
        min_length=1,
        description="Question to search for in the internal policy documents.",
    )


class SearchableVectorStore(Protocol):
    def search(self, query: str, *, top_k: int) -> list[Any]: ...


def get_inventory(
    arguments: ProductIdArguments,
    repository: SupplyChainRepository,
) -> dict[str, Any]:
    """Return exact stock data and the calculated available quantity."""
    item = repository.get_inventory(arguments.product_id)
    result = item.model_dump(mode="json")
    result["available_units"] = item.available_units
    return result


def get_approved_suppliers(
    arguments: ProductIdArguments,
    repository: SupplyChainRepository,
) -> list[dict[str, Any]]:
    """Return only approved supplier options for an exact product ID."""
    return [
        supplier.model_dump(mode="json")
        for supplier in repository.get_approved_suppliers(arguments.product_id)
    ]


def get_open_purchase_orders(
    arguments: ProductIdArguments,
    repository: SupplyChainRepository,
) -> list[dict[str, Any]]:
    """Return confirmed and in-transit purchase orders for a product."""
    return [
        order.model_dump(mode="json")
        for order in repository.get_open_purchase_orders(arguments.product_id)
    ]


def get_replenishment_assessment(
    arguments: ProductIdArguments,
    repository: SupplyChainRepository,
) -> dict[str, Any]:
    """Return a calculated draft that always remains subject to human controls."""
    return assess_replenishment(
        arguments.product_id,
        repository,
    ).model_dump(mode="json")


def search_policy(
    arguments: PolicySearchArguments,
    vector_store: SearchableVectorStore,
) -> list[dict[str, Any]]:
    """Retrieve policy evidence; the LLM will use it to compose the answer."""
    return [
        {
            "text": result.chunk.text,
            "source": result.chunk.source,
            "page": result.chunk.page,
            "score": round(result.score, 4),
        }
        for result in vector_store.search(arguments.query, top_k=3)
    ]


def _tool_definition(
    name: str,
    description: str,
    arguments_model: type[BaseModel],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": arguments_model.model_json_schema(),
        },
    }


TOOL_DEFINITIONS = [
    _tool_definition(
        "get_inventory",
        "Get exact current inventory for one product ID, including on-hand, "
        "allocated, and available units.",
        ProductIdArguments,
    ),
    _tool_definition(
        "get_approved_suppliers",
        "List approved supplier options for one exact product ID.",
        ProductIdArguments,
    ),
    _tool_definition(
        "get_open_purchase_orders",
        "List confirmed or in-transit purchase orders for one exact product ID.",
        ProductIdArguments,
    ),
    _tool_definition(
        "assess_replenishment",
        "Calculate stock risk, reorder point, draft order quantity, estimated "
        "cost, required approver, and data-quality warnings for one product. "
        "This tool prepares a draft only and never places or approves an order.",
        ProductIdArguments,
    ),
    _tool_definition(
        "search_policy",
        "Search internal policy documents for rules, formulas, approval limits, "
        "definitions, or procedures. Use this instead of inventory tools for "
        "questions about what the company policy says.",
        PolicySearchArguments,
    ),
]


def execute_tool(
    name: str,
    raw_arguments: dict[str, Any],
    repository: SupplyChainRepository,
    vector_store: SearchableVectorStore | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Validate and execute one allow-listed tool request."""
    if name == "search_policy":
        if vector_store is None:
            raise RuntimeError("The policy vector store is unavailable.")
        arguments = PolicySearchArguments.model_validate(raw_arguments)
        return search_policy(arguments, vector_store)

    product_functions = {
        "get_inventory": get_inventory,
        "get_approved_suppliers": get_approved_suppliers,
        "get_open_purchase_orders": get_open_purchase_orders,
        "assess_replenishment": get_replenishment_assessment,
    }
    function = product_functions.get(name)
    if function is None:
        raise UnknownToolError(f"Unknown tool: {name}")
    arguments = ProductIdArguments.model_validate(raw_arguments)
    return function(arguments, repository)
