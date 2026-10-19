from pathlib import Path

import pytest

from supply_chain_agent.data_repository import (
    ProductNotFoundError,
    SupplyChainRepository,
)


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "structured"


@pytest.fixture
def repository() -> SupplyChainRepository:
    return SupplyChainRepository(DATA_DIRECTORY)


def test_get_inventory_returns_exact_product(repository: SupplyChainRepository) -> None:
    item = repository.get_inventory("sku-104")

    assert item.product_id == "SKU-104"
    assert item.on_hand_units == 120
    assert item.allocated_units == 0
    assert item.available_units == 120


def test_get_approved_suppliers_excludes_unapproved_supplier(
    repository: SupplyChainRepository,
) -> None:
    suppliers = repository.get_approved_suppliers("SKU-105")

    assert [supplier.supplier_id for supplier in suppliers] == ["SUP-007"]


def test_get_open_purchase_orders_returns_only_open_statuses(
    repository: SupplyChainRepository,
) -> None:
    orders = repository.get_open_purchase_orders("SKU-102")

    assert [order.purchase_order_id for order in orders] == ["PO-2026-002"]


def test_received_purchase_order_is_not_open(
    repository: SupplyChainRepository,
) -> None:
    assert repository.get_open_purchase_orders("SKU-104") == []


def test_unknown_product_has_a_controlled_error(
    repository: SupplyChainRepository,
) -> None:
    with pytest.raises(ProductNotFoundError, match="SKU-999"):
        repository.get_inventory("SKU-999")
