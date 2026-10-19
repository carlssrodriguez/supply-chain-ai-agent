from datetime import date

from supply_chain_agent.data_repository import SupplyChainRepository
from supply_chain_agent.replenishment import assess_replenishment


def test_high_risk_item_produces_rounded_draft_recommendation() -> None:
    result = assess_replenishment(
        "SKU-104",
        SupplyChainRepository(),
        as_of_date=date(2026, 2, 10),
    )

    assert result.available_units == 120
    assert result.confirmed_inbound_units == 0
    assert result.inventory_position_units == 120
    assert result.days_of_cover == 4.8
    assert result.fastest_lead_time_days == 6
    assert result.reorder_point_units == 400
    assert result.target_stock_units == 575
    assert result.recommended_order_quantity == 500
    assert result.recommended_supplier_id == "SUP-002"
    assert result.estimated_order_value_eur == 4275.0
    assert result.required_approver == "Procurement Manager"
    assert result.risk_level == "high"
    assert result.status == "pending_approval"
    assert result.warnings == []


def test_open_order_is_included_in_inventory_position() -> None:
    result = assess_replenishment(
        "SKU-102",
        SupplyChainRepository(),
        as_of_date=date(2026, 2, 10),
    )

    assert result.confirmed_inbound_units == 300
    assert result.inventory_position_units == 400
    assert result.recommended_order_quantity == 300


def test_stale_or_contradictory_data_requires_human_review() -> None:
    result = assess_replenishment(
        "SKU-102",
        SupplyChainRepository(),
        as_of_date=date(2026, 10, 19),
    )

    assert result.status == "pending_human_review"
    assert any("Inventory snapshot is stale" in warning for warning in result.warnings)
    assert any("older than 180 days" in warning for warning in result.warnings)
    assert any("PO-2026-002" in warning for warning in result.warnings)


def test_healthy_inventory_requires_no_order() -> None:
    result = assess_replenishment(
        "SKU-101",
        SupplyChainRepository(),
        as_of_date=date(2026, 2, 10),
    )

    assert result.inventory_position_units == 750
    assert result.reorder_point_units == 320
    assert result.recommended_order_quantity == 0
    assert result.risk_level == "none"
    assert result.status == "no_action"
