"""Deterministic replenishment calculations based on company policy."""

from __future__ import annotations

import math
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from supply_chain_agent.data_repository import SupplyChainRepository


class ReplenishmentAssessment(BaseModel):
    product_id: str
    product_name: str
    risk_level: Literal["none", "normal", "high"]
    available_units: int
    confirmed_inbound_units: int
    inventory_position_units: int
    days_of_cover: float | None = Field(ge=0)
    fastest_lead_time_days: int = Field(gt=0)
    reorder_point_units: int = Field(ge=0)
    replenishment_review_required: bool
    target_stock_units: int = Field(ge=0)
    recommended_order_quantity: int = Field(ge=0)
    recommended_supplier_id: str
    estimated_order_value_eur: float = Field(ge=0)
    required_approver: str | None
    status: Literal["no_action", "pending_approval", "pending_human_review"]
    warnings: list[str]


def assess_replenishment(
    product_id: str,
    repository: SupplyChainRepository,
    *,
    as_of_date: date | None = None,
) -> ReplenishmentAssessment:
    """Calculate stock risk and a draft order without taking any action."""
    as_of_date = as_of_date or date.today()
    item = repository.get_inventory(product_id)
    suppliers = repository.get_approved_suppliers(product_id)
    open_orders = repository.get_open_purchase_orders(product_id)

    if not suppliers:
        raise ValueError(f"No approved supplier exists for {item.product_id}")

    supplier = min(
        suppliers,
        key=lambda option: (option.lead_time_days, option.unit_price_eur),
    )
    confirmed_inbound = sum(order.quantity for order in open_orders)
    inventory_position = (
        item.on_hand_units + confirmed_inbound - item.allocated_units
    )
    reorder_point = math.ceil(
        item.average_daily_demand * supplier.lead_time_days
        + item.safety_stock_units
    )
    review_required = inventory_position <= reorder_point
    target_stock = math.ceil(
        item.average_daily_demand
        * (supplier.lead_time_days + item.review_cycle_days)
        + item.safety_stock_units
    )

    raw_quantity = max(0, target_stock - inventory_position)
    recommended_quantity = (
        math.ceil(raw_quantity / supplier.minimum_order_quantity)
        * supplier.minimum_order_quantity
        if review_required and raw_quantity > 0
        else 0
    )
    order_value = round(recommended_quantity * supplier.unit_price_eur, 2)

    days_of_cover = (
        round(item.available_units / item.average_daily_demand, 2)
        if item.average_daily_demand > 0
        else None
    )
    high_risk = (
        item.service_class == "critical"
        and days_of_cover is not None
        and days_of_cover < supplier.lead_time_days
    )
    risk_level: Literal["none", "normal", "high"] = (
        "high" if high_risk else "normal" if review_required else "none"
    )

    warnings = _data_quality_warnings(
        item.inventory_snapshot_date,
        [option.last_review_date for option in suppliers],
        [(order.purchase_order_id, order.expected_delivery_date) for order in open_orders],
        as_of_date,
    )
    status: Literal["no_action", "pending_approval", "pending_human_review"]
    if warnings:
        status = "pending_human_review"
    elif recommended_quantity > 0:
        status = "pending_approval"
    else:
        status = "no_action"

    return ReplenishmentAssessment(
        product_id=item.product_id,
        product_name=item.product_name,
        risk_level=risk_level,
        available_units=item.available_units,
        confirmed_inbound_units=confirmed_inbound,
        inventory_position_units=inventory_position,
        days_of_cover=days_of_cover,
        fastest_lead_time_days=supplier.lead_time_days,
        reorder_point_units=reorder_point,
        replenishment_review_required=review_required,
        target_stock_units=target_stock,
        recommended_order_quantity=recommended_quantity,
        recommended_supplier_id=supplier.supplier_id,
        estimated_order_value_eur=order_value,
        required_approver=_required_approver(order_value),
        status=status,
        warnings=warnings,
    )


def _required_approver(order_value_eur: float) -> str | None:
    if order_value_eur <= 0:
        return None
    if order_value_eur <= 10_000:
        return "Procurement Manager"
    if order_value_eur <= 50_000:
        return "Director of Supply Chain"
    return "Chief Operating Officer"


def _data_quality_warnings(
    inventory_snapshot_date: date,
    supplier_review_dates: list[date],
    open_order_deliveries: list[tuple[str, date]],
    as_of_date: date,
) -> list[str]:
    warnings: list[str] = []
    if (as_of_date - inventory_snapshot_date).days > 1:
        warnings.append(
            f"Inventory snapshot is stale: {inventory_snapshot_date.isoformat()}."
        )
    if any((as_of_date - review_date).days > 180 for review_date in supplier_review_dates):
        warnings.append("At least one approved supplier review is older than 180 days.")
    overdue_orders = [
        order_id
        for order_id, delivery_date in open_order_deliveries
        if delivery_date < as_of_date
    ]
    if overdue_orders:
        warnings.append(
            "Open purchase orders have past expected-delivery dates: "
            + ", ".join(overdue_orders)
            + "."
        )
    return warnings
