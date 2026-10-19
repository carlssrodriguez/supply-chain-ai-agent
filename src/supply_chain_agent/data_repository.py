"""Deterministic access to the structured supply-chain CSV files."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "structured"
OPEN_PURCHASE_ORDER_STATUSES = {"confirmed", "in_transit"}


class ProductNotFoundError(LookupError):
    """Raised when an exact product identifier does not exist."""


class InventoryItem(BaseModel):
    product_id: str
    product_name: str
    category: str
    service_class: Literal["critical", "standard", "non-critical"]
    on_hand_units: int = Field(ge=0)
    allocated_units: int = Field(ge=0)
    average_daily_demand: float = Field(ge=0)
    safety_stock_units: int = Field(ge=0)
    unit_cost_eur: float = Field(ge=0)
    review_cycle_days: int = Field(gt=0)
    inventory_snapshot_date: date

    @property
    def available_units(self) -> int:
        """Units that are physically present and not already reserved."""
        return self.on_hand_units - self.allocated_units


class SupplierOption(BaseModel):
    supplier_id: str
    supplier_name: str
    product_id: str
    lead_time_days: int = Field(ge=0)
    minimum_order_quantity: int = Field(gt=0)
    unit_price_eur: float = Field(ge=0)
    approved: bool
    last_review_date: date
    on_time_delivery_rate: float = Field(ge=0, le=1)


class PurchaseOrder(BaseModel):
    purchase_order_id: str
    product_id: str
    supplier_id: str
    quantity: int = Field(gt=0)
    order_date: date
    expected_delivery_date: date
    status: Literal["confirmed", "in_transit", "received", "cancelled"]
    unit_price_eur: float = Field(ge=0)


class SupplyChainRepository:
    """Read exact, validated records from the project's CSV data."""

    def __init__(self, data_directory: Path = DEFAULT_DATA_DIRECTORY) -> None:
        self.data_directory = data_directory

    def get_inventory(self, product_id: str) -> InventoryItem:
        normalized_id = product_id.strip().upper()
        for row in self._read_csv("inventory.csv"):
            if row["product_id"].upper() == normalized_id:
                return InventoryItem.model_validate(row)
        raise ProductNotFoundError(f"Unknown product_id: {product_id}")

    def get_approved_suppliers(self, product_id: str) -> list[SupplierOption]:
        # Confirm the product exists so an unknown ID is not confused with
        # a real product that simply has no approved suppliers.
        self.get_inventory(product_id)
        normalized_id = product_id.strip().upper()
        return [
            supplier
            for row in self._read_csv("suppliers.csv")
            if row["product_id"].upper() == normalized_id
            and (supplier := SupplierOption.model_validate(row)).approved
        ]

    def get_open_purchase_orders(self, product_id: str) -> list[PurchaseOrder]:
        self.get_inventory(product_id)
        normalized_id = product_id.strip().upper()
        return [
            order
            for row in self._read_csv("purchase_orders.csv")
            if row["product_id"].upper() == normalized_id
            and (order := PurchaseOrder.model_validate(row)).status
            in OPEN_PURCHASE_ORDER_STATUSES
        ]

    def _read_csv(self, filename: str) -> list[dict[str, str]]:
        path = self.data_directory / filename
        with path.open(encoding="utf-8", newline="") as csv_file:
            return list(csv.DictReader(csv_file))
