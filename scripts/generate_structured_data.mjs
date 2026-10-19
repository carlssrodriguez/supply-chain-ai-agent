import fs from "node:fs/promises";
import path from "node:path";

import { Workbook } from "@oai/artifact-tool";


const projectRoot = path.resolve(import.meta.dirname, "..");
const outputDir = path.join(projectRoot, "data", "structured");

const datasets = {
  "inventory.csv": {
    sheetName: "Inventory",
    headers: [
      "product_id",
      "product_name",
      "category",
      "service_class",
      "on_hand_units",
      "allocated_units",
      "average_daily_demand",
      "safety_stock_units",
      "unit_cost_eur",
      "review_cycle_days",
      "inventory_snapshot_date",
    ],
    rows: [
      ["SKU-101", "Industrial Sensor Module", "Electronics", "critical", 500, 50, 20, 200, 18.5, 7, "2026-02-10"],
      ["SKU-102", "Precision Control Valve", "Mechanical", "critical", 120, 20, 25, 250, 31.2, 7, "2026-02-10"],
      ["SKU-103", "Assembly Fastener Kit", "Hardware", "standard", 900, 100, 60, 420, 2.4, 14, "2026-02-10"],
      ["SKU-104", "Lithium Battery Pack", "Electronics", "critical", 120, 0, 25, 250, 8.1, 7, "2026-02-10"],
      ["SKU-105", "Protective Packaging Set", "Packaging", "non-critical", 600, 100, 50, 150, 1.65, 30, "2026-02-10"],
    ],
  },
  "suppliers.csv": {
    sheetName: "Suppliers",
    headers: [
      "supplier_id",
      "supplier_name",
      "product_id",
      "lead_time_days",
      "minimum_order_quantity",
      "unit_price_eur",
      "approved",
      "last_review_date",
      "on_time_delivery_rate",
    ],
    rows: [
      ["SUP-001", "Iberia Components", "SKU-101", 9, 100, 18.5, true, "2025-12-01", 0.94],
      ["SUP-003", "Nordic Industrial Systems", "SKU-101", 6, 150, 19.1, true, "2025-11-15", 0.97],
      ["SUP-004", "Valvemec Europe", "SKU-102", 14, 50, 31.2, true, "2025-10-20", 0.91],
      ["SUP-005", "FastFlow Manufacturing", "SKU-102", 8, 100, 33.0, true, "2025-12-12", 0.95],
      ["SUP-006", "Fixing Solutions", "SKU-103", 5, 500, 2.4, true, "2025-09-30", 0.96],
      ["SUP-001", "Iberia Components", "SKU-104", 12, 100, 8.1, true, "2025-12-01", 0.94],
      ["SUP-002", "RapidCell Europe", "SKU-104", 6, 50, 8.55, true, "2025-12-18", 0.98],
      ["SUP-007", "PackRight Logistics", "SKU-105", 4, 200, 1.65, true, "2025-11-02", 0.93],
      ["SUP-008", "Budget Packaging Group", "SKU-105", 7, 500, 1.45, false, "2024-09-01", 0.82],
    ],
  },
  "purchase_orders.csv": {
    sheetName: "PurchaseOrders",
    headers: [
      "purchase_order_id",
      "product_id",
      "supplier_id",
      "quantity",
      "order_date",
      "expected_delivery_date",
      "status",
      "unit_price_eur",
    ],
    rows: [
      ["PO-2026-001", "SKU-101", "SUP-003", 300, "2026-02-05", "2026-02-11", "in_transit", 19.1],
      ["PO-2026-002", "SKU-102", "SUP-005", 300, "2026-02-06", "2026-02-14", "confirmed", 33.0],
      ["PO-2026-003", "SKU-103", "SUP-006", 1000, "2026-02-04", "2026-02-09", "received", 2.4],
      ["PO-2026-004", "SKU-105", "SUP-007", 400, "2026-02-08", "2026-02-12", "in_transit", 1.65],
      ["PO-2025-098", "SKU-104", "SUP-001", 200, "2025-12-10", "2025-12-22", "received", 8.1],
    ],
  },
};

function csvCell(value) {
  const text = String(value);
  if (/[",\r\n]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function toCsv(headers, rows) {
  return [headers, ...rows]
    .map((row) => row.map(csvCell).join(","))
    .join("\n") + "\n";
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function validateData() {
  const inventory = datasets["inventory.csv"];
  const suppliers = datasets["suppliers.csv"];
  const orders = datasets["purchase_orders.csv"];

  const productIds = inventory.rows.map((row) => row[0]);
  assert(new Set(productIds).size === productIds.length, "Duplicate product_id");

  const supplierProductKeys = suppliers.rows.map((row) => `${row[0]}|${row[2]}`);
  assert(
    new Set(supplierProductKeys).size === supplierProductKeys.length,
    "Duplicate supplier-product relationship",
  );

  for (const row of inventory.rows) {
    assert(row.slice(4, 10).every((value) => Number(value) >= 0), `Negative inventory value for ${row[0]}`);
    assert(row[9] > 0, `Invalid review cycle for ${row[0]}`);
    assert(!Number.isNaN(Date.parse(row[10])), `Invalid inventory date for ${row[0]}`);
  }

  for (const row of suppliers.rows) {
    assert(productIds.includes(row[2]), `Unknown supplier product ${row[2]}`);
    assert(row[3] > 0 && row[4] > 0 && row[5] > 0, `Invalid supplier values for ${row[0]}`);
    assert(row[8] >= 0 && row[8] <= 1, `Invalid delivery rate for ${row[0]}`);
    assert(!Number.isNaN(Date.parse(row[7])), `Invalid supplier review date for ${row[0]}`);
  }

  const orderIds = orders.rows.map((row) => row[0]);
  assert(new Set(orderIds).size === orderIds.length, "Duplicate purchase_order_id");

  for (const row of orders.rows) {
    assert(productIds.includes(row[1]), `Unknown order product ${row[1]}`);
    assert(supplierProductKeys.includes(`${row[2]}|${row[1]}`), `Invalid supplier-product pair for ${row[0]}`);
    assert(row[3] > 0 && row[7] > 0, `Invalid order value for ${row[0]}`);
    assert(!Number.isNaN(Date.parse(row[4])) && !Number.isNaN(Date.parse(row[5])), `Invalid order date for ${row[0]}`);
    assert(Date.parse(row[5]) >= Date.parse(row[4]), `Delivery precedes order date for ${row[0]}`);
  }
}

async function main() {
  validateData();
  await fs.mkdir(outputDir, { recursive: true });

  for (const [filename, dataset] of Object.entries(datasets)) {
    const csvText = toCsv(dataset.headers, dataset.rows);
    const workbook = await Workbook.fromCSV(csvText, { sheetName: dataset.sheetName });
    const lastColumn = String.fromCharCode(64 + dataset.headers.length);
    const inspection = await workbook.inspect({
      kind: "region",
      sheetId: dataset.sheetName,
      range: `A1:${lastColumn}${dataset.rows.length + 1}`,
      maxChars: 5000,
      tableMaxRows: 20,
      tableMaxCols: 12,
    });
    console.log(`\n${filename}\n${inspection.ndjson}`);
    await fs.writeFile(path.join(outputDir, filename), csvText, "utf8");
  }

  console.log(`\nCreated ${Object.keys(datasets).length} validated CSV files in ${outputDir}`);
}

await main();
