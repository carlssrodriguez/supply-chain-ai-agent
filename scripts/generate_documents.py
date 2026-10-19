"""Generate the fictional internal documents used by the RAG system."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_PATH = OUTPUT_DIR / "inventory_policy.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2D6A9F")
PALE_BLUE = colors.HexColor("#EAF2F8")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
DARK_GRAY = colors.HexColor("#3F4A54")


def _draw_page(canvas, document) -> None:
    """Add a consistent header, footer, and page number."""
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D7DEE5"))
    canvas.line(18 * mm, height - 17 * mm, width - 18 * mm, height - 17 * mm)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(NAVY)
    canvas.drawString(18 * mm, height - 13 * mm, "NORTHSTAR COMPONENTS")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(DARK_GRAY)
    canvas.drawRightString(width - 18 * mm, height - 13 * mm, "Internal Policy")

    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.drawString(18 * mm, 10 * mm, "Confidential - Training Dataset")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_inventory_policy(output_path: Path = OUTPUT_PATH) -> Path:
    """Create the fictional inventory policy PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=24 * mm,
        bottomMargin=22 * mm,
        title="Inventory Planning and Replenishment Policy",
        author="Northstar Components",
        subject="Fictional supply chain policy for an educational RAG system",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="DocumentTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=BLUE,
            spaceBefore=5 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PolicyBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=DARK_GRAY,
            spaceAfter=2.5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=15,
            textColor=NAVY,
            backColor=PALE_BLUE,
            borderColor=BLUE,
            borderWidth=0.8,
            borderPadding=8,
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        )
    )

    story = [
        Spacer(1, 10 * mm),
        Paragraph("Inventory Planning and Replenishment Policy", styles["DocumentTitle"]),
        Paragraph(
            "Policy ID: SC-POL-001 | Version: 1.0 | Effective: 15 January 2026",
            ParagraphStyle(
                "Subtitle",
                parent=styles["BodyText"],
                alignment=TA_CENTER,
                fontSize=10,
                textColor=DARK_GRAY,
                spaceAfter=8 * mm,
            ),
        ),
    ]

    metadata = [
        ["Policy owner", "Director of Supply Chain"],
        ["Applies to", "All stocked finished goods and critical components"],
        ["Review cycle", "Annual, or after a material supply disruption"],
        ["Document status", "Approved for internal use"],
    ]
    metadata_table = Table(metadata, colWidths=[42 * mm, 105 * mm])
    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), PALE_BLUE),
                ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C5D0DA")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([metadata_table, Spacer(1, 5 * mm)])

    sections = [
        (
            "1. Purpose",
            [
                "This policy defines how Northstar Components plans inventory, "
                "identifies stockout risk, and approves replenishment. Its objective "
                "is to protect customer service while avoiding unnecessary working capital.",
            ],
        ),
        (
            "2. Product classifications",
            [
                "Every stocked item must be assigned one of three service classes. "
                "Critical items can stop production or prevent delivery of a contracted "
                "customer order. Standard items support normal operations but have viable "
                "substitutes. Non-critical items have limited operational impact and may "
                "be replenished through routine purchasing cycles.",
            ],
        ),
    ]

    for heading, paragraphs in sections:
        story.append(Paragraph(heading, styles["SectionHeading"]))
        story.extend(Paragraph(text, styles["PolicyBody"]) for text in paragraphs)

    class_data = [
        ["Service class", "Safety stock requirement", "Planning priority"],
        ["Critical", "Greater of 10 days of average demand or 100 units", "Highest"],
        ["Standard", "7 days of average demand", "Normal"],
        ["Non-critical", "3 days of average demand", "Routine"],
    ]
    class_table = Table(class_data, colWidths=[31 * mm, 83 * mm, 33 * mm], repeatRows=1)
    class_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C5D0DA")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([class_table, Spacer(1, 3 * mm)])

    story.extend(
        [
            Paragraph("3. Replenishment calculations", styles["SectionHeading"]),
            Paragraph(
                "Average daily demand must be calculated from the most recent 90 days "
                "of fulfilled demand. Documented exceptional events may be excluded only "
                "with approval from the Inventory Planning Manager.",
                styles["PolicyBody"],
            ),
            Paragraph(
                "Reorder point = (average daily demand x supplier lead time in days) "
                "+ safety stock.",
                styles["Callout"],
            ),
            Paragraph(
                "A replenishment review must be triggered when the inventory position "
                "is at or below the reorder point. Inventory position equals on-hand "
                "stock plus confirmed inbound quantity minus allocated customer demand.",
                styles["PolicyBody"],
            ),
            Paragraph(
                "Recommended order quantity = target stock - inventory position. The "
                "result must be rounded up to the supplier minimum order quantity. "
                "Target stock covers lead-time demand, safety stock, and the next review cycle.",
                styles["PolicyBody"],
            ),
            Paragraph("4. Stockout risk", styles["SectionHeading"]),
            Paragraph(
                "An item is at risk of stockout when projected available inventory before "
                "the next confirmed delivery falls below its required safety stock. A "
                "critical item with fewer days of cover than the fastest approved supplier "
                "lead time must be classified as high risk.",
                styles["PolicyBody"],
            ),
            Paragraph(
                "High-risk critical items must be reviewed on the same business day. The "
                "planner must compare approved suppliers, open purchase orders, transport "
                "options, minimum order quantities, and unit cost before recommending action.",
                styles["PolicyBody"],
            ),
            Paragraph("5. Purchase approval thresholds", styles["SectionHeading"]),
            Paragraph(
                "The AI assistant may prepare a recommendation and a draft purchase order, "
                "but it must never create, approve, transmit, or modify a purchase order "
                "without an explicit human approval recorded by the application.",
                styles["Callout"],
            ),
        ]
    )

    approval_data = [
        ["Total order value", "Minimum approver"],
        ["Up to EUR 10,000", "Procurement Manager"],
        ["EUR 10,000.01 to EUR 50,000", "Director of Supply Chain"],
        ["Above EUR 50,000", "Chief Operating Officer"],
    ]
    approval_table = Table(approval_data, colWidths=[70 * mm, 77 * mm], repeatRows=1)
    approval_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C5D0DA")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend(
        [
            approval_table,
            Spacer(1, 3 * mm),
            Paragraph(
                "Splitting one requirement into multiple orders to avoid an approval "
                "threshold is prohibited. Emergency purchases still require the designated "
                "approver, although approval may be recorded after verbal authorization "
                "when a production shutdown is imminent.",
                styles["PolicyBody"],
            ),
            Paragraph("6. Data quality and exceptions", styles["SectionHeading"]),
            Paragraph(
                "Recommendations must identify missing or stale inputs. Inventory snapshots "
                "older than 24 hours, supplier lead times not reviewed within 180 days, and "
                "unconfirmed inbound orders must be disclosed to the decision maker.",
                styles["PolicyBody"],
            ),
            Paragraph(
                "The assistant must cite the policy section and operational records used. "
                "If required evidence is unavailable or contradictory, it must ask for human "
                "review rather than presenting a definitive recommendation.",
                styles["PolicyBody"],
            ),
            Paragraph("7. Roles and responsibilities", styles["SectionHeading"]),
            KeepTogether(
                [
                    Paragraph(
                        "Inventory Planner: monitors exceptions and prepares replenishment "
                        "recommendations. Procurement Manager: validates supplier and commercial "
                        "terms. Supply Chain Director: owns this policy and approves material "
                        "exceptions. System Owner: maintains access controls, logs, and model "
                        "evaluation records.",
                        styles["PolicyBody"],
                    )
                ]
            ),
            Spacer(1, 8 * mm),
            Paragraph(
                "End of policy - SC-POL-001 version 1.0",
                ParagraphStyle(
                    "EndNote",
                    parent=styles["BodyText"],
                    alignment=TA_CENTER,
                    fontSize=8.5,
                    textColor=DARK_GRAY,
                ),
            ),
        ]
    )

    document.build(story, onFirstPage=_draw_page, onLaterPages=_draw_page)
    return output_path


if __name__ == "__main__":
    generated_path = build_inventory_policy()
    print(generated_path)
