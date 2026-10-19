"""Tests for PDF extraction and chunking."""

from pathlib import Path

import pytest

from supply_chain_agent.documents import extract_pdf_pages, split_sections, split_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "data" / "documents" / "inventory_policy.pdf"


def test_extract_pdf_pages_removes_repeated_page_furniture() -> None:
    pages = extract_pdf_pages(POLICY_PATH)

    assert len(pages) == 2
    assert all("NORTHSTAR COMPONENTS" not in page for page in pages)
    assert all("Confidential - Training Dataset" not in page for page in pages)
    assert "Inventory Planning and Replenishment Policy" in pages[0]


def test_split_text_creates_overlapping_chunks() -> None:
    text = " ".join(f"word{number}" for number in range(100))

    chunks = split_text(text, chunk_size=120, chunk_overlap=30)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_split_text_rejects_overlap_equal_to_chunk_size() -> None:
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        split_text("Example text", chunk_size=100, chunk_overlap=100)


def test_split_text_returns_no_chunks_for_blank_text() -> None:
    assert split_text("   \n  ") == []


def test_split_sections_keeps_numbered_topics_separate() -> None:
    text = "Front matter 1. Purpose First topic. 2. Product classes Second topic."

    sections = split_sections(text)

    assert sections == [
        "Front matter",
        "1. Purpose First topic.",
        "2. Product classes Second topic.",
    ]
