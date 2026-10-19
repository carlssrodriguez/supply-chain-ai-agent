"""Document extraction and transparent text chunking utilities."""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


PAGE_FURNITURE = {
    "NORTHSTAR COMPONENTS",
    "Internal Policy",
    "Confidential - Training Dataset",
}


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable piece of a source document and its traceability data."""

    text: str
    source: str
    page: int
    chunk_index: int


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """Extract normalized text from each page of a PDF."""
    reader = PdfReader(pdf_path)
    pages: list[str] = []

    for page in reader.pages:
        raw_text = page.extract_text() or ""
        content_lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip()
            and line.strip() not in PAGE_FURNITURE
            and not line.strip().startswith("Page ")
        ]
        normalized_text = " ".join(" ".join(content_lines).split())
        pages.append(normalized_text)

    return pages


def split_text(
    text: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    """Split text into overlapping chunks, preferring word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized_text = " ".join(text.split())
    if not normalized_text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(normalized_text):
        proposed_end = min(start + chunk_size, len(normalized_text))
        end = proposed_end

        if proposed_end < len(normalized_text):
            last_space = normalized_text.rfind(" ", start, proposed_end)
            if last_space > start:
                end = last_space

        chunks.append(normalized_text[start:end].strip())

        if end == len(normalized_text):
            break

        next_start = max(0, end - chunk_overlap)
        next_space = normalized_text.find(" ", next_start)
        start = next_space + 1 if next_space != -1 else end

    return chunks


def split_sections(text: str) -> list[str]:
    """Split the controlled policy format at numbered section headings."""
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return []

    sections = re.split(r"(?=\b\d+\.\s+[A-Z])", normalized_text)
    return [section.strip() for section in sections if section.strip()]


def load_pdf_chunks(
    pdf_path: Path,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    """Extract a PDF and return traceable chunks for later indexing."""
    chunks: list[DocumentChunk] = []

    for page_number, page_text in enumerate(extract_pdf_pages(pdf_path), start=1):
        page_chunks = []
        for section in split_sections(page_text):
            page_chunks.extend(
                split_text(
                    section,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source=pdf_path.name,
                    page=page_number,
                    chunk_index=chunk_index,
                )
            )

    return chunks
