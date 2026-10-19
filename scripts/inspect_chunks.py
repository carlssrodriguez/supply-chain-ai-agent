"""Print document chunks so their quality can be inspected before indexing."""

from pathlib import Path

from supply_chain_agent.documents import load_pdf_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = PROJECT_ROOT / "data" / "documents" / "inventory_policy.pdf"


def main() -> None:
    chunks = load_pdf_chunks(DOCUMENT_PATH)

    print(f"Document: {DOCUMENT_PATH.name}")
    print(f"Chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 80)
        print(
            f"source={chunk.source} page={chunk.page} "
            f"chunk_index={chunk.chunk_index} characters={len(chunk.text)}"
        )
        print("-" * 80)
        print(chunk.text)


if __name__ == "__main__":
    main()
