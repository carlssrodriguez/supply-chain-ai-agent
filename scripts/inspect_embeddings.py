"""Inspect local vectors and semantic search results before adding FAISS."""

from pathlib import Path

from supply_chain_agent.documents import load_pdf_chunks
from supply_chain_agent.embeddings import LocalEmbeddingModel, semantic_search


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = PROJECT_ROOT / "data" / "documents" / "inventory_policy.pdf"

QUESTIONS = [
    "What is the minimum safety stock for critical products?",
    "Who must approve a purchase order worth EUR 20,000?",
    "When is a critical item considered at high risk of stockout?",
]


def main() -> None:
    chunks = load_pdf_chunks(DOCUMENT_PATH)
    model = LocalEmbeddingModel()
    first_vector = model.embed_passages([chunks[0].text])[0]

    print(f"Embedding dimensions: {len(first_vector)}")
    print(f"First 8 values: {first_vector[:8]}")

    for question in QUESTIONS:
        print("\n" + "=" * 100)
        print(f"QUESTION: {question}")

        for rank, result in enumerate(
            semantic_search(question, chunks, model, top_k=3),
            start=1,
        ):
            chunk = result.chunk
            preview = chunk.text[:220]
            print(
                f"\n{rank}. score={result.score:.4f} source={chunk.source} "
                f"page={chunk.page} chunk={chunk.chunk_index}"
            )
            print(f"   {preview}...")


if __name__ == "__main__":
    main()
