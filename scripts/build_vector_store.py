"""Build the local FAISS index from the project's source documents."""

from pathlib import Path

from supply_chain_agent.documents import load_pdf_chunks
from supply_chain_agent.embeddings import LocalEmbeddingModel
from supply_chain_agent.vector_store import build_vector_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = PROJECT_ROOT / "data" / "documents" / "inventory_policy.pdf"
STORE_DIR = PROJECT_ROOT / "data" / "vector_store"


def main() -> None:
    chunks = load_pdf_chunks(DOCUMENT_PATH)
    model = LocalEmbeddingModel()
    build_vector_store(chunks, model, STORE_DIR)

    print(f"Indexed {len(chunks)} chunks")
    print(f"Vector store: {STORE_DIR}")


if __name__ == "__main__":
    main()
