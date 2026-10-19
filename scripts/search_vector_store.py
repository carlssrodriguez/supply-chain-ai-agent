"""Search the persisted FAISS index without re-embedding the documents."""

import argparse
from pathlib import Path

from supply_chain_agent.embeddings import LocalEmbeddingModel
from supply_chain_agent.vector_store import FaissVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = PROJECT_ROOT / "data" / "vector_store"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question used for semantic search")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results")
    args = parser.parse_args()

    store = FaissVectorStore(STORE_DIR, LocalEmbeddingModel())
    results = store.search(args.query, top_k=args.top_k)

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(
            f"\n{rank}. score={result.score:.4f} source={chunk.source} "
            f"page={chunk.page} chunk={chunk.chunk_index}"
        )
        print(chunk.text)


if __name__ == "__main__":
    main()
