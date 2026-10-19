"""Ask one grounded question using the complete RAG pipeline."""

import argparse
import sys
from pathlib import Path

from supply_chain_agent.embeddings import LocalEmbeddingModel
from supply_chain_agent.rag import answer_with_rag
from supply_chain_agent.vector_store import FaissVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = PROJECT_ROOT / "data" / "vector_store"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question about the indexed policies")
    parser.add_argument("--top-k", type=int, default=3, help="Retrieved chunks")
    args = parser.parse_args()

    store = FaissVectorStore(STORE_DIR, LocalEmbeddingModel())
    result = answer_with_rag(args.question, store, top_k=args.top_k)

    print(f"\nANSWER\n{result.answer}")
    print("\nRETRIEVED SOURCES")
    for source in result.retrieved_sources:
        print(f"- {source.document}, page {source.page}")


if __name__ == "__main__":
    main()
