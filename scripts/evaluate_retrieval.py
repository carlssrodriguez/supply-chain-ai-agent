"""Evaluate whether semantic search retrieves known policy evidence."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from supply_chain_agent.embeddings import LocalEmbeddingModel
from supply_chain_agent.hybrid_search import HybridRetriever
from supply_chain_agent.vector_store import FaissVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "retrieval_questions.json"
STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
MAX_K = 3


@dataclass(frozen=True)
class QuestionResult:
    """Retrieval position of the expected evidence for one question."""

    question_id: str
    rank: int | None
    top_score: float


def evaluate_question(
    item: dict[str, object],
    store: FaissVectorStore,
) -> QuestionResult:
    """Find the rank of the first chunk containing the expected evidence."""
    question = str(item["question"])
    results = store.search(question, top_k=MAX_K)
    expected_phrases = item.get("evidence_contains_any") or [
        item["evidence_contains"]
    ]
    expected = [str(phrase).casefold() for phrase in expected_phrases]
    rank = next(
        (
            position
            for position, result in enumerate(results, start=1)
            if any(
                phrase in result.chunk.text.casefold()
                for phrase in expected
            )
        ),
        None,
    )

    return QuestionResult(
        question_id=item["id"],
        rank=rank,
        top_score=results[0].score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Combine FAISS semantic ranking with BM25 lexical ranking",
    )
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    vector_store = FaissVectorStore(STORE_DIR, LocalEmbeddingModel())
    store = HybridRetriever(vector_store) if args.hybrid else vector_store
    results = [evaluate_question(item, store) for item in dataset]

    hit_at_1 = sum(result.rank == 1 for result in results) / len(results)
    hit_at_3 = sum(result.rank is not None for result in results) / len(results)
    mrr = sum(1 / result.rank if result.rank else 0 for result in results) / len(results)

    retrieval_name = "HYBRID" if args.hybrid else "FAISS"
    print(f"RETRIEVAL EVALUATION ({retrieval_name})")
    print("=" * 72)
    for result in results:
        rank = str(result.rank) if result.rank is not None else "MISS"
        print(
            f"{result.question_id:<28} rank={rank:<4} "
            f"top_score={result.top_score:.4f}"
        )

    print("=" * 72)
    print(f"Questions: {len(results)}")
    print(f"Hit@1:    {hit_at_1:.1%}")
    print(f"Hit@3:    {hit_at_3:.1%}")
    print(f"MRR:      {mrr:.3f}")


if __name__ == "__main__":
    main()
