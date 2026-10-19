"""Tests for vector similarity calculations."""

import numpy as np
import pytest

from supply_chain_agent.embeddings import cosine_scores


def test_cosine_scores_rank_same_direction_highest() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    passages = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    scores = cosine_scores(query, passages)

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(-1.0)


def test_cosine_scores_reject_zero_vector() -> None:
    query = np.array([0.0, 0.0], dtype=np.float32)
    passages = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="zero-length vector"):
        cosine_scores(query, passages)
