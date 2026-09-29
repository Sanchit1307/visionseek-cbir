"""Hand-computed sanity check for src/evaluate.py.

Run from the repo root:  python tests\\test_evaluate.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import evaluate, rank_by_similarity  # noqa: E402


def test_toy_metrics() -> None:
    gallery_labels = np.array([0, 1, 0, 1, 0, 1])
    query_labels = np.array([0, 1])
    ranked = np.array([
        [0, 1, 2, 3, 4, 5],   # relevance 1 0 1 0 1 0
        [1, 3, 5, 0, 2, 4],   # relevance 1 1 1 0 0 0
    ])
    res = evaluate(ranked, query_labels, gallery_labels, ks=(2, 3))
    # Query 1: P@2 = 1/2, P@3 = 2/3, AP = (1/1 + 2/3 + 3/5) / 3 = 0.755556
    # Query 2: P@2 = 1,   P@3 = 1,   AP = 1
    assert abs(res["P@2"] - 0.75) < 1e-9, res
    assert abs(res["P@3"] - (2 / 3 + 1) / 2) < 1e-9, res
    assert abs(res["mAP"] - (0.7555555555555555 + 1.0) / 2) < 1e-9, res


def test_rank_excludes_self() -> None:
    vecs = np.eye(4, dtype=np.float32)
    vecs[1] = vecs[0] * 0.9 + vecs[1] * 0.1
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ranked = rank_by_similarity(vecs[[0]], vecs, exclude_ids=np.array([0]))
    assert ranked.shape == (1, 3)
    assert 0 not in ranked[0]
    assert ranked[0, 0] == 1


if __name__ == "__main__":
    test_toy_metrics()
    test_rank_excludes_self()
    print("evaluate.py tests passed")