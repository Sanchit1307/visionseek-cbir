"""Retrieval evaluation (Person A): Precision@K, mAP and latency.

Protocol: the query image itself must NOT appear in its own ranking. Either
exclude it while ranking (`rank_by_similarity(..., exclude_ids=...)`) or drop
it before calling `evaluate`.
"""

from __future__ import annotations

import time

import numpy as np


def evaluate(ranked_ids: np.ndarray,
             query_labels: np.ndarray,
             gallery_labels: np.ndarray,
             ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    """Compute P@K and mAP averaged over queries.

    Args:
        ranked_ids: (Q, M) gallery indices, best match first, self-match already
            removed. For the standard mAP pass the FULL ranking (M = N - 1);
            with a truncated ranking, AP is computed over the given list only.
        query_labels: (Q,) class label of each query.
        gallery_labels: (N,) class label of each gallery image.
        ks: cut-offs for precision; M must be >= max(ks).

    Returns:
        {'P@5': ..., 'P@10': ..., 'mAP': ...} (keys follow `ks`).
    """
    ranked_ids = np.asarray(ranked_ids)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)

    if ranked_ids.ndim != 2:
        raise ValueError("ranked_ids must be 2-D (Q, M)")
    n_q, m = ranked_ids.shape
    if query_labels.shape[0] != n_q:
        raise ValueError("query_labels length must equal number of queries")
    if m < max(ks):
        raise ValueError(f"ranking length {m} is shorter than max(ks)={max(ks)}")

    rel = (gallery_labels[ranked_ids] == query_labels[:, None]).astype(np.float64)

    out: dict[str, float] = {}
    for k in ks:
        out[f"P@{k}"] = float(rel[:, :k].sum(axis=1).mean() / k)

    ranks = np.arange(1, m + 1, dtype=np.float64)
    prec_at_i = np.cumsum(rel, axis=1) / ranks
    n_rel = rel.sum(axis=1)
    ap = np.where(n_rel > 0,
                  (prec_at_i * rel).sum(axis=1) / np.maximum(n_rel, 1.0),
                  0.0)
    out["mAP"] = float(ap.mean())
    return out


def rank_by_similarity(query_vecs: np.ndarray,
                       gallery_vecs: np.ndarray,
                       exclude_ids: np.ndarray | None = None,
                       chunk: int = 256) -> np.ndarray:
    """Exact cosine ranking (vectors must be L2-normalised).

    Args:
        query_vecs: (Q, D) float32.
        gallery_vecs: (N, D) float32.
        exclude_ids: (Q,) gallery index of each query, removed from its own
            ranking (use when the queries are gallery images).
    Returns:
        (Q, N) or (Q, N-1) int64 array of gallery indices, best first.
    """
    q = np.asarray(query_vecs, dtype=np.float32)
    g = np.asarray(gallery_vecs, dtype=np.float32)
    n = g.shape[0]
    if exclude_ids is not None:
        exclude_ids = np.asarray(exclude_ids)
        if exclude_ids.shape[0] != q.shape[0]:
            raise ValueError("exclude_ids must have one entry per query")

    keep = n - 1 if exclude_ids is not None else n
    out = np.empty((q.shape[0], keep), dtype=np.int64)
    for s in range(0, q.shape[0], chunk):
        sims = q[s:s + chunk] @ g.T
        if exclude_ids is not None:
            sims[np.arange(sims.shape[0]), exclude_ids[s:s + chunk]] = -np.inf
        order = np.argsort(-sims, axis=1, kind="stable")
        out[s:s + chunk] = order[:, :keep]  # the excluded id sorts last
    return out


def measure_search_latency(query_vecs: np.ndarray,
                           gallery_vecs: np.ndarray,
                           k: int = 10,
                           warmup: int = 5) -> float:
    """Mean per-query exact top-k search time in milliseconds (NumPy)."""
    q = np.asarray(query_vecs, dtype=np.float32)
    g = np.asarray(gallery_vecs, dtype=np.float32)
    if k >= g.shape[0]:
        raise ValueError("k must be smaller than the gallery size")

    def one(v: np.ndarray) -> np.ndarray:
        sims = g @ v
        top = np.argpartition(-sims, k)[:k]
        return top[np.argsort(-sims[top])]

    for i in range(min(warmup, q.shape[0])):
        one(q[i])
    t0 = time.perf_counter()
    for i in range(q.shape[0]):
        one(q[i])
    return (time.perf_counter() - t0) / q.shape[0] * 1000.0