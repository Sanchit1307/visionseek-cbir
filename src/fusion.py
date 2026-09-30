"""Score-level fusion of retrievers (Person B, WP9).

Public API (interface contract):
    fuse(scores, weights, norm="zscore") -> np.ndarray
        scores : {retriever name: similarity array}, all the same shape
                 ((N,) for one query or (Q, N) for many; N = gallery size)
        weights: {retriever name: non-negative weight}; rescaled to sum to 1,
                 zero weights are ignored
        returns the fused similarity, same shape as one input array.

Why normalise: CLIP cosines and the classical cosines live on different scales
(classical vectors are non-negative histograms, so their cosines are high and
bunched together). With norm="zscore" every retriever's scores are standardised
per query over the whole gallery before the weighted sum, so a weight of 0.3
really means "30% of the say". norm="none" uses raw cosines.

The same function serves the fixed hybrid (weights tuned once on the validation
split) and Person A's adaptive fusion (a different weight dict per quality bucket).
`tune` / `grid_weights` are shared helpers so both use the same search code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import evaluate

DEFAULT_WEIGHTS = {"clip": 0.7, "classical_concat": 0.3}   # plan: 0.7 CLIP + 0.3 classical
DEFAULT_NORM = "zscore"
NORMS = ("zscore", "none")


def normalize_scores(s: np.ndarray, method: str = DEFAULT_NORM) -> np.ndarray:
    """Per-query normalisation over the last axis (the gallery)."""
    s = np.asarray(s, dtype=np.float32)
    if method == "none":
        return s
    if method == "zscore":
        mu = s.mean(axis=-1, keepdims=True)
        sd = s.std(axis=-1, keepdims=True)
        return (s - mu) / np.maximum(sd, 1e-8)
    raise ValueError(f"unknown norm '{method}', expected one of {NORMS}")


def fuse(scores: dict[str, np.ndarray], weights: dict[str, float],
         norm: str = DEFAULT_NORM) -> np.ndarray:
    """Weighted sum of (normalised) per-retriever similarities."""
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")
    used = {k: float(w) for k, w in weights.items() if w > 0}
    if not used:
        raise ValueError("at least one weight must be > 0")
    missing = [k for k in used if k not in scores]
    if missing:
        raise KeyError(f"no scores for retriever(s) {missing}")
    shapes = {np.shape(scores[k]) for k in used}
    if len(shapes) != 1:
        raise ValueError(f"score arrays must share one shape, got {shapes}")
    total = sum(used.values())
    out = None
    for k, w in used.items():
        term = (w / total) * normalize_scores(scores[k], norm)
        out = term if out is None else out + term
    return out.astype(np.float32)


def rank_from_scores(scores: np.ndarray, exclude_ids: np.ndarray | None = None) -> np.ndarray:
    """(Q, N) similarities -> (Q, N) or (Q, N-1) gallery ids, best first.

    exclude_ids removes each query's own gallery id (same protocol as
    src.evaluate.rank_by_similarity)."""
    s = np.array(scores, dtype=np.float32, copy=True)
    n = s.shape[1]
    if exclude_ids is not None:
        exclude_ids = np.asarray(exclude_ids)
        s[np.arange(s.shape[0]), exclude_ids] = -np.inf
    order = np.argsort(-s, axis=1, kind="stable")
    return order[:, : n - 1 if exclude_ids is not None else n]


def evaluate_fusion(scores: dict[str, np.ndarray], weights: dict[str, float],
                    labels: np.ndarray, query_ids: np.ndarray, norm: str = DEFAULT_NORM,
                    ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    fused = fuse(scores, weights, norm)
    ranked = rank_from_scores(fused, exclude_ids=query_ids)
    return evaluate(ranked, labels[query_ids], labels, ks=ks)


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def grid_weights(names: list[str] | tuple[str, ...], step: float = 0.1):
    """Every weight dict over `names` with weights in multiples of `step` summing to 1."""
    n = int(round(1.0 / step))
    if abs(n * step - 1.0) > 1e-9:
        raise ValueError("step must divide 1 (0.05, 0.1, 0.25, ...)")
    for comp in _compositions(n, len(names)):
        yield {nm: c / n for nm, c in zip(names, comp)}


def tune(scores: dict[str, np.ndarray], labels: np.ndarray, query_ids: np.ndarray,
         names: list[str] | tuple[str, ...], step: float = 0.1,
         norms: tuple[str, ...] = NORMS, ks: tuple[int, ...] = (5, 10),
         progress=None) -> pd.DataFrame:
    """Grid search over weights (and normalisation). Returns one row per setting,
    sorted best first by mAP (ties: P@10). Call it with VALIDATION queries only."""
    combos = list(grid_weights(names, step))
    rows = []
    for norm in norms:
        normed = {k: normalize_scores(scores[k], norm) for k in names}
        for w in (progress(combos, desc=f"norm={norm}") if progress else combos):
            m = evaluate_fusion(normed, w, labels, query_ids, norm="none", ks=ks)
            rows.append({"norm": norm, **{f"w_{k}": w[k] for k in names}, **m})
    df = pd.DataFrame(rows)
    return df.sort_values(["mAP", "P@10"], ascending=False).reset_index(drop=True)


def save_weights(path: Path, weights: dict[str, float], norm: str, **meta) -> None:
    Path(path).write_text(json.dumps({"weights": weights, "norm": norm, **meta}, indent=2),
                          encoding="utf-8")


def load_weights(path: Path) -> tuple[dict[str, float], str]:
    """Tuned (weights, norm) from `path`, or the defaults if the file is missing/invalid."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        w = {k: float(v) for k, v in d["weights"].items()}
        if d.get("norm", DEFAULT_NORM) in NORMS and sum(w.values()) > 0:
            return w, d.get("norm", DEFAULT_NORM)
    except (OSError, ValueError, KeyError):
        pass
    return dict(DEFAULT_WEIGHTS), DEFAULT_NORM
