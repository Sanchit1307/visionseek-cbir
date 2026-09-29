"""Exact inner-product search: FAISS IndexFlatIP with a NumPy fallback (Person B).

Public API (interface contract):
    build_index(vectors)               -> index (faiss.IndexFlatIP)
    search(index, query_vecs, k)       -> (scores (Q,k), ids (Q,k))

Vectors must be float32 and L2-normalised, so inner product = cosine similarity.
If faiss cannot be imported (Windows wheel issues), a NumPy exact-search class
with the same interface is used; at this scale (~8k vectors) both are exact and
give identical rankings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:  # pragma: no cover - depends on the machine
    faiss = None
    HAVE_FAISS = False


class NumpyFlatIP:
    """Minimal stand-in for faiss.IndexFlatIP (exact inner-product search)."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.d = self.vectors.shape[1]

    @property
    def ntotal(self) -> int:
        return self.vectors.shape[0]

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims = q @ self.vectors.T
        k = min(k, self.ntotal)
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[rows, part], axis=1, kind="stable")
        ids = part[rows, order]
        return sims[rows, ids].astype(np.float32), ids.astype(np.int64)


def _check(v: np.ndarray, name: str) -> np.ndarray:
    v = np.ascontiguousarray(v, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"{name} must be 2-D (N, D)")
    return v


def build_index(vectors: np.ndarray):
    """Build an exact inner-product index over (N, D) float32 vectors."""
    v = _check(vectors, "vectors")
    if HAVE_FAISS:
        index = faiss.IndexFlatIP(v.shape[1])
        index.add(v)
        return index
    return NumpyFlatIP(v)


def search(index, query_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k search. Returns (scores, ids), each (Q, k), best match first."""
    q = _check(query_vecs, "query_vecs")
    k = min(int(k), index.ntotal)
    scores, ids = index.search(q, k)
    return scores, ids


def save_index(index, path: Path) -> None:
    """Write a faiss index to disk (no-op with the NumPy fallback: the .npy
    vectors are the source of truth and the index is rebuilt in milliseconds)."""
    if HAVE_FAISS and not isinstance(index, NumpyFlatIP):
        faiss.write_index(index, str(path))


def load_index(path: Path, vectors: np.ndarray | None = None):
    """Load a saved .faiss file, or rebuild from `vectors` if it is missing."""
    path = Path(path)
    if HAVE_FAISS and path.exists():
        return faiss.read_index(str(path))
    if vectors is None:
        raise FileNotFoundError(f"{path} not found and no vectors given to rebuild from")
    return build_index(vectors)
