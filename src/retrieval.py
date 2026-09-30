"""Search engine tying features, indexes and encoders together (Person B).

Used by app/app.py and the evaluation scripts. One FAISS index per retriever;
the file layout in index/ is:

    clip_image.npy / .faiss          CLIP image embeddings        (N, 512)
    classical_<name>.npy / .faiss    colour / texture / edge / dct
    classical_concat.npy / .faiss    L2-normalised concat of the four
    labels.npy, query_ids.npy        class labels, fixed query sample

Image convention: query images are BGR uint8 (OpenCV); they are converted to RGB
only at the CLIP boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import index as ix
from src.dataset import INDEX_DIR, bgr_to_rgb
from src.features import clip_encoder
from src.explain import EXPLAIN_NAMES
from src.features.classical import FEATURE_NAMES, extract_classical
from src.fusion import DEFAULT_NORM, DEFAULT_WEIGHTS, fuse

RETRIEVERS = ("clip", "classical_concat") + tuple(FEATURE_NAMES)


def vector_file(name: str) -> str:
    if name == "clip":
        return "clip_image.npy"
    if name == "classical_concat":
        return "classical_concat.npy"
    return f"classical_{name}.npy"


def index_file(name: str) -> str:
    return vector_file(name).replace(".npy", ".faiss")


def l2_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def concat_classical(feats: dict[str, np.ndarray]) -> np.ndarray:
    """Same construction as scripts/baseline_classical.py (`classical_concat`)."""
    return l2_rows(np.hstack([feats[n] for n in FEATURE_NAMES]))


def clip_query_encoder(images_bgr: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Query encoder for scripts/run_experiments.py: BGR images -> {'clip': (Q, 512)}.

    Plug in with:  run_experiments(..., encoders=(classical_encoder, clip_query_encoder))
    and add gallery['clip'] = np.load(index/clip_image.npy) in load_gallery().
    """
    rgb = [bgr_to_rgb(im) for im in images_bgr]
    return {"clip": clip_encoder.encode_images(rgb)}


@dataclass
class SearchResult:
    retriever: str
    ids: np.ndarray          # (k,) gallery indices, best first
    scores: np.ndarray       # (k,) similarity (cosine)
    extract_ms: float        # query feature extraction time
    search_ms: float         # index search time

    @property
    def total_ms(self) -> float:
        return self.extract_ms + self.search_ms


@dataclass
class FusedResult(SearchResult):
    """Hybrid search result. `scores` are fused (normalised) scores; `parts` holds
    the raw cosine of each retriever for the returned ids (for the explanation panel)."""
    parts: dict = field(default_factory=dict)      # retriever -> (k,) raw cosine
    weights: dict = field(default_factory=dict)    # weights actually used (sum to 1)
    norm: str = DEFAULT_NORM


class SearchEngine:
    """Loads every available vector file + index from `index_dir`."""

    def __init__(self, index_dir: Path = INDEX_DIR, clip=None) -> None:
        self.index_dir = Path(index_dir)
        self.clip = clip if clip is not None else clip_encoder  # injectable for tests
        self.vectors: dict[str, np.ndarray] = {}
        self.indexes: dict[str, object] = {}
        for name in RETRIEVERS:
            vp = self.index_dir / vector_file(name)
            if vp.exists():
                self.vectors[name] = np.load(vp)
        # classical_concat is derivable even if its file was never written
        if ("classical_concat" not in self.vectors
                and all(n in self.vectors for n in FEATURE_NAMES)):
            self.vectors["classical_concat"] = concat_classical(
                {n: self.vectors[n] for n in FEATURE_NAMES})
        for name, vecs in self.vectors.items():
            self.indexes[name] = ix.load_index(self.index_dir / index_file(name), vecs)
        lp = self.index_dir / "labels.npy"
        self.labels = np.load(lp) if lp.exists() else None

    # ------------------------------------------------------------------ info
    @property
    def available(self) -> list[str]:
        return [n for n in RETRIEVERS if n in self.indexes]

    @property
    def n_gallery(self) -> int:
        return next(iter(self.vectors.values())).shape[0] if self.vectors else 0

    # ---------------------------------------------------------------- queries
    def query_vector(self, img_bgr: np.ndarray, retriever: str) -> tuple[np.ndarray, float]:
        """(1, D) query vector for a retriever, and extraction time in ms."""
        t0 = time.perf_counter()
        if retriever == "clip":
            q = self.clip.encode_images([bgr_to_rgb(img_bgr)])
        else:
            f = extract_classical(img_bgr)
            q = (concat_classical({n: f[n][None] for n in FEATURE_NAMES})
                 if retriever == "classical_concat" else f[retriever][None])
        return np.asarray(q, dtype=np.float32), (time.perf_counter() - t0) * 1000.0

    def _search(self, retriever: str, q: np.ndarray, k: int, extract_ms: float,
                exclude_id: int | None) -> SearchResult:
        if retriever not in self.indexes:
            raise KeyError(f"retriever '{retriever}' has no index in {self.index_dir}")
        kk = k + 1 if exclude_id is not None else k
        t0 = time.perf_counter()
        scores, ids = ix.search(self.indexes[retriever], q, kk)
        search_ms = (time.perf_counter() - t0) * 1000.0
        scores, ids = scores[0], ids[0]
        if exclude_id is not None:                      # drop the query itself
            keep = ids != exclude_id
            scores, ids = scores[keep][:k], ids[keep][:k]
        return SearchResult(retriever, ids, scores, extract_ms, search_ms)

    def search_image(self, img_bgr: np.ndarray, retriever: str = "clip", k: int = 10,
                     exclude_id: int | None = None) -> SearchResult:
        q, ms = self.query_vector(img_bgr, retriever)
        return self._search(retriever, q, k, ms, exclude_id)

    def search_text(self, text: str, k: int = 10) -> SearchResult:
        """Text -> image: CLIP text embedding against the CLIP image index only."""
        t0 = time.perf_counter()
        q = np.asarray(self.clip.encode_text([text]), dtype=np.float32)
        return self._search("clip", q, k, (time.perf_counter() - t0) * 1000.0, None)

    def search_image_text(self, img_bgr: np.ndarray, text: str, k: int = 10,
                          w_img: float = 0.5, exclude_id: int | None = None) -> SearchResult:
        """Image + text: q = normalize(w_img * clip_img + (1 - w_img) * clip_txt)."""
        t0 = time.perf_counter()
        qi = self.clip.encode_images([bgr_to_rgb(img_bgr)])
        qt = self.clip.encode_text([text])
        q = l2_rows(w_img * np.asarray(qi) + (1.0 - w_img) * np.asarray(qt))
        return self._search("clip", q, k, (time.perf_counter() - t0) * 1000.0, exclude_id)

    def _query_vectors(self, img_bgr: np.ndarray, names: list[str]) -> tuple[dict, float]:
        """{retriever: (1, D)} for several retrievers; the classical descriptors are
        extracted once. Also returns the extraction time in ms."""
        t0 = time.perf_counter()
        qvecs: dict[str, np.ndarray] = {}
        if "clip" in names:
            qvecs["clip"] = self.query_vector(img_bgr, "clip")[0]
        classical = [n for n in names if n != "clip"]
        if classical:
            f = extract_classical(img_bgr)
            for n in classical:
                qvecs[n] = (concat_classical({m: f[m][None] for m in FEATURE_NAMES})
                            if n == "classical_concat" else f[n][None])
        return qvecs, (time.perf_counter() - t0) * 1000.0

    def all_similarities(self, img_bgr: np.ndarray,
                         names: tuple[str, ...] = EXPLAIN_NAMES + ("classical_concat",)) -> dict:
        """{retriever: (N,) cosine of the query to the whole gallery} (for src.explain)."""
        names = [n for n in names if n in self.vectors]
        qvecs, _ = self._query_vectors(img_bgr, names)
        return {n: (np.asarray(qvecs[n], np.float32) @ self.vectors[n].T)[0] for n in names}

    def search_fused(self, img_bgr: np.ndarray, weights: dict[str, float] | None = None,
                     k: int = 10, norm: str | None = None,
                     exclude_id: int | None = None) -> FusedResult:
        """Hybrid image search: per-retriever cosine to the whole gallery, normalised,
        weighted sum (src.fusion.fuse), top-k. Exact, no candidate truncation."""
        weights = dict(weights or DEFAULT_WEIGHTS)
        norm = norm or DEFAULT_NORM
        names = [n for n, w in weights.items() if w > 0]
        missing = [n for n in names if n not in self.vectors]
        if missing:
            raise KeyError(f"retriever(s) {missing} have no vectors in {self.index_dir}")
        total = sum(weights[n] for n in names)

        qvecs, extract_ms = self._query_vectors(img_bgr, names)

        t0 = time.perf_counter()
        sims = {n: (np.asarray(qvecs[n], np.float32) @ self.vectors[n].T)[0] for n in names}
        fused = fuse(sims, weights, norm)
        if exclude_id is not None:
            fused[exclude_id] = -np.inf
        k = min(k, len(fused) - (1 if exclude_id is not None else 0))
        top = np.argpartition(-fused, k - 1)[:k]
        top = top[np.argsort(-fused[top], kind="stable")]
        search_ms = (time.perf_counter() - t0) * 1000.0
        return FusedResult("hybrid", top, fused[top], extract_ms, search_ms,
                           parts={n: sims[n][top] for n in names},
                           weights={n: weights[n] / total for n in names}, norm=norm)
