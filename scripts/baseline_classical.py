"""Classical-only baseline on Oxford Flowers-102 (Person A).

Run from the repo root:
    python scripts\\baseline_classical.py

Protocol: gallery = all 8,189 images (train + val + test, in that order);
queries = fixed seeded sample of test images; the query is excluded from its own
ranking; relevant = same class. Features are cached in index\\ (gitignored) so
that Person B can reuse them; results go to results\\baseline_classical.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate, measure_search_latency, rank_by_similarity  # noqa: E402
from src.features.classical import (  # noqa: E402
    FEATURE_NAMES, color_feature, dct_feature, edge_feature,
    extract_classical, texture_feature,
)

DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "results"

SEED = 42
N_QUERIES = 500
KS = (5, 10)

SINGLE_EXTRACTORS = {
    "color": color_feature,
    "texture": texture_feature,
    "edge": edge_feature,
    "dct": dct_feature,
}


class Flowers102Gallery:
    """Flowers-102 train + val + test as one gallery of 8,189 images."""

    SPLITS = ("train", "val", "test")

    def __init__(self, root: Path) -> None:
        from torchvision.datasets import Flowers102  # lazy: needs torch

        root.mkdir(parents=True, exist_ok=True)
        self.sets = [Flowers102(root=str(root), split=s, download=True)
                     for s in self.SPLITS]
        self.offsets = np.cumsum([0] + [len(s) for s in self.sets])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def test_range(self) -> tuple[int, int]:
        return int(self.offsets[2]), int(self.offsets[3])

    def get(self, g: int) -> tuple[np.ndarray, int]:
        """Return (BGR uint8 image, class label) for global index g."""
        k = int(np.searchsorted(self.offsets, g, side="right") - 1)
        pil, label = self.sets[k][g - int(self.offsets[k])]
        bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        return bgr, int(label)


def select_query_ids(test_start: int, test_end: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(n, test_end - test_start)
    return np.sort(rng.choice(np.arange(test_start, test_end), size=n, replace=False))


def _row_l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def load_or_extract(gallery, recompute: bool) -> tuple[dict, np.ndarray]:
    paths = {name: INDEX_DIR / f"classical_{name}.npy" for name in FEATURE_NAMES}
    labels_path = INDEX_DIR / "labels.npy"
    if not recompute and labels_path.exists() and all(p.exists() for p in paths.values()):
        feats = {name: np.load(p) for name, p in paths.items()}
        labels = np.load(labels_path)
        if len(labels) == len(gallery) and all(len(v) == len(gallery) for v in feats.values()):
            print(f"Loaded cached features from {INDEX_DIR}")
            return feats, labels
        print("Cached features do not match the gallery size, recomputing.")

    n = len(gallery)
    buckets = {name: [] for name in FEATURE_NAMES}
    labels = np.empty(n, dtype=np.int64)
    for g in tqdm(range(n), desc="Extracting classical features"):
        img, label = gallery.get(g)
        f = extract_classical(img)
        for name in FEATURE_NAMES:
            buckets[name].append(f[name])
        labels[g] = label
    feats = {name: np.stack(v).astype(np.float32) for name, v in buckets.items()}

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in feats.items():
        np.save(paths[name], arr)
    np.save(labels_path, labels)
    return feats, labels


def time_extraction(images: list[np.ndarray]) -> dict[str, float]:
    """Mean per-image extraction time (ms) for each single descriptor."""
    out = {}
    for name, fn in SINGLE_EXTRACTORS.items():
        fn(images[0])  # warm-up
        t0 = time.perf_counter()
        for im in images:
            fn(im)
        out[name] = (time.perf_counter() - t0) / len(images) * 1000.0
    return out


def run(feats: dict[str, np.ndarray], labels: np.ndarray, query_ids: np.ndarray,
        extract_ms: dict[str, float]) -> pd.DataFrame:
    retrievers = dict(feats)
    retrievers["classical_concat"] = _row_l2(np.hstack([feats[n] for n in FEATURE_NAMES]))
    extract_ms = dict(extract_ms)
    extract_ms["classical_concat"] = sum(extract_ms[n] for n in FEATURE_NAMES)

    rows = []
    for name, vecs in retrievers.items():
        q = vecs[query_ids]
        ranked = rank_by_similarity(q, vecs, exclude_ids=query_ids)
        metrics = evaluate(ranked, labels[query_ids], labels, ks=KS)
        search_ms = measure_search_latency(q, vecs, k=max(KS))
        rows.append({
            "retriever": name,
            "dim": vecs.shape[1],
            **{k: round(v, 4) for k, v in metrics.items()},
            "extract_ms": round(extract_ms[name], 2),
            "search_ms": round(search_ms, 3),
            "total_ms": round(extract_ms[name] + search_ms, 2),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true", help="ignore cached features")
    ap.add_argument("--queries", type=int, default=N_QUERIES, help="number of test queries")
    ap.add_argument("--latency-samples", type=int, default=100,
                    help="query images used to time feature extraction")
    args = ap.parse_args(argv)

    gallery = Flowers102Gallery(DATA_DIR)
    feats, labels = load_or_extract(gallery, args.recompute)

    t_start, t_end = gallery.test_range()
    query_ids = select_query_ids(t_start, t_end, args.queries, SEED)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "query_ids.npy", query_ids)

    timing_imgs = [gallery.get(int(g))[0] for g in query_ids[:args.latency_samples]]
    extract_ms = time_extraction(timing_imgs)

    df = run(feats, labels, query_ids, extract_ms)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "baseline_classical.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nGallery: {len(gallery)} images | classes: {len(np.unique(labels))} | "
          f"queries: {len(query_ids)} (seed {SEED}, self-match excluded)\n")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()