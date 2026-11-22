"""Qualitative figures for the report and slides: query + top-k results per retriever (Person B).

Run from the repo root after scripts/build_index.py (Flowers-102 or any gallery built by it):
    python scripts/make_qualitative.py                              # 2 queries, clean + noise 50 + blur 15
    python scripts/make_qualitative.py --n 3 --k 6 --seed 11
    python scripts/make_qualitative.py --conditions clean,jpeg:10 --retrievers color,clip,hybrid

One PNG per condition: results/qualitative_<condition>.png. For every query there is one row
per retriever: the (degraded) query on the left, then its top-k results. A green frame means
the result has the same class as the query, a red frame means a different class; each row is
labelled with its precision@k. "hybrid" is the tuned CLIP + classical fusion
(results/fusion_weights.json, or 0.7 / 0.3 if that file is missing). The queries are drawn from
the saved test queries (index/query_ids.npy) with the given seed; the query itself is excluded
from its own results, the same protocol as every other table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import INDEX_DIR, RESULTS_DIR, bgr_to_rgb, open_gallery  # noqa: E402
from src.degrade import degrade  # noqa: E402
from src.fusion import load_weights  # noqa: E402
from src.retrieval import SearchEngine  # noqa: E402

NOISE_SEED = 1234
GREEN, RED = "#2e9e44", "#d03b3b"


def parse_condition(text: str) -> tuple[str, int | None]:
    """'clean' | 'noise:50' | 'blur:15' | 'jpeg:10' -> (kind, level)."""
    if text == "clean":
        return "clean", None
    kind, _, level = text.partition(":")
    if kind not in ("noise", "blur", "jpeg") or not level.isdigit():
        raise ValueError(f"bad condition '{text}': use clean, noise:<sigma>, blur:<odd kernel> or jpeg:<quality>")
    return kind, int(level)


def condition_slug(text: str) -> str:
    return text.replace(":", "")


def search(engine: SearchEngine, retriever: str, img: np.ndarray, k: int, exclude_id: int):
    if retriever == "hybrid":
        w, norm = load_weights(RESULTS_DIR / "fusion_weights.json")
        return engine.search_fused(img, w, k, norm, exclude_id=exclude_id)
    return engine.search_image(img, retriever, k, exclude_id=exclude_id)


def make_figure(engine, gallery, labels, qids, condition: str, retrievers, k: int, path: Path) -> Path:
    kind, level = parse_condition(condition)
    n_rows = len(qids) * len(retrievers)
    fig, axes = plt.subplots(n_rows, k + 1, figsize=(1.75 * (k + 1), 1.75 * n_rows), squeeze=False)
    row = 0
    for qid in qids:
        img = gallery.get(int(qid))[0]
        if kind != "clean":
            img = degrade(img, kind, level, seed=NOISE_SEED + int(qid))
        for name in retrievers:
            res = search(engine, name, img, k, int(qid))
            rel = labels[res.ids] == labels[int(qid)]
            ax = axes[row, 0]
            ax.imshow(bgr_to_rgb(img))
            ax.set_title(f"query #{int(qid)}" if name == retrievers[0] else "", fontsize=8)
            ax.set_ylabel(f"{name}\nP@{k} = {rel.mean():.2f}", fontsize=8)
            for j, (g, ok) in enumerate(zip(res.ids, rel), start=1):
                a = axes[row, j]
                a.imshow(bgr_to_rgb(gallery.get(int(g))[0]))
                for s in a.spines.values():
                    s.set_edgecolor(GREEN if ok else RED)
                    s.set_linewidth(3)
                if row == 0:
                    a.set_title(f"top {j}", fontsize=8)
            for a in axes[row]:
                a.set_xticks([])
                a.set_yticks([])
            row += 1
    cond = "clean queries" if kind == "clean" else f"{kind} {level} on the query"
    fig.suptitle(f"Top-{k} results, {cond} (green = same class, red = different class)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="flowers102")
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ap.add_argument("--n", type=int, default=2, help="number of query images")
    ap.add_argument("--k", type=int, default=5, help="results shown per retriever")
    ap.add_argument("--seed", type=int, default=7, help="which test queries are drawn")
    ap.add_argument("--conditions", default="clean,noise:50,blur:15")
    ap.add_argument("--retrievers", default="color,clip,hybrid")
    args = ap.parse_args(argv)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        parse_condition(c)                                    # fail early on a typo
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    engine = SearchEngine(args.index_dir)
    wanted = [r for r in retrievers if r != "hybrid"] + (["clip", "classical_concat"] if "hybrid" in retrievers else [])
    missing = sorted({r for r in wanted if r not in engine.available})
    if missing:
        raise SystemExit(f"Missing retriever(s) {missing} in {args.index_dir}. Run scripts/build_index.py first.")
    labels = engine.labels
    pool = np.load(args.index_dir / "query_ids.npy") if (args.index_dir / "query_ids.npy").exists() \
        else np.arange(engine.n_gallery)
    qids = np.sort(np.random.default_rng(args.seed).choice(pool, size=min(args.n, len(pool)), replace=False))
    gallery = open_gallery(args.dataset)
    print(f"Queries {qids.tolist()} | retrievers {retrievers} | k = {args.k}")
    for c in conditions:
        out = RESULTS_DIR / f"qualitative_{condition_slug(c)}.png"
        make_figure(engine, gallery, labels, qids, c, retrievers, args.k, out)
        print("saved", out)


if __name__ == "__main__":
    main()
