"""CLIP baseline: same protocol and columns as scripts/baseline_classical.py (Person B).

Run from the repo root (after scripts\\build_index.py):
    python scripts\\eval_clip.py

Writes results\\baseline_clip.csv (one row, retriever = clip) and prints a
zero-shot text-search sanity check on Flowers-102. Protocol: gallery = all
images, queries = the saved 500-query sample (index\\query_ids.npy), self-match
excluded, relevant = same class.

The zero-shot check encodes "a photo of a <flower>, a type of flower." for the
102 class names and classifies every test image. CLIP ViT-B/32 reaches roughly
60-70% on Flowers-102; a much lower number means either the model did not load
its pretrained weights or the class-name order does not match the labels.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import (FLOWERS102_CLASSES, INDEX_DIR, RESULTS_DIR,  # noqa: E402
                         bgr_to_rgb, open_gallery)
from src.evaluate import evaluate, measure_search_latency, rank_by_similarity  # noqa: E402
from src.features import clip_encoder  # noqa: E402

KS = (5, 10)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="flowers102")
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ap.add_argument("--latency-samples", type=int, default=50,
                    help="query images encoded one at a time to time the CLIP encoder")
    ap.add_argument("--no-zeroshot", action="store_true")
    args = ap.parse_args(argv)

    d = args.index_dir
    for f in ("clip_image.npy", "labels.npy", "query_ids.npy"):
        if not (d / f).exists():
            raise SystemExit(f"Missing {d / f}. Run scripts\\build_index.py first.")
    vecs, labels, qids = (np.load(d / "clip_image.npy"), np.load(d / "labels.npy"),
                          np.load(d / "query_ids.npy"))

    ranked = rank_by_similarity(vecs[qids], vecs, exclude_ids=qids)
    metrics = evaluate(ranked, labels[qids], labels, ks=KS)
    search_ms = measure_search_latency(vecs[qids], vecs, k=max(KS))

    gallery = open_gallery(args.dataset)
    imgs = [bgr_to_rgb(gallery.get(int(g))[0]) for g in qids[:args.latency_samples]]
    clip_encoder.encode_images(imgs[:1])                       # warm-up (model load)
    t0 = time.perf_counter()
    for im in imgs:
        clip_encoder.encode_images([im])
    extract_ms = (time.perf_counter() - t0) / len(imgs) * 1000.0

    df = pd.DataFrame([{
        "retriever": "clip", "dim": vecs.shape[1],
        **{k: round(v, 4) for k, v in metrics.items()},
        "extract_ms": round(extract_ms, 2), "search_ms": round(search_ms, 3),
        "total_ms": round(extract_ms + search_ms, 2),
    }])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baseline_clip.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {out}")

    if args.dataset == "flowers102" and not args.no_zeroshot:
        prompts = [f"a photo of a {n}, a type of flower." for n in FLOWERS102_CLASSES]
        txt = clip_encoder.encode_text(prompts)
        pred = (vecs @ txt.T).argmax(axis=1)
        acc = float((pred == labels).mean())
        print(f"\nZero-shot text->class accuracy on all {len(labels)} images: {acc:.1%}")
        if acc < 0.4:
            print("WARNING: far below the expected ~60-70%. Check that the pretrained "
                  "weights loaded (VISIONSEEK_CLIP_PRETRAINED) and the class-name order.")


if __name__ == "__main__":
    main()
