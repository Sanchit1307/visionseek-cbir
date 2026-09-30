"""Tune the fixed hybrid on the VALIDATION split, report on the TEST queries (Person B, WP9).

Run from the repo root after scripts\\build_index.py (needs the full CLIP index):
    python scripts\\tune_fusion.py                # CLIP vs classical_concat, step 0.05
    python scripts\\tune_fusion.py --full-grid    # optional: clip + 4 single descriptors, step 0.1

Protocol: tuning queries = all 1,020 Flowers-102 val images (global ids 1020..2039),
gallery = all 8,189 images, own image excluded, objective = mAP. The chosen
(weights, norm) are then evaluated ONCE on the 500 saved test queries
(index\\query_ids.npy), the same queries as every other table. Nothing is tuned on them.

Writes:
    results\\fusion_tuning.csv       every grid point with its val P@5 / P@10 / mAP
    results\\fusion_weights.json     best weights + norm (read by the app and by Person A)
    results\\fusion_comparison.csv   test metrics: CLIP, classical, fixed 0.7/0.3, tuned hybrid
(--full-grid writes fusion_tuning_full.csv / fusion_weights_full.json instead.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import INDEX_DIR, RESULTS_DIR  # noqa: E402
from src.features.classical import FEATURE_NAMES  # noqa: E402
from src.fusion import (DEFAULT_NORM, DEFAULT_WEIGHTS, evaluate_fusion,  # noqa: E402
                        save_weights, tune)
from src.retrieval import concat_classical, vector_file  # noqa: E402

VAL_RANGE = (1020, 2040)     # Flowers-102 official split sizes: train 1020, val 1020, test 6149
N_FLOWERS = 8189


def load_vectors(index_dir: Path, names: list[str]) -> dict[str, np.ndarray]:
    vecs = {}
    for n in names:
        p = index_dir / vector_file(n)
        if n == "classical_concat" and not p.exists():
            vecs[n] = concat_classical({m: np.load(index_dir / vector_file(m)) for m in FEATURE_NAMES})
            continue
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run scripts\\build_index.py first.")
        vecs[n] = np.load(p)
    return vecs


def sims_for(vecs: dict[str, np.ndarray], qids: np.ndarray) -> dict[str, np.ndarray]:
    return {n: v[qids] @ v.T for n, v in vecs.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ap.add_argument("--full-grid", action="store_true")
    ap.add_argument("--step", type=float, default=None,
                    help="weight grid step (default 0.05, or 0.1 with --full-grid)")
    ap.add_argument("--val-start", type=int, default=VAL_RANGE[0])
    ap.add_argument("--val-end", type=int, default=VAL_RANGE[1])
    args = ap.parse_args(argv)

    names = (["clip", *FEATURE_NAMES] if args.full_grid else ["clip", "classical_concat"])
    step = args.step or (0.1 if args.full_grid else 0.05)
    suffix = "_full" if args.full_grid else ""
    vecs = load_vectors(args.index_dir, sorted(set(names) | {"classical_concat"}))
    labels = np.load(args.index_dir / "labels.npy")
    test_ids = np.load(args.index_dir / "query_ids.npy")
    n = len(labels)
    if n != N_FLOWERS and (args.val_start, args.val_end) == VAL_RANGE:
        raise SystemExit(f"Gallery has {n} images, not the Flowers-102 {N_FLOWERS}; "
                         "pass --val-start/--val-end explicitly.")
    val_ids = np.arange(args.val_start, args.val_end)
    assert not set(val_ids) & set(test_ids.tolist()), "val and test queries must not overlap"
    print(f"Tuning on {len(val_ids)} val queries; grid step {step}; retrievers {names}")

    df = tune(sims_for({k: vecs[k] for k in names}, val_ids), labels, val_ids, names,
              step=step, progress=lambda it, desc: tqdm(it, desc=desc))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(RESULTS_DIR / f"fusion_tuning{suffix}.csv", index=False)
    best = df.iloc[0]
    w = {k: float(best[f"w_{k}"]) for k in names}
    save_weights(RESULTS_DIR / f"fusion_weights{suffix}.json", w, str(best["norm"]),
                 tuned_on=f"val ids {args.val_start}..{args.val_end - 1}",
                 objective="mAP", val_mAP=round(float(best["mAP"]), 4))
    print("\nTop 5 on validation:")
    print(df.head(5).round(4).to_string(index=False))
    print(f"\nBest: {w}  norm={best['norm']}  (val mAP {best['mAP']:.4f})")

    # ---- report on the test queries (once)
    test_scores = sims_for(vecs, test_ids)
    rows = []
    for label, wts, norm in (
        ("clip only", {"clip": 1.0}, DEFAULT_NORM),
        ("classical_concat only", {"classical_concat": 1.0}, DEFAULT_NORM),
        ("fixed hybrid 0.7 CLIP + 0.3 classical", DEFAULT_WEIGHTS, DEFAULT_NORM),
        ("tuned hybrid (val)", w, str(best["norm"])),
    ):
        m = evaluate_fusion(test_scores, wts, labels, test_ids, norm=norm)
        rows.append({"method": label, "weights": str({k: v for k, v in wts.items() if v}),
                     "norm": norm, **{k: round(v, 4) for k, v in m.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"fusion_comparison{suffix}.csv", index=False)
    print(f"\nTest queries ({len(test_ids)}):")
    print(out.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
