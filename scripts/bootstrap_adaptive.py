"""Paired bootstrap: is adaptive fusion better than the fixed hybrid? (Person A, WP10)

Run from the repo root, AFTER scripts\\run_adaptive.py (it reuses the cached features
index\\adaptive_test.npz and the tuned table results\\adaptive_weights.json; nothing is
re-encoded, nothing is re-tuned):
    python scripts\\bootstrap_adaptive.py
    python scripts\\bootstrap_adaptive.py --boot 20000 --alpha 0.05

For every test query the average precision (AP) of the fixed hybrid and of the adaptive
hybrid is computed, and d = AP_adaptive - AP_fixed. mAP gain = mean(d). The bootstrap
resamples the 500 test IMAGES with replacement (the same images are used in every
condition, so for pooled groups all conditions of an image are resampled together) and
gives a percentile confidence interval for the mean gain.

Verdict per row (the only wording to use in the report):
    gain  -> the CI is entirely above 0
    loss  -> the CI is entirely below 0
    tie   -> the CI contains 0: report "no measurable difference"
Default alpha is 0.01 (99% CI) because many comparisons are made at once.

Outputs: results\\adaptive_significance.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_classical as bc  # noqa: E402
import run_adaptive as ra  # noqa: E402
import run_experiments as rx  # noqa: E402
from src.adaptive import load_table  # noqa: E402
from src.fusion import fuse, load_weights, rank_from_scores  # noqa: E402

GROUPS = {   # pooled rows (all conditions of an image resampled together)
    "ALL degraded": [c for c in ra.COND_ORDER if c != "clean"],
    "noise (s10, s25, s50)": ["noise_s10", "noise_s25", "noise_s50"],
    "blur (k5, k9, k15)": ["blur_k5", "blur_k9", "blur_k15"],
    "jpeg (q30, q10)": ["jpeg_q30", "jpeg_q10"],
    "dark/low-contrast (4 conds)": ["dark_x0.6", "dark_x0.4", "lowcon_x0.6", "lowcon_x0.4"],
}


def per_query_ap(ranked: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    """AP of every query over its full ranking (same definition as src.evaluate.evaluate)."""
    rel = (gallery_labels[ranked] == query_labels[:, None]).astype(np.float64)
    prec = np.cumsum(rel, axis=1) / np.arange(1, rel.shape[1] + 1)
    n_rel = rel.sum(axis=1)
    return np.where(n_rel > 0, (prec * rel).sum(axis=1) / np.maximum(n_rel, 1.0), 0.0)


def paired_bootstrap(d: np.ndarray, n_boot: int, alpha: float, seed: int = 0) -> tuple[float, float, float, str]:
    """d: (n_images,) per-image mean difference. Returns (mean, ci_low, ci_high, verdict)."""
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    means = np.empty(n_boot)
    step = 2000
    for s in range(0, n_boot, step):
        m = min(step, n_boot - s)
        idx = rng.integers(0, n, size=(m, n))
        means[s:s + m] = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    verdict = "gain" if lo > 0 else "loss" if hi < 0 else "tie"
    return float(d.mean()), float(lo), float(hi), verdict


def ap_by_condition(test: dict, gal: dict, labels: np.ndarray, table: dict, fixed_w: dict,
                    norm: str) -> tuple[dict, dict, np.ndarray]:
    """Per-image AP of the fixed and the adaptive hybrid for every condition.
    Returns ({cond: AP_fixed}, {cond: AP_adaptive}, sorted unique image ids)."""
    img_ids = np.unique(test["qid"])
    fixed_ap, adapt_ap = {}, {}
    for cond in ra.COND_ORDER:
        idx = np.flatnonzero(test["cond"] == cond)
        order = np.argsort(test["qid"][idx], kind="stable")
        idx = idx[order]
        qids = test["qid"][idx]
        assert np.array_equal(qids, img_ids), f"{cond}: queries differ from the other conditions"
        scores = {"clip": test["clip"][idx] @ gal["clip"].T,
                  "classical_concat": test["cls"][idx] @ gal["classical_concat"].T}
        ranked = rank_from_scores(fuse(scores, fixed_w, norm), exclude_ids=qids)
        fixed_ap[cond] = per_query_ap(ranked, labels[qids], labels)

        pred = test["bucket"][idx]
        ranked_a = np.empty_like(ranked)
        for b in np.unique(pred):
            m = pred == b
            fused = fuse({k: v[m] for k, v in scores.items()}, table[b], norm)
            ranked_a[m] = rank_from_scores(fused, exclude_ids=qids[m])
        adapt_ap[cond] = per_query_ap(ranked_a, labels[qids], labels)
    return fixed_ap, adapt_ap, img_ids


def significance_table(fixed_ap: dict, adapt_ap: dict, n_boot: int, alpha: float) -> pd.DataFrame:
    rows = []
    entries = [(c, [c]) for c in ra.COND_ORDER] + list(GROUPS.items())
    for name, conds in entries:
        d = np.mean([adapt_ap[c] - fixed_ap[c] for c in conds], axis=0)
        mean, lo, hi, verdict = paired_bootstrap(d, n_boot, alpha)
        rows.append({"row": name, "kind": "condition" if len(conds) == 1 else "pooled",
                     "mAP_fixed": round(float(np.mean([fixed_ap[c].mean() for c in conds])), 4),
                     "mAP_adaptive": round(float(np.mean([adapt_ap[c].mean() for c in conds])), 4),
                     "gain": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "verdict": verdict})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--alpha", type=float, default=0.01, help="1 - confidence level (default 0.01)")
    args = ap.parse_args(argv)

    cache = bc.INDEX_DIR / "adaptive_test.npz"
    if not cache.exists():
        raise SystemExit(f"{cache} not found. Run scripts\\run_adaptive.py first.")
    z = np.load(cache, allow_pickle=False)
    test = {k: z[k] for k in z.files}
    if list(test["conds"]) != ra.COND_ORDER:
        raise SystemExit("Cached conditions differ from run_adaptive.py; rerun it with --rebuild.")

    gallery, labels, _ = rx.load_gallery(with_clip=True)
    if "clip" not in gallery:
        raise SystemExit("index\\clip_image.npy is required.")
    fixed_w, norm = load_weights(bc.RESULTS_DIR / "fusion_weights.json")
    table, table_norm = load_table(bc.RESULTS_DIR / "adaptive_weights.json")
    if table_norm != norm:
        raise SystemExit(f"norm mismatch: fixed hybrid '{norm}' vs adaptive table '{table_norm}'")

    fixed_ap, adapt_ap, img_ids = ap_by_condition(test, gallery, labels, table, fixed_w, norm)
    print(f"{len(img_ids)} test images x {len(ra.COND_ORDER)} conditions, "
          f"{args.boot} resamples, {(1 - args.alpha) * 100:.0f}% CI\n")
    df = significance_table(fixed_ap, adapt_ap, args.boot, args.alpha)
    out = bc.RESULTS_DIR / "adaptive_significance.csv"
    bc.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("\nReport rule: call a row a gain/loss only if its verdict says so; every 'tie' is "
          "'no measurable difference'.")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()