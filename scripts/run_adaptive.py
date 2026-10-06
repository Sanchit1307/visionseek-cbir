"""Adaptive vs fixed hybrid fusion (Person A, WP10).

Run from the repo root (needs index\\clip_image.npy, results\\fusion_weights.json and
src\\quality_thresholds.json, i.e. build_index, tune_fusion and calibrate_quality done):
    python scripts\\run_adaptive.py
    python scripts\\run_adaptive.py --limit 20            # smoke test (no cache, nothing saved)
    python scripts\\run_adaptive.py --val-n 500           # faster: 500 of the 1,020 val images

1. Features: every query is degraded under CONDITIONS, analysed (bucket predicted by
   src.quality), and encoded (CLIP + classical_concat). Cached in index\\adaptive_{val,test}.npz.
2. Tuning (VAL images only, ids 1020..2039): for each predicted bucket, grid-search the
   CLIP / classical weights (same norm as Person B's tuned hybrid). A bucket keeps the
   fixed weights unless its best weights beat them by >= --min-gain val mAP.
3. Report (the 500 fixed TEST queries, once): clip only, classical only, fixed hybrid,
   adaptive (predicted bucket) and adaptive-oracle (true condition's bucket) per condition.

Outputs: results\\adaptive_weights.json, adaptive_val_tuning.csv, adaptive_val_buckets.csv,
adaptive_comparison.csv, adaptive_summary.csv, adaptive_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_classical as bc  # noqa: E402
import run_experiments as rx  # noqa: E402
from calibrate_quality import apply_condition  # noqa: E402
from src import quality  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.fusion import (evaluate_fusion, fuse, load_weights, rank_from_scores,  # noqa: E402
                        tune)

NAMES = ["clip", "classical_concat"]
KS = (5, 10)
NOISE_SEED = rx.NOISE_SEED          # same seeding as run_experiments / calibrate_quality
BLOCK = 128                          # images degraded + encoded at a time (memory)
VAL_RANGE = (1020, 2040)

CONDITIONS = [
    ("clean", None, 0),
    ("noise_s10", "noise", 10), ("noise_s25", "noise", 25), ("noise_s50", "noise", 50),
    ("blur_k5", "blur", 5), ("blur_k9", "blur", 9), ("blur_k15", "blur", 15),
    ("jpeg_q30", "jpeg", 30), ("jpeg_q10", "jpeg", 10),
    ("dark_x0.6", "dark", 0.6), ("dark_x0.4", "dark", 0.4),
    ("lowcon_x0.6", "lowcon", 0.6), ("lowcon_x0.4", "lowcon", 0.4),
]
COND_ORDER = [c[0] for c in CONDITIONS]
EXPECTED = {"clean": "clean", "noise_s10": "noisy", "noise_s25": "noisy", "noise_s50": "noisy",
            "blur_k5": "blurred", "blur_k9": "blurred", "blur_k15": "blurred",
            "jpeg_q30": "compressed", "jpeg_q10": "compressed",
            "dark_x0.6": "dark/low-contrast", "dark_x0.4": "dark/low-contrast",
            "lowcon_x0.6": "dark/low-contrast", "lowcon_x0.4": "dark/low-contrast"}
METHODS = ["clip only", "classical only", "fixed hybrid", "adaptive", "adaptive (oracle bucket)"]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def encode(imgs: list[np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(rx.classical_encoder(imgs))
    out.update(rx.clip_query_encoder(imgs))
    return out


def build_features(ds, ids: np.ndarray, cache: Path | None, th: dict, rebuild: bool) -> dict:
    if cache is not None and cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=False)
        if np.array_equal(z["ids"], ids) and list(z["conds"]) == COND_ORDER:
            print(f"Reusing cached features: {cache.name}")
            return {k: z[k] for k in z.files}
        print(f"{cache.name} does not match the requested queries/conditions, rebuilding.")

    cond, qid, bucket, clip, cls = [], [], [], [], []
    for name, kind, level in CONDITIONS:
        for s in tqdm(range(0, len(ids), BLOCK), desc=f"{name:12s}"):
            block = ids[s:s + BLOCK]
            imgs = []
            for g in block:
                im = apply_condition(ds.get(int(g))[0], kind, level, NOISE_SEED + int(g))
                imgs.append(im)
                bucket.append(quality.analyze(im, th)["bucket"])
            enc = encode(imgs)
            clip.append(enc["clip"])
            cls.append(enc["classical_concat"])
            cond += [name] * len(block)
            qid += [int(g) for g in block]
    data = {"ids": ids, "conds": np.array(COND_ORDER), "cond": np.array(cond),
            "qid": np.array(qid, dtype=np.int64), "bucket": np.array(bucket),
            "clip": np.concatenate(clip).astype(np.float32),
            "cls": np.concatenate(cls).astype(np.float32)}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **data)
    return data


# ---------------------------------------------------------------------------
# Tuning on val
# ---------------------------------------------------------------------------
def fit_table(val: dict, gal: dict, labels: np.ndarray, fixed_w: dict, norm: str, args):
    rng = np.random.default_rng(7)
    table, meta, tuning = {}, {}, []
    for b in quality.BUCKETS:
        idx = np.flatnonzero(val["bucket"] == b)
        entry = {"n_val_total": int(len(idx))}
        if len(idx) < args.min_bucket_n:
            table[b] = dict(fixed_w)
            entry.update(used="fixed (too few val queries)", weights=dict(fixed_w))
            meta[b] = entry
            print(f"  {b:18s} n={len(idx):5d}  -> fixed weights (too few queries)")
            continue
        if len(idx) > args.max_per_bucket:
            idx = np.sort(rng.choice(idx, args.max_per_bucket, replace=False))
        qids = val["qid"][idx]
        scores = {"clip": val["clip"][idx] @ gal["clip"].T,
                  "classical_concat": val["cls"][idx] @ gal["classical_concat"].T}
        df = tune(scores, labels, qids, NAMES, step=args.step, norms=(norm,))
        best = df.iloc[0]
        w_best = {k: float(best[f"w_{k}"]) for k in NAMES}
        m_fixed = evaluate_fusion(scores, fixed_w, labels, qids, norm=norm)
        gain = float(best["mAP"]) - m_fixed["mAP"]
        adopt = gain >= args.min_gain
        table[b] = w_best if adopt else dict(fixed_w)
        entry.update(n_val_used=int(len(idx)), weights=table[b], best_grid_weights=w_best,
                     val_mAP_best=round(float(best["mAP"]), 4),
                     val_mAP_fixed=round(m_fixed["mAP"], 4), val_gain=round(gain, 4),
                     used="tuned" if adopt else f"fixed (gain < {args.min_gain})")
        meta[b] = entry
        df.insert(0, "bucket", b)
        tuning.append(df.round(4))
        print(f"  {b:18s} n={len(idx):5d}  best clip={w_best['clip']:.2f}  "
              f"val mAP {best['mAP']:.4f} vs fixed {m_fixed['mAP']:.4f} ({gain:+.4f})  -> {entry['used']}")
    return table, meta, (pd.concat(tuning, ignore_index=True) if tuning else pd.DataFrame())


# ---------------------------------------------------------------------------
# Evaluation on test
# ---------------------------------------------------------------------------
def eval_grouped(scores, qids, keys, weight_of, norm, labels):
    n = labels.shape[0]
    ranked = np.empty((len(qids), n - 1), dtype=np.int64)
    for key in np.unique(keys):
        m = keys == key
        fused = fuse({k: v[m] for k, v in scores.items()}, weight_of(key), norm)
        ranked[m] = rank_from_scores(fused, exclude_ids=qids[m])
    return evaluate(ranked, labels[qids], labels, ks=KS)


def evaluate_test(test: dict, gal: dict, labels: np.ndarray, table: dict, fixed_w: dict,
                  norm: str) -> pd.DataFrame:
    rows = []
    for cond in COND_ORDER:
        idx = np.flatnonzero(test["cond"] == cond)
        qids = test["qid"][idx]
        scores = {"clip": test["clip"][idx] @ gal["clip"].T,
                  "classical_concat": test["cls"][idx] @ gal["classical_concat"].T}
        one = np.zeros(len(idx), dtype="<U1")
        pred = test["bucket"][idx]
        oracle = np.full(len(idx), EXPECTED[cond])
        runs = {
            "clip only": (one, lambda _: {"clip": 1.0}),
            "classical only": (one, lambda _: {"classical_concat": 1.0}),
            "fixed hybrid": (one, lambda _: fixed_w),
            "adaptive": (pred, lambda b: table[b]),
            "adaptive (oracle bucket)": (oracle, lambda b: table[b]),
        }
        for method, (keys, wf) in runs.items():
            m = eval_grouped(scores, qids, keys, wf, norm, labels)
            rows.append({"condition": cond, "method": method, "n_queries": len(idx),
                         **{k: round(v, 4) for k, v in m.items()}})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: Path) -> None:
    piv = df.pivot(index="condition", columns="method", values="mAP").reindex(COND_ORDER)[METHODS]
    fig, ax = plt.subplots(figsize=(15, 5))
    width = 0.8 / len(METHODS)
    x = np.arange(len(COND_ORDER))
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - (len(METHODS) - 1) / 2) * width, piv[m].values, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_ORDER, rotation=45, ha="right")
    ax.set_ylabel("mAP")
    ax.set_title("Adaptive vs fixed hybrid fusion (test queries)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=float, default=0.05, help="weight grid step")
    ap.add_argument("--val-n", type=int, default=None, help="use N of the 1,020 val images")
    ap.add_argument("--max-per-bucket", type=int, default=2000,
                    help="max val query rows used to tune one bucket")
    ap.add_argument("--min-bucket-n", type=int, default=100,
                    help="buckets with fewer val rows keep the fixed weights")
    ap.add_argument("--min-gain", type=float, default=0.002,
                    help="val mAP gain over the fixed hybrid needed to adopt tuned weights")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: N val and N test images, no cache, nothing saved")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached features")
    args = ap.parse_args(argv)

    if not quality.is_calibrated():
        print("WARNING: src\\quality_thresholds.json missing -> uncalibrated placeholder thresholds.")
    th = quality.load_thresholds()

    gallery, labels, test_ids = rx.load_gallery(with_clip=True)
    if "clip" not in gallery:
        raise SystemExit("index\\clip_image.npy is required (Person B: build_index.py).")
    fixed_w, norm = load_weights(bc.RESULTS_DIR / "fusion_weights.json")
    if set(fixed_w) != set(NAMES):
        fixed_w = {"clip": 0.7, "classical_concat": 0.3}
        print("fusion_weights.json is not the 2-way hybrid, using 0.7 / 0.3.")
    print(f"Fixed hybrid: {fixed_w}, norm={norm}")

    ds = bc.Flowers102Gallery(bc.DATA_DIR)
    val_ids = np.arange(VAL_RANGE[0], VAL_RANGE[1])
    if args.val_n:
        val_ids = np.sort(np.random.default_rng(7).choice(val_ids, args.val_n, replace=False))
    if args.limit:
        val_ids, test_ids = val_ids[:args.limit], test_ids[:args.limit]
    assert not set(val_ids.tolist()) & set(test_ids.tolist()), "val and test must not overlap"

    cache = lambda s: None if args.limit else bc.INDEX_DIR / f"adaptive_{s}.npz"  # noqa: E731
    print("\nVAL features")
    val = build_features(ds, val_ids, cache("val"), th, args.rebuild)
    print("\nTEST features")
    test = build_features(ds, test_ids, cache("test"), th, args.rebuild)

    comp = pd.crosstab(val["cond"], val["bucket"]).reindex(COND_ORDER)
    print("\nVAL: rows per (true condition, predicted bucket):")
    print(comp.to_string())

    print("\nTuning per predicted bucket (val only):")
    table, meta, tuning = fit_table(val, gallery, labels, fixed_w, norm, args)

    print("\nEvaluating on test queries...")
    df = evaluate_test(test, gallery, labels, table, fixed_w, norm)

    piv = df.pivot(index="condition", columns="method", values="mAP").reindex(COND_ORDER)[METHODS]
    deg = [c for c in COND_ORDER if c != "clean"]
    summ = pd.DataFrame({"clean": piv.loc["clean"], "mean_degraded": piv.loc[deg].mean(),
                         "mean_all": piv.mean()}).round(4)
    summ["gain_vs_fixed_degraded"] = (summ["mean_degraded"] - summ.loc["fixed hybrid", "mean_degraded"]).round(4)
    print("\nTest mAP per condition:")
    print(piv.round(4).to_string())
    print("\nSummary (mean mAP):")
    print(summ.to_string())
    print("\nNote: with 500 queries, differences below ~0.005 mAP are within noise.")

    if args.limit:
        print("\n(smoke test: nothing saved)")
        return
    out = bc.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "adaptive_weights.json").write_text(json.dumps({
        "norm": norm, "fixed_weights": fixed_w, "buckets": meta,
        "tuned_on": f"flowers102 val ids {VAL_RANGE[0]}..{VAL_RANGE[1] - 1} "
                    f"({len(val_ids)} images x {len(CONDITIONS)} conditions), objective mAP",
        "min_gain": args.min_gain, "step": args.step}, indent=2), encoding="utf-8")
    tuning.to_csv(out / "adaptive_val_tuning.csv", index=False)
    comp.to_csv(out / "adaptive_val_buckets.csv")
    df.to_csv(out / "adaptive_comparison.csv", index=False)
    summ.to_csv(out / "adaptive_summary.csv")
    plot(df, out / "adaptive_comparison.png")
    print(f"\nSaved to {out}: adaptive_weights.json, adaptive_val_tuning.csv, "
          "adaptive_val_buckets.csv, adaptive_comparison.csv, adaptive_summary.csv, adaptive_comparison.png")


if __name__ == "__main__":
    main()