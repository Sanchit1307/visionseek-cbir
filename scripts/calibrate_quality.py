"""Calibrate and evaluate the image quality analyzer (Person A, WP7).

Run from the repo root (Flowers-102 must already be downloaded, see baseline script):
    python scripts\\calibrate_quality.py
    python scripts\\calibrate_quality.py --limit 50     # quick smoke test (does not save thresholds)

1. Calibration: the five metrics are measured on the CLEAN images of the Flowers-102
   VAL split; each threshold is the clean percentile that gives a false-alarm rate of
   `--fp-rate` per detector (default 2%). Thresholds go to src\\quality_thresholds.json.
2. Evaluation: on the 500 fixed TEST queries, every degradation used in the robustness
   experiment (same seeds) plus two synthetic ones (darkening, contrast reduction) is
   analysed; the bucket assigned to each image is compared with the expected one.

Outputs: results\\quality_metrics_test.csv, quality_calibration.csv (median metric per
condition), quality_confusion.csv (% of images per bucket) and quality_metrics.png.
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
from src import quality  # noqa: E402
from src.degrade import degrade  # noqa: E402

NOISE_SEED = 1234   # same seeding as scripts/run_experiments.py
BLOCKINESS_FLOOR = 1.10   # 1.0 = no blocking; values below ~1.10 are not visible blocking

CONDITIONS = [
    ("clean", None, 0),
    ("noise_s10", "noise", 10), ("noise_s25", "noise", 25), ("noise_s50", "noise", 50),
    ("blur_k5", "blur", 5), ("blur_k9", "blur", 9), ("blur_k15", "blur", 15),
    ("jpeg_q30", "jpeg", 30), ("jpeg_q10", "jpeg", 10),
    ("dark_x0.4", "dark", 0.4), ("lowcon_x0.4", "lowcon", 0.4),
]
COND_ORDER = [c[0] for c in CONDITIONS]
EXPECTED = {"clean": "clean", "noise_s10": "noisy", "noise_s25": "noisy", "noise_s50": "noisy",
            "blur_k5": "blurred", "blur_k9": "blurred", "blur_k15": "blurred",
            "jpeg_q30": "compressed", "jpeg_q10": "compressed",
            "dark_x0.4": "dark/low-contrast", "lowcon_x0.4": "dark/low-contrast"}


def apply_condition(img: np.ndarray, kind: str | None, level: float, seed: int) -> np.ndarray:
    if kind is None:
        return img
    if kind in ("noise", "blur", "jpeg"):
        return degrade(img, kind, int(level), seed=seed)
    x = img.astype(np.float32)
    if kind == "dark":                                    # global darkening
        out = x * float(level)
    elif kind == "lowcon":                                # contrast squeeze around the mean
        m = x.mean()
        out = m + (x - m) * float(level)
    else:
        raise ValueError(kind)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def calibrate(gallery, limit: int | None, fp_rate: float) -> tuple[dict, int]:
    start, end = int(gallery.offsets[1]), int(gallery.offsets[2])      # val split
    ids = np.arange(start, end)
    if limit:
        ids = ids[:limit]
    rows = [quality.compute_metrics(gallery.get(int(g))[0])
            for g in tqdm(ids, desc="Calibrating on clean val images")]
    df = pd.DataFrame(rows)
    th = {
        "noise_max": float(df["noise"].quantile(1 - fp_rate)),
        "blockiness_max": max(float(df["jpeg_blockiness"].quantile(1 - fp_rate)), BLOCKINESS_FLOOR),
        "sharpness_min": float(df["sharpness"].quantile(fp_rate)),
        "brightness_min": float(df["brightness"].quantile(fp_rate)),
        "contrast_min": float(df["contrast"].quantile(fp_rate)),
    }
    return th, len(ids)


def measure_test(gallery, query_ids: np.ndarray) -> pd.DataFrame:
    rows = []
    for qid in tqdm(query_ids, desc="Analysing test queries"):
        img = gallery.get(int(qid))[0]
        for name, kind, level in CONDITIONS:
            x = apply_condition(img, kind, level, NOISE_SEED + int(qid))
            rows.append({"condition": name, "query_id": int(qid), **quality.compute_metrics(x)})
    return pd.DataFrame(rows)


def is_monotonic(values: list[float], increasing: bool) -> bool:
    d = np.diff(values)
    return bool(np.all(d > 0)) if increasing else bool(np.all(d < 0))


def ordering_checks(med: pd.DataFrame) -> list[tuple[str, bool]]:
    def seq(metric: str, conds: list[str]) -> list[float]:
        return [float(med.loc[c, metric]) for c in conds]
    return [
        ("noise estimate rises with sigma (clean, 10, 25, 50)",
         is_monotonic(seq("noise", ["clean", "noise_s10", "noise_s25", "noise_s50"]), True)),
        ("sharpness falls with blur (clean, k5, k9, k15)",
         is_monotonic(seq("sharpness", ["clean", "blur_k5", "blur_k9", "blur_k15"]), False)),
        ("blockiness rises with JPEG damage (clean, q30, q10)",
         is_monotonic(seq("jpeg_blockiness", ["clean", "jpeg_q30", "jpeg_q10"]), True)),
        ("brightness falls when darkened (clean, dark)",
         is_monotonic(seq("brightness", ["clean", "dark_x0.4"]), False)),
        ("contrast falls when squeezed (clean, lowcon)",
         is_monotonic(seq("contrast", ["clean", "lowcon_x0.4"]), False)),
    ]


def plot_metrics(df: pd.DataFrame, th: dict, path: Path) -> None:
    panels = [("sharpness", "sharpness (var. of Laplacian)", "sharpness_min", True),
              ("noise", "noise estimate (Immerkaer, grey levels)", "noise_max", False),
              ("jpeg_blockiness", "JPEG blockiness (boundary / interior gradient)", "blockiness_max", False),
              ("brightness", "brightness (mean V)", "brightness_min", False),
              ("contrast", "contrast (std of grey)", "contrast_min", False)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (col, title, tkey, logy) in zip(axes.ravel(), panels):
        data = [df.loc[df["condition"] == c, col].values for c in COND_ORDER]
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(COND_ORDER, rotation=60, ha="right", fontsize=8)
        ax.axhline(th[tkey], color="r", linestyle="--", linewidth=1, label=f"threshold {th[tkey]:.2f}")
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp-rate", type=float, default=0.02,
                    help="false-alarm rate per detector on clean val images (default 0.02)")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N val images and N test queries (smoke test; thresholds not saved)")
    args = ap.parse_args(argv)
    if not 0.0 < args.fp_rate < 0.5:
        raise SystemExit("--fp-rate must be between 0 and 0.5")

    gallery = bc.Flowers102Gallery(bc.DATA_DIR)
    th, n_cal = calibrate(gallery, args.limit, args.fp_rate)

    t0, t1 = gallery.test_range()
    query_ids = bc.select_query_ids(t0, t1, bc.N_QUERIES, bc.SEED)
    if args.limit:
        query_ids = query_ids[:args.limit]
    df = measure_test(gallery, query_ids)
    df["bucket"] = [quality.assign_bucket(m, th) for m in df[list(quality.METRIC_NAMES)].to_dict("records")]

    out = bc.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / "quality_metrics_test.csv", index=False)

    med = df.groupby("condition")[list(quality.METRIC_NAMES)].median().reindex(COND_ORDER)
    med.round(3).to_csv(out / "quality_calibration.csv")

    conf = (pd.crosstab(df["condition"], df["bucket"])
            .reindex(index=COND_ORDER, columns=list(quality.BUCKETS), fill_value=0))
    pct = (conf.div(conf.sum(axis=1), axis=0) * 100).round(1)
    pct.insert(0, "expected", [EXPECTED[c] for c in pct.index])
    pct["hit_%"] = [pct.loc[c, EXPECTED[c]] for c in pct.index]
    pct.to_csv(out / "quality_confusion.csv")
    plot_metrics(df, th, out / "quality_metrics.png")

    print(f"\nThresholds (clean val percentiles, fp-rate {args.fp_rate}, n={n_cal}):")
    for k, v in th.items():
        print(f"  {k:15s} {v:9.3f}")
    if th["blockiness_max"] == BLOCKINESS_FLOOR:
        print(f"  (blockiness_max set to the floor {BLOCKINESS_FLOOR}: the clean percentile was lower)")
    if args.limit:
        print("  (smoke test: thresholds NOT saved)")
    else:
        meta = {"calibrated_on": "flowers102 val, clean images", "n_images": n_cal,
                "fp_rate": args.fp_rate}
        quality.THRESHOLDS_PATH.write_text(json.dumps({**th, "meta": meta}, indent=2), encoding="utf-8")
        print(f"  saved to {quality.THRESHOLDS_PATH}")

    print(f"\nMedian metrics per condition ({len(query_ids)} test queries):")
    print(med.round(2).to_string())
    print("\nBucket assignment, % of images per condition (hit_% = expected bucket):")
    print(pct.to_string())
    print("\nMetric ordering vs degradation strength:")
    for name, ok in ordering_checks(med):
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    print(f"\nSaved to {out}: quality_metrics_test.csv, quality_calibration.csv, "
          "quality_confusion.csv, quality_metrics.png")


if __name__ == "__main__":
    main()