"""Robustness and restoration experiments (Person A).

Run from the repo root (after scripts\\baseline_classical.py has been run once):
    python scripts\\run_experiments.py
    python scripts\\run_experiments.py --limit 50      # quick smoke test

Protocol: gallery = clean features of all images (from index\\); queries = the 500
saved test queries, degraded (noise / blur / JPEG) at native resolution; the
clean version of each query is excluded from its own ranking; relevant = same
class. Restoration = filter the degraded query before feature extraction.

Outputs: results\\robustness.csv/.png and results\\restoration.csv/.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_classical as bc  # noqa: E402
from src.degrade import degrade, restore  # noqa: E402
from src.evaluate import evaluate, rank_by_similarity  # noqa: E402
from src.features.classical import FEATURE_NAMES, extract_classical  # noqa: E402

NOISE_SIGMAS = (10, 25, 50)
BLUR_KERNELS = (5, 9, 15)
JPEG_QUALITIES = (30, 10)
RESTORATIONS = (("median", 3), ("median", 5), ("gaussian", 3), ("gaussian", 5))
KS = (5, 10)
NOISE_SEED = 1234

CONDITIONS = (
    [("noise", s) for s in NOISE_SIGMAS]
    + [("blur", k) for k in BLUR_KERNELS]
    + [("jpeg", q) for q in JPEG_QUALITIES]
)
_LEVEL_TAG = {"noise": "s", "blur": "k", "jpeg": "q"}
ORDER = ["color", "texture", "edge", "dct", "classical_concat", "clip"]
CLIP_FILE = "clip_image.npy"   # Person B: gallery CLIP embeddings, same row order as labels.npy


def cond_name(kind: str, level: int) -> str:
    return f"{kind}_{_LEVEL_TAG[kind]}{level}"


def _row_l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# Query encoders: images (BGR list) -> {retriever name: (Q, D) vectors}.
# To add CLIP later, append another encoder that returns {"clip": (Q, 512)}
# and add the matching gallery vectors {"clip": ...} in load_gallery().
# ---------------------------------------------------------------------------
def classical_encoder(images: list[np.ndarray]) -> dict[str, np.ndarray]:
    buckets = {name: [] for name in FEATURE_NAMES}
    for im in images:
        f = extract_classical(im)
        for name in FEATURE_NAMES:
            buckets[name].append(f[name])
    out = {name: np.stack(v).astype(np.float32) for name, v in buckets.items()}
    out["classical_concat"] = _row_l2(np.hstack([out[n] for n in FEATURE_NAMES]))
    return out


def clip_query_encoder(images: list[np.ndarray]) -> dict[str, np.ndarray]:
    """BGR images -> {'clip': (Q, 512)} with Person B's encoder (converted to RGB here)."""
    from src.features import clip_encoder  # lazy: torch / open_clip load only when used

    rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images]
    return {"clip": clip_encoder.encode_images(rgb)}


def load_gallery(with_clip: bool = True) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    need = [bc.INDEX_DIR / f"classical_{n}.npy" for n in FEATURE_NAMES]
    need += [bc.INDEX_DIR / "labels.npy", bc.INDEX_DIR / "query_ids.npy"]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {missing} in {bc.INDEX_DIR}. "
                         "Run scripts\\baseline_classical.py first.")
    feats = {n: np.load(bc.INDEX_DIR / f"classical_{n}.npy") for n in FEATURE_NAMES}
    feats["classical_concat"] = _row_l2(np.hstack([feats[n] for n in FEATURE_NAMES]))
    labels = np.load(bc.INDEX_DIR / "labels.npy")
    query_ids = np.load(bc.INDEX_DIR / "query_ids.npy")
    if with_clip:
        cp = bc.INDEX_DIR / CLIP_FILE
        if cp.exists():
            clip = np.load(cp)
            if clip.ndim != 2 or clip.shape[0] != len(labels):
                raise SystemExit(f"{cp.name} has shape {clip.shape}, expected "
                                 f"({len(labels)}, D) in the same order as labels.npy.")
            feats["clip"] = _row_l2(clip)
        else:
            print(f"NOTE: {cp} not found -> running the classical retrievers only.")
    return feats, labels, query_ids


def score(qvecs: dict[str, np.ndarray], gallery: dict[str, np.ndarray],
          labels: np.ndarray, query_ids: np.ndarray) -> dict[str, dict[str, float]]:
    res = {}
    for name, q in qvecs.items():
        ranked = rank_by_similarity(q, gallery[name], exclude_ids=query_ids)
        res[name] = evaluate(ranked, labels[query_ids], labels, ks=KS)
    return res


def run_experiments(images: list[np.ndarray], gallery: dict[str, np.ndarray],
                    labels: np.ndarray, query_ids: np.ndarray,
                    encoders=(classical_encoder,)) -> tuple[pd.DataFrame, pd.DataFrame]:
    def encode(imgs: list[np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for enc in encoders:
            out.update(enc(imgs))
        return out

    clean_scores = score(encode(images), gallery, labels, query_ids)
    rob_rows, res_rows = [], []
    for name, m in clean_scores.items():
        rob_rows.append({"kind": "clean", "level": 0, "condition": "clean",
                         "retriever": name, **m})

    for kind, level in tqdm(CONDITIONS, desc="Conditions"):
        cname = cond_name(kind, level)
        degraded = [degrade(im, kind, level, seed=NOISE_SEED + int(q))
                    for im, q in zip(images, query_ids)]
        deg_scores = score(encode(degraded), gallery, labels, query_ids)
        for name, m in deg_scores.items():
            rob_rows.append({"kind": kind, "level": level, "condition": cname,
                             "retriever": name, **m})

        for rkind, ksize in RESTORATIONS:
            restored = [restore(im, rkind, ksize) for im in degraded]
            rest_scores = score(encode(restored), gallery, labels, query_ids)
            for name, m in rest_scores.items():
                clean_map, deg_map = clean_scores[name]["mAP"], deg_scores[name]["mAP"]
                gap = clean_map - deg_map
                gain = m["mAP"] - deg_map
                res_rows.append({
                    "kind": kind, "level": level, "condition": cname,
                    "restoration": f"{rkind}{ksize}", "retriever": name, **m,
                    "mAP_clean": clean_map, "mAP_degraded": deg_map,
                    "mAP_gain": gain,
                    "recovery_pct": (gain / gap * 100.0) if gap > 0.005 else np.nan,
                })

    rob = pd.DataFrame(rob_rows).round(4)
    res = pd.DataFrame(res_rows).round(4)
    return rob, res


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_robustness(rob: pd.DataFrame, path: Path) -> None:
    order = ["clean"] + [cond_name(k, l) for k, l in CONDITIONS]
    retrievers = list(dict.fromkeys(rob["retriever"]))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    width = 0.8 / len(retrievers)
    x = np.arange(len(order))
    for ax, metric in zip(axes, ("P@10", "mAP")):
        piv = rob.pivot(index="condition", columns="retriever", values=metric).reindex(order)
        for i, r in enumerate(retrievers):
            ax.bar(x + (i - (len(retrievers) - 1) / 2) * width, piv[r].values, width, label=r)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} under query degradation")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_restoration(res: pd.DataFrame, rob: pd.DataFrame, path: Path) -> None:
    order = [cond_name(k, l) for k, l in CONDITIONS]
    retrievers = list(dict.fromkeys(res["retriever"]))
    variants = ["none"] + [f"{k}{s}" for k, s in RESTORATIONS]
    n = len(retrievers)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    width = 0.8 / len(variants)
    x = np.arange(len(order))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, r in zip(axes.ravel(), retrievers):
        base = rob[rob["retriever"] == r].set_index("condition")["mAP"].reindex(order)
        for i, v in enumerate(variants):
            if v == "none":
                vals = base.values
            else:
                sub = res[(res["retriever"] == r) & (res["restoration"] == v)]
                vals = sub.set_index("condition")["mAP"].reindex(order).values
            ax.bar(x + (i - (len(variants) - 1) / 2) * width, vals, width, label=v)
        clean_map = rob[(rob["retriever"] == r) & (rob["condition"] == "clean")]["mAP"].iloc[0]
        ax.axhline(clean_map, color="k", linestyle="--", linewidth=1, label="clean")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{r}: mAP with restoration")
        ax.set_ylabel("mAP")
        ax.grid(axis="y", alpha=0.3)
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def check_against_baseline(rob: pd.DataFrame) -> None:
    """The clean rows must reproduce the baseline CSVs (same queries, same protocol)."""
    clean = rob[rob["condition"] == "clean"].set_index("retriever")
    for fname in ("baseline_classical.csv", "baseline_clip.csv"):
        p = bc.RESULTS_DIR / fname
        if not p.exists():
            continue
        base = pd.read_csv(p).set_index("retriever")
        for name in base.index:
            if name not in clean.index:
                continue
            got, ref = float(clean.loc[name, "mAP"]), float(base.loc[name, "mAP"])
            status = "OK" if abs(got - ref) < 5e-4 else "MISMATCH"
            print(f"  clean check  {name:18s} mAP {got:.4f} vs baseline {ref:.4f}  {status}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N saved queries (quick test)")
    ap.add_argument("--no-clip", action="store_true",
                    help="skip the CLIP retriever (classical only)")
    args = ap.parse_args(argv)

    gallery, labels, query_ids = load_gallery(with_clip=not args.no_clip)
    encoders = (classical_encoder,) + ((clip_query_encoder,) if "clip" in gallery else ())
    print("Retrievers:", ", ".join(gallery))
    if args.limit:
        query_ids = query_ids[:args.limit]

    ds = bc.Flowers102Gallery(bc.DATA_DIR)
    print(f"Loading {len(query_ids)} query images...")
    images = [ds.get(int(g))[0] for g in tqdm(query_ids, desc="Loading")]

    rob, res = run_experiments(images, gallery, labels, query_ids, encoders=encoders)

    out = bc.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    rob.to_csv(out / "robustness.csv", index=False)
    res.to_csv(out / "restoration.csv", index=False)
    plot_robustness(rob, out / "robustness.png")
    plot_restoration(res, rob, out / "restoration.png")

    if not args.limit:
        print("\nClean rows vs baseline CSVs:")
        check_against_baseline(rob)

    print("\nmAP by condition (rows) and retriever (columns):")
    piv = rob.pivot(index="condition", columns="retriever", values="mAP")
    piv = piv.reindex(["clean"] + [cond_name(k, l) for k, l in CONDITIONS])
    print(piv[[c for c in ORDER if c in piv.columns]].to_string())
    best = (res.sort_values("mAP_gain", ascending=False)
            .groupby(["condition", "retriever"]).head(1))
    for r in ("classical_concat", "clip"):
        if r in set(best["retriever"]):
            print(f"\nBest restoration per condition for {r}:")
            print(best[best["retriever"] == r]
                  [["condition", "restoration", "mAP_degraded", "mAP", "recovery_pct"]]
                  .to_string(index=False))
    print(f"\nSaved to {out}: robustness.csv/.png, restoration.csv/.png")


if __name__ == "__main__":
    main()