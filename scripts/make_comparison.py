"""Comparison table and plots for the slides (Person B, WP4b).

Run from the repo root after Person A's baseline and eval_clip.py:
    python scripts\\make_comparison.py

Reads results\\baseline_classical.csv (+ baseline_clip.csv if present, and
robustness.csv for the heatmap) and writes:
    results\\comparison_table.csv / .md     P@5, P@10, mAP, latency per retriever
    results\\comparison.png                 quality bars + latency bars
    results\\robustness_heatmap.png         mAP retained (% of clean) per condition
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import RESULTS_DIR  # noqa: E402

ORDER = ["color", "texture", "edge", "dct", "classical_concat", "clip"]
LABEL = {"color": "Colour (HSV)", "texture": "Texture (LBP)", "edge": "Edge",
         "dct": "DCT", "classical_concat": "Classical (concat)", "clip": "CLIP"}


def load_table(res: Path) -> pd.DataFrame:
    frames = []
    for f in ("baseline_classical.csv", "baseline_clip.csv"):
        p = res / f
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit(f"No baseline CSVs found in {res}")
    df = pd.concat(frames, ignore_index=True)
    df["order"] = df["retriever"].map({n: i for i, n in enumerate(ORDER)})
    return df.sort_values("order").drop(columns="order").reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    cols = ["retriever", "dim", "P@5", "P@10", "mAP", "extract_ms", "search_ms", "total_ms"]
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return head + "\n".join(rows) + "\n"


def plot_comparison(df: pd.DataFrame, path: Path) -> None:
    names = [LABEL.get(r, r) for r in df["retriever"]]
    x = np.arange(len(df))
    colors = ["#4c72b0" if r != "clip" else "#dd8452" for r in df["retriever"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    w = 0.38
    axes[0].bar(x - w / 2, df["P@5"], w, label="P@5", color="#4c72b0")
    axes[0].bar(x + w / 2, df["P@10"], w, label="P@10", color="#55a868")
    axes[0].set_title("Precision at K (500 queries, Flowers-102)")
    axes[0].legend()

    bars = axes[1].bar(x, df["mAP"], color=colors)
    axes[1].set_title("mean Average Precision")
    for b, v in zip(bars, df["mAP"]):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    tot = axes[2].bar(x, df["total_ms"], color=colors)
    axes[2].set_yscale("log")
    axes[2].set_title("Total latency per query (extraction + search, ms, log scale)")
    for b, v in zip(tot, df["total_ms"]):
        axes[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    axes[2].set_ylim(top=df["total_ms"].max() * 2.5)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_robustness_heatmap(rob: pd.DataFrame, path: Path) -> None:
    order = list(dict.fromkeys(rob["condition"]))
    piv = rob.pivot(index="retriever", columns="condition", values="mAP").reindex(columns=order)
    piv = piv.reindex([r for r in ORDER if r in piv.index])
    rel = piv.div(piv["clean"], axis=0) * 100.0
    rel = rel.drop(columns="clean")

    fig, ax = plt.subplots(figsize=(1.1 * rel.shape[1] + 3, 0.7 * rel.shape[0] + 2))
    im = ax.imshow(rel.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(rel.shape[1]))
    ax.set_xticklabels(rel.columns, rotation=45, ha="right")
    ax.set_yticks(range(rel.shape[0]))
    ax.set_yticklabels([LABEL.get(r, r) for r in rel.index])
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            ax.text(j, i, f"{rel.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("mAP retained under degradation (% of clean mAP)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    res = RESULTS_DIR
    df = load_table(res)
    df.to_csv(res / "comparison_table.csv", index=False)
    (res / "comparison_table.md").write_text(to_markdown(df), encoding="utf-8")
    plot_comparison(df, res / "comparison.png")
    print(df.to_string(index=False))
    saved = ["comparison_table.csv/.md", "comparison.png"]
    if "clip" not in set(df["retriever"]):
        print("\n(no CLIP row yet: run scripts\\build_index.py then scripts\\eval_clip.py)")

    rp = res / "robustness.csv"
    if rp.exists():
        plot_robustness_heatmap(pd.read_csv(rp), res / "robustness_heatmap.png")
        saved.append("robustness_heatmap.png")
    print(f"\nSaved to {res}: " + ", ".join(saved))


if __name__ == "__main__":
    main()
