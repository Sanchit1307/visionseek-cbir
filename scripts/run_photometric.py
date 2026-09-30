"""Photometric robustness: darkening and contrast reduction (Person A).

Run from the repo root (after baseline_classical.py; CLIP is included if
index\\clip_image.npy exists):
    python scripts\\run_photometric.py
    python scripts\\run_photometric.py --limit 50      # quick smoke test
    python scripts\\run_photometric.py --no-clip

Same protocol as scripts\\run_experiments.py (500 saved test queries, clean gallery,
self-match excluded, relevant = same class). Conditions: brightness scaled by
0.6 / 0.4 ("dark") and contrast squeezed around the mean by 0.6 / 0.4 ("lowcon").
These give the dark/low-contrast quality bucket its own retrieval numbers.

Output: results\\photometric.csv
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_classical as bc  # noqa: E402
import run_experiments as rx  # noqa: E402
from calibrate_quality import apply_condition  # noqa: E402

CONDITIONS = [("dark_x0.6", "dark", 0.6), ("dark_x0.4", "dark", 0.4),
              ("lowcon_x0.6", "lowcon", 0.6), ("lowcon_x0.4", "lowcon", 0.4)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N saved queries (quick test)")
    ap.add_argument("--no-clip", action="store_true", help="skip the CLIP retriever")
    args = ap.parse_args(argv)

    gallery, labels, query_ids = rx.load_gallery(with_clip=not args.no_clip)
    if args.limit:
        query_ids = query_ids[:args.limit]
    encoders = (rx.classical_encoder,) + ((rx.clip_query_encoder,) if "clip" in gallery else ())
    print("Retrievers:", ", ".join(gallery))

    def encode(imgs: list[np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for enc in encoders:
            out.update(enc(imgs))
        return out

    ds = bc.Flowers102Gallery(bc.DATA_DIR)
    print(f"Loading {len(query_ids)} query images...")
    images = [ds.get(int(g))[0] for g in tqdm(query_ids, desc="Loading")]

    clean = rx.score(encode(images), gallery, labels, query_ids)
    rows = [{"condition": "clean", "kind": "clean", "factor": 1.0, "retriever": n, **m}
            for n, m in clean.items()]
    for name, kind, factor in tqdm(CONDITIONS, desc="Conditions"):
        degraded = [apply_condition(im, kind, factor, 0) for im in images]
        for n, m in rx.score(encode(degraded), gallery, labels, query_ids).items():
            rows.append({"condition": name, "kind": kind, "factor": factor, "retriever": n, **m})

    df = pd.DataFrame(rows).round(4)
    df["mAP_clean"] = df["retriever"].map({n: round(m["mAP"], 4) for n, m in clean.items()})
    df["mAP_retained_pct"] = (df["mAP"] / df["mAP_clean"] * 100).round(1)
    out = bc.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "photometric.csv", index=False)

    if not args.limit:
        print("\nClean rows vs baseline CSVs:")
        rx.check_against_baseline(df[df["condition"] == "clean"])

    order = ["clean"] + [c[0] for c in CONDITIONS]
    for col, title in (("mAP", "mAP"), ("mAP_retained_pct", "% of clean mAP retained")):
        piv = df.pivot(index="condition", columns="retriever", values=col).reindex(order)
        print(f"\n{title} by condition (rows) and retriever (columns):")
        print(piv[[c for c in rx.ORDER if c in piv.columns]].to_string())
    print(f"\nSaved: {out / 'photometric.csv'}")


if __name__ == "__main__":
    main()