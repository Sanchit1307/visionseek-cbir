"""Build the practical demo dataset from Kaggle "Fashion Product Images (Small)" (Person A, WP13a).

Expected layout (what you already have):
    data\\fashion\\styles.csv
    data\\fashion\\images\\<id>.jpg           (60 x 80 px)

Run from the repo root:
    python scripts\\prepare_demo_dataset.py --list-types       # show the article types and counts
    python scripts\\prepare_demo_dataset.py --limit 40          # smoke test (40 images per type is NOT used; see below)
    python scripts\\prepare_demo_dataset.py                     # full build
    python scripts\\prepare_demo_dataset.py --no-clip           # classical only (no CLIP download)

What it does
  1. Selects --per-type images for each article type in --types (seeded random sample,
     only rows whose image file exists and has a known base colour).
  2. Extracts the classical descriptors and the CLIP embeddings (the same functions the
     Flowers-102 pipeline uses) and writes them with Person B's file names to
         index\\demo\\{clip_image, classical_color, classical_texture, classical_edge,
                      classical_dct, classical_concat, labels}.npy
         index\\demo\\metadata.csv   (row i = vector i; articleType, baseColour, ... for the UI filters)
         index\\demo\\meta.json
     so B can load it with  SearchEngine(INDEX_DIR / "demo").
  3. Sanity check (results\\demo_eval.csv): P@5 / P@10 / mAP for "same article type" over
     random gallery queries (query excluded), P@10 for "same base colour" (is the HSV
     descriptor a usable colour filter?) and CLIP zero-shot text accuracy ("a photo of a <type>").

Demo data only: these numbers are NOT headline results (Flowers-102 is the benchmark).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import DATA_DIR, INDEX_DIR, RESULTS_DIR  # noqa: E402
from src.evaluate import evaluate, rank_by_similarity  # noqa: E402
from src.features.classical import FEATURE_NAMES, extract_classical  # noqa: E402

DEFAULT_TYPES = ["Tshirts", "Shirts", "Casual Shoes", "Sports Shoes",
                 "Watches", "Handbags", "Sunglasses", "Backpacks"]
SEED = 42
KS = (5, 10)
META_COLS = ["id", "articleType", "masterCategory", "subCategory", "gender",
             "baseColour", "productDisplayName"]


def l2_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def read_styles(root: Path) -> pd.DataFrame:
    csv = root / "styles.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found. Unzip the Kaggle dataset so that styles.csv and "
                         f"the images folder are inside {root}.")
    df = pd.read_csv(csv, on_bad_lines="skip")
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"styles.csv lacks the columns {missing}; found {df.columns.tolist()}")
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(np.int64)
    return df


def select_rows(df: pd.DataFrame, images_dir: Path, types: list[str], per_type: int,
                seed: int) -> pd.DataFrame:
    have = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    ok = df[df["id"].astype(str).isin(have)]
    ok = ok[ok["baseColour"].notna() & (ok["baseColour"].astype(str).str.upper() != "NA")]
    counts = ok["articleType"].value_counts()
    unknown = [t for t in types if t not in counts.index]
    if unknown:
        raise SystemExit(f"Unknown article type(s) {unknown}. Most common types:\n"
                         + counts.head(40).to_string())
    rng = np.random.default_rng(seed)
    parts = []
    for t in types:
        sub = ok[ok["articleType"] == t]
        n = min(per_type, len(sub))
        if n < per_type:
            print(f"WARNING: only {len(sub)} usable images for '{t}' (wanted {per_type}).")
        pick = rng.choice(len(sub), size=n, replace=False)
        parts.append(sub.iloc[np.sort(pick)])
    out = pd.concat(parts, ignore_index=True)
    return out[META_COLS].reset_index(drop=True)


def load_images(meta: pd.DataFrame, images_dir: Path) -> tuple[list[np.ndarray], pd.DataFrame]:
    imgs, keep = [], []
    for i, pid in enumerate(tqdm(meta["id"], desc="Reading images")):
        im = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = images_dir / f"{pid}{ext}"
            if p.exists():
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                break
        if im is not None and im.ndim == 3 and im.shape[2] == 3 and min(im.shape[:2]) >= 16:
            imgs.append(im)
            keep.append(i)
    if len(keep) < len(meta):
        print(f"WARNING: dropped {len(meta) - len(keep)} unreadable or too small images.")
    return imgs, meta.iloc[keep].reset_index(drop=True)


def classical_features(imgs: list[np.ndarray]) -> dict[str, np.ndarray]:
    buckets = {n: [] for n in FEATURE_NAMES}
    for im in tqdm(imgs, desc="Classical features"):
        f = extract_classical(im)
        for n in FEATURE_NAMES:
            buckets[n].append(f[n])
    feats = {n: np.stack(v).astype(np.float32) for n, v in buckets.items()}
    feats["classical_concat"] = l2_rows(np.hstack([feats[n] for n in FEATURE_NAMES]))
    return feats


def clip_features(imgs: list[np.ndarray], chunk: int = 256) -> np.ndarray:
    from src.features import clip_encoder  # lazy: loads torch / open_clip only when needed

    out = []
    for s in tqdm(range(0, len(imgs), chunk), desc="CLIP embeddings"):
        rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs[s:s + chunk]]
        out.append(clip_encoder.encode_images(rgb))
    return np.concatenate(out).astype(np.float32)


def sanity_check(vecs: dict[str, np.ndarray], type_codes: np.ndarray, colour_codes: np.ndarray,
                 n_queries: int, seed: int) -> pd.DataFrame:
    n = len(type_codes)
    qids = np.sort(np.random.default_rng(seed).choice(n, size=min(n_queries, n), replace=False))
    rows = []
    for name, v in vecs.items():
        ranked = rank_by_similarity(v[qids], v, exclude_ids=qids)
        by_type = evaluate(ranked, type_codes[qids], type_codes, ks=KS)
        by_col = evaluate(ranked, colour_codes[qids], colour_codes, ks=KS)
        rows.append({"retriever": name, "n_queries": len(qids),
                     **{f"type_{k}": round(v_, 4) for k, v_ in by_type.items()},
                     "colour_P@10": round(by_col["P@10"], 4)})
    return pd.DataFrame(rows)


def zero_shot_accuracy(clip_vecs: np.ndarray, type_codes: np.ndarray, types: list[str]) -> float:
    from src.features import clip_encoder

    txt = clip_encoder.encode_text([f"a photo of a {t.lower()}" for t in types])
    pred = (clip_vecs @ txt.T).argmax(axis=1)
    return float((pred == type_codes).mean())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DATA_DIR / "fashion",
                    help="folder with styles.csv and images\\ (default data\\fashion)")
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR / "demo")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--types", nargs="+", default=DEFAULT_TYPES,
                    help="article types (quote names with spaces)")
    ap.add_argument("--per-type", type=int, default=250)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--queries", type=int, default=300, help="queries for the sanity check")
    ap.add_argument("--list-types", action="store_true", help="print article types and exit")
    ap.add_argument("--no-clip", action="store_true", help="skip CLIP (classical only)")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: use at most N images per type and do not write files")
    args = ap.parse_args(argv)

    images_dir = args.root / "images"
    df = read_styles(args.root)
    if args.list_types:
        print(df["articleType"].value_counts().head(40).to_string())
        return
    if not images_dir.is_dir():
        raise SystemExit(f"{images_dir} not found.")

    per_type = min(args.per_type, args.limit) if args.limit else args.per_type
    meta = select_rows(df, images_dir, args.types, per_type, args.seed)
    imgs, meta = load_images(meta, images_dir)
    print(f"\nDemo gallery: {len(meta)} images, {meta['articleType'].nunique()} article types, "
          f"{meta['baseColour'].nunique()} base colours; sizes e.g. {imgs[0].shape[1]}x{imgs[0].shape[0]} px")
    print(meta["articleType"].value_counts().to_string())

    feats = classical_features(imgs)
    if not args.no_clip:
        feats["clip"] = clip_features(imgs)

    types = [t for t in args.types if t in set(meta["articleType"])]
    type_codes = meta["articleType"].map({t: i for i, t in enumerate(types)}).to_numpy(np.int64)
    colour_codes = pd.factorize(meta["baseColour"])[0].astype(np.int64)

    check_names = [n for n in ("color", "classical_concat", "clip") if n in feats]
    res = sanity_check({n: feats[n] for n in check_names}, type_codes, colour_codes,
                       args.queries, args.seed)
    if "clip" in feats:
        res["clip_zero_shot_type_acc"] = np.nan
        res.loc[res["retriever"] == "clip", "clip_zero_shot_type_acc"] = round(
            zero_shot_accuracy(feats["clip"], type_codes, types), 4)
    print("\nSanity check (same article type = relevant; colour_P@10 = same base colour):")
    print(res.to_string(index=False))
    if "clip" in feats:
        p10 = float(res.loc[res["retriever"] == "clip", "type_P@10"].iloc[0])
        print("\nCLIP P@10 on article type: %.2f -> %s" % (
            p10, "usable for the demo." if p10 >= 0.7 else
            "LOW: the 60x80 images may be too small; tell me before building the UI on this."))

    if args.limit:
        print("\n(smoke test: nothing saved)")
        return
    args.index_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in feats.items():
        fname = "clip_image.npy" if name == "clip" else (
            "classical_concat.npy" if name == "classical_concat" else f"classical_{name}.npy")
        np.save(args.index_dir / fname, arr.astype(np.float32))
    np.save(args.index_dir / "labels.npy", type_codes)
    meta.to_csv(args.index_dir / "metadata.csv", index_label="row")
    (args.index_dir / "meta.json").write_text(json.dumps({
        "source": "Kaggle: paramaggarwal/fashion-product-images-small",
        "image_root": str(images_dir), "types": types, "per_type": per_type,
        "seed": args.seed, "n": int(len(meta)), "label": "index into 'types' (articleType)"},
        indent=2), encoding="utf-8")
    res.to_csv(args.results_dir / "demo_eval.csv", index=False)
    print(f"\nSaved vectors, labels.npy, metadata.csv, meta.json to {args.index_dir}")
    print(f"Saved {args.results_dir / 'demo_eval.csv'}")


if __name__ == "__main__":
    main()