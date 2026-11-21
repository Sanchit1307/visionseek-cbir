"""Fabricates a small product catalogue in the layout of scripts/prepare_demo_dataset.py
(Person A): index dir with the vector files + labels.npy + metadata.csv + meta.json,
and a flat image folder of <id>.jpg files (60 x 80 px). Test helper only."""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TYPES = {"Shoes": ("Footwear", "Shoes", "Men"), "Tshirts": ("Apparel", "Topwear", "Women"),
         "Watches": ("Accessories", "Watches", "Men"), "Handbags": ("Accessories", "Bags", "Women")}
COLOURS = ["Red", "Blue", "Black", "White", "Green", None]          # None -> missing value
BGR = {"Red": (30, 30, 220), "Blue": (220, 60, 30), "Black": (20, 20, 20), "White": (240, 240, 240),
       "Green": (40, 200, 40), None: (120, 120, 120)}


def make_catalog(tmp: Path, per_type: int = 12, with_id_column: bool = True):
    """Returns (index_dir, image_root, metadata DataFrame). Needs the CLIP stand-in installed."""
    import build_index

    src, flat = tmp / "catalog_src", tmp / "catalog_images"
    flat.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows, next_id = [], 10001
    for ti, (atype, (master, sub, gender)) in enumerate(sorted(TYPES.items())):
        (src / atype).mkdir(parents=True, exist_ok=True)
        for j in range(per_type):
            colour = COLOURS[(ti + j) % len(COLOURS)]
            img = np.full((80, 60, 3), BGR[colour], np.uint8)
            for _ in range(3):                                       # a few type-specific blobs
                cv2.circle(img, (int(rng.integers(8, 52)), int(rng.integers(8, 72))),
                           int(rng.integers(4, 12)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
            if ti % 2:
                img = cv2.GaussianBlur(img, (5, 5), 0)
            name = f"{next_id}.jpg"
            cv2.imwrite(str(src / atype / name), img)
            cv2.imwrite(str(flat / name), img)
            rows.append({"id": next_id, "articleType": atype, "masterCategory": master, "subCategory": sub,
                         "gender": gender, "baseColour": colour,
                         "productDisplayName": f"{colour or 'Plain'} {atype} model {j}"})
            next_id += 1
    idx = tmp / "index_demo_cat"
    build_index.main(["--dataset", f"folder:{src}", "--index-dir", str(idx)])
    meta = pd.DataFrame(rows)       # same order as FolderGallery: sorted type folder, sorted file name
    # same format as scripts/prepare_demo_dataset.py: index column "row", absolute image_root, extra keys
    (meta if with_id_column else meta.drop(columns=["id"])).to_csv(idx / "metadata.csv", index_label="row")
    (idx / "meta.json").write_text(json.dumps({
        "source": "synthetic", "image_root": str(flat.resolve()), "types": sorted(TYPES), "per_type": per_type,
        "seed": 42, "n": int(len(meta)), "label": "index into 'types' (articleType)"}, indent=2), encoding="utf-8")
    shutil.rmtree(src, ignore_errors=True)
    return idx, flat, meta
