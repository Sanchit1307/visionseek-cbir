"""Build all retrieval indexes (Person B).

Run from the repo root:
    python scripts\\build_index.py                    # Flowers-102, everything
    python scripts\\build_index.py --skip-clip        # classical only (fast)
    python scripts\\build_index.py --dataset folder:data\\demo --index-dir index_demo

Writes to index\\ (gitignored; share the .npy files via drive/pendrive):
    labels.npy, query_ids.npy                       (if missing)
    classical_{color,texture,edge,dct}.npy          (reused from Person A if present and
                                                     the right size, otherwise computed)
    classical_concat.npy
    clip_image.npy                                  (N, 512), resumable: an interrupted run
                                                     continues from clip_image.partial.npy
    <name>.faiss for every retriever above

CLIP encoding on CPU takes minutes to tens of minutes for 8,189 images; use
--limit N first to check the pipeline on a few images.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import index as ix  # noqa: E402
from src import retrieval as rt  # noqa: E402
from src.dataset import (INDEX_DIR, bgr_to_rgb, gallery_query_range,  # noqa: E402
                         open_gallery, select_query_ids)
from src.features import clip_encoder  # noqa: E402  (call via module: patchable in tests)
from src.features.classical import FEATURE_NAMES, extract_classical  # noqa: E402

SAVE_EVERY = 20  # batches between CLIP checkpoints


class LimitedGallery:
    """View of the first n images of a gallery (for quick smoke tests)."""

    def __init__(self, base, n: int) -> None:
        self.base, self.n = base, min(n, len(base))
        self.class_names = getattr(base, "class_names", None)

    def __len__(self) -> int:
        return self.n

    def get(self, g: int):
        return self.base.get(g)

    def labels(self) -> np.ndarray:
        return self.base.labels()[: self.n]


def ensure_labels(gallery, index_dir: Path) -> np.ndarray:
    p = index_dir / "labels.npy"
    if p.exists():
        lab = np.load(p)
        if len(lab) == len(gallery):
            return lab
        print("labels.npy does not match the gallery size, rewriting.")
    lab = np.asarray(gallery.labels(), dtype=np.int64)
    np.save(p, lab)
    return lab


def ensure_query_ids(gallery, index_dir: Path) -> np.ndarray:
    p = index_dir / "query_ids.npy"
    if p.exists():
        q = np.load(p)
        if q.max(initial=-1) < len(gallery):
            return q
    start, end = gallery_query_range(gallery)
    q = select_query_ids(start, end)
    np.save(p, q)
    return q


def ensure_classical(gallery, index_dir: Path) -> dict[str, np.ndarray]:
    paths = {n: index_dir / f"classical_{n}.npy" for n in FEATURE_NAMES}
    if all(p.exists() for p in paths.values()):
        feats = {n: np.load(p) for n, p in paths.items()}
        if all(len(v) == len(gallery) for v in feats.values()):
            print("Classical features: reusing existing files.")
            return feats
        print("Classical feature files do not match the gallery size, recomputing.")
    buckets = {n: [] for n in FEATURE_NAMES}
    for g in tqdm(range(len(gallery)), desc="Classical features"):
        f = extract_classical(gallery.get(g)[0])
        for n in FEATURE_NAMES:
            buckets[n].append(f[n])
    feats = {n: np.stack(v).astype(np.float32) for n, v in buckets.items()}
    for n, arr in feats.items():
        np.save(paths[n], arr)
    return feats


def encode_gallery_clip(gallery, index_dir: Path, batch_size: int) -> np.ndarray:
    out_path = index_dir / "clip_image.npy"
    part_path = index_dir / "clip_image.partial.npy"
    n = len(gallery)
    if out_path.exists():
        vecs = np.load(out_path)
        if vecs.shape == (n, clip_encoder.EMBED_DIM):
            print("CLIP embeddings: reusing existing clip_image.npy (use --rebuild to redo).")
            return vecs
        print("clip_image.npy does not match the gallery, recomputing.")
    done = np.zeros((0, clip_encoder.EMBED_DIM), dtype=np.float32)
    if part_path.exists():
        done = np.load(part_path)[:n]
        print(f"Resuming CLIP encoding from image {len(done)}.")
    chunks = [done] if len(done) else []
    start, batches = len(done), 0
    t0 = time.perf_counter()
    with tqdm(total=n, initial=start, desc="CLIP image embeddings") as bar:
        for s in range(start, n, batch_size):
            e = min(s + batch_size, n)
            imgs = [bgr_to_rgb(gallery.get(g)[0]) for g in range(s, e)]
            chunks.append(clip_encoder.encode_images(imgs))
            bar.update(e - s)
            batches += 1
            if batches % SAVE_EVERY == 0:
                np.save(part_path, np.concatenate(chunks, axis=0))
    vecs = np.concatenate(chunks, axis=0).astype(np.float32)
    assert vecs.shape == (n, clip_encoder.EMBED_DIM), vecs.shape
    np.save(out_path, vecs)
    if part_path.exists():
        part_path.unlink()
    print(f"CLIP encoding took {(time.perf_counter() - t0) / 60:.1f} min.")
    return vecs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="flowers102",
                    help="'flowers102' or 'folder:<path>' (root/<class>/<images>)")
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete existing clip_image.npy / partial before encoding")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N images")
    ap.add_argument("--batch-size", type=int, default=clip_encoder.BATCH_SIZE)
    args = ap.parse_args(argv)

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    gallery = open_gallery(args.dataset)
    if args.limit:
        gallery = LimitedGallery(gallery, args.limit)
    print(f"Gallery: {args.dataset}, {len(gallery)} images -> {index_dir}")

    if args.rebuild:
        for f in ("clip_image.npy", "clip_image.partial.npy"):
            (index_dir / f).unlink(missing_ok=True)

    labels = ensure_labels(gallery, index_dir)
    ensure_query_ids(gallery, index_dir)
    feats = ensure_classical(gallery, index_dir)
    vectors = dict(feats)
    vectors["classical_concat"] = rt.concat_classical(feats)
    np.save(index_dir / rt.vector_file("classical_concat"), vectors["classical_concat"])

    if not args.skip_clip:
        vectors["clip"] = encode_gallery_clip(gallery, index_dir, args.batch_size)

    print()
    for name in rt.RETRIEVERS:
        if name not in vectors:
            continue
        v = vectors[name]
        assert v.shape[0] == len(labels), f"{name}: {v.shape[0]} rows vs {len(labels)} labels"
        index = ix.build_index(v)
        ix.save_index(index, index_dir / rt.index_file(name))
        print(f"  {name:18s} dim {v.shape[1]:4d}  ntotal {index.ntotal}")
    print(f"\nDone. faiss available: {ix.HAVE_FAISS}")


if __name__ == "__main__":
    main()
