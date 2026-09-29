"""End-to-end plumbing test for Person B's modules (no dataset download, no CLIP weights).

Run from the repo root:  python tests\\test_pipeline.py

Builds a small synthetic image-folder gallery, replaces the CLIP model with a
deterministic stand-in (a fixed random projection of the colour descriptor), then
checks: build_index, FAISS vs NumPy ranking, self-exclusion, text / image+text
search, and agreement of the FAISS index with Person A's rank_by_similarity.
The real CLIP model is checked separately by scripts/eval_clip.py on real data.
"""

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import index as ix  # noqa: E402
from src import retrieval as rt  # noqa: E402
from src.evaluate import evaluate, rank_by_similarity  # noqa: E402
from src.features import clip_encoder  # noqa: E402
from src.features.classical import color_feature  # noqa: E402

N_CLASSES, PER_CLASS = 5, 16
_PROJ = np.random.default_rng(0).normal(size=(256, 512)).astype(np.float32)


def fake_encode_images(imgs_rgb, batch_size=64):
    if not imgs_rgb:
        return np.zeros((0, 512), np.float32)
    v = np.stack([color_feature(cv2.cvtColor(im, cv2.COLOR_RGB2BGR)) for im in imgs_rgb])
    return rt.l2_rows(v @ _PROJ)


def fake_encode_text(texts, batch_size=256):
    out = []
    for t in texts:
        seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
        out.append(np.random.default_rng(seed).normal(size=512))
    return rt.l2_rows(np.asarray(out, np.float32))


def make_gallery(root: Path) -> None:
    rng = np.random.default_rng(1)
    hues = np.linspace(0, 170, N_CLASSES).astype(np.uint8)
    for c in range(N_CLASSES):
        d = root / f"class_{c}"
        d.mkdir(parents=True)
        for i in range(PER_CLASS):
            hsv = np.zeros((96, 128, 3), np.uint8)
            hsv[..., 0] = (int(hues[c]) + rng.integers(-6, 7)) % 180
            hsv[..., 1] = rng.integers(140, 255)
            hsv[..., 2] = rng.integers(120, 255, (96, 128))
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(str(d / f"{i:02d}.png"), bgr)


def main() -> None:
    clip_encoder.encode_images = fake_encode_images
    clip_encoder.encode_text = fake_encode_text
    import build_index

    tmp = Path(tempfile.mkdtemp(prefix="vs_test_"))
    try:
        data, idx = tmp / "gallery", tmp / "index"
        make_gallery(data)
        build_index.main(["--dataset", f"folder:{data}", "--index-dir", str(idx)])

        eng = rt.SearchEngine(idx)
        n = N_CLASSES * PER_CLASS
        assert eng.n_gallery == n and set(eng.available) == set(rt.RETRIEVERS), eng.available
        labels = np.load(idx / "labels.npy")
        assert labels.shape == (n,) and set(labels) == set(range(N_CLASSES))

        # 1) FAISS results == Person A's exact ranking, for every retriever.
        #    Compared by score (ties may legitimately swap ids).
        for name in eng.available:
            g = eng.vectors[name]
            q = g[:10]
            ref = rank_by_similarity(q, g, exclude_ids=np.arange(10))[:, :5]
            ref_scores = np.take_along_axis(q @ g.T, ref, axis=1)
            scores, ids = ix.search(eng.indexes[name], q, 6)
            got = np.array([[sc for sc, j in zip(srow, irow) if j != i][:5]
                            for i, (srow, irow) in enumerate(zip(scores, ids))])
            assert np.allclose(ref_scores, got, atol=1e-5), name
        print("ok: index search agrees with rank_by_similarity")

        # 2) NumPy fallback == FAISS
        v = eng.vectors["color"]
        s1, i1 = ix.search(ix.build_index(v), v[:5], 5)
        s2, i2 = ix.search(ix.NumpyFlatIP(v), v[:5], 5)
        assert np.allclose(s1, s2, atol=1e-5)  # ids may differ only on exact ties
        print("ok: NumPy fallback matches FAISS")

        # 3) image search with self-exclusion; colour classes are separable
        from src.dataset import FolderGallery
        gal = FolderGallery(data)
        for name in eng.available:
            img, lab = gal.get(3)
            r = eng.search_image(img, name, k=5, exclude_id=3)
            assert 3 not in r.ids and len(r.ids) == 5 and (np.diff(r.scores) <= 1e-6).all()
        r = eng.search_image(gal.get(3)[0], "clip", k=10, exclude_id=3)
        # synthetic classes overlap: chance is 15/79 = 0.19, stand-in CLIP scores ~0.55
        assert (labels[r.ids] == labels[3]).mean() >= 0.3
        print("ok: image search, self-exclusion, sorted scores")

        # 4) text and image+text
        rt_ = eng.search_text("a red flower", k=7)
        assert rt_.ids.shape == (7,) and rt_.retriever == "clip"
        rc = eng.search_image_text(gal.get(3)[0], "a red flower", k=7, w_img=0.5)
        assert rc.ids.shape == (7,)
        print("ok: text and image+text search")

        # 5) metrics through A's evaluate on the engine's output
        ranked = rank_by_similarity(eng.vectors["clip"][:20], eng.vectors["clip"],
                                    exclude_ids=np.arange(20))
        m = evaluate(ranked, labels[:20], labels, ks=(5, 10))
        assert m["P@10"] > 0.4, m   # chance = 0.19
        print("ok: evaluate on CLIP stand-in", {k: round(v, 3) for k, v in m.items()})

        # 6) resume: a partial file with the first 30 rows is continued and the final
        #    result equals the uninterrupted run; the partial file is then removed
        full = np.load(idx / "clip_image.npy")
        (idx / "clip_image.npy").unlink()
        np.save(idx / "clip_image.partial.npy", full[:30])
        build_index.main(["--dataset", f"folder:{data}", "--index-dir", str(idx), "--batch-size", "7"])
        assert (idx / "clip_image.npy").exists() and not (idx / "clip_image.partial.npy").exists()
        assert np.allclose(np.load(idx / "clip_image.npy"), full, atol=1e-6)
        print("ok: build_index re-run / resume path")
        print("\nAll pipeline tests passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
