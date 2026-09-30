"""Tests for src/explain.py (Person B, WP11).

Run from the repo root:  python tests\\test_explain.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from src import explain as ex  # noqa: E402


def test_numbers_and_reasons() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    sims = {"clip": rng.normal(0.3, 0.1, n).astype(np.float32),
            "color": rng.normal(0.6, 0.1, n).astype(np.float32),
            "texture": rng.normal(0.5, 0.1, n).astype(np.float32)}
    sims["clip"][7] = 0.95      # rarest by far for CLIP
    sims["color"][7] = 0.99     # also very high for colour
    sims["texture"][7] = 0.10   # far below average texture
    e = ex.explain_results(sims, np.array([7]), {"clip": 0.8, "color": 0.2})[0]

    rows = {r["retriever"]: r for r in e.rows}
    assert set(rows) == {"clip", "color", "texture"}            # classical_concat not used -> hidden
    assert abs(rows["clip"]["cosine"] - 0.95) < 1e-6
    assert rows["clip"]["top_pct"] == 0.0 and rows["texture"]["top_pct"] > 90
    assert abs(rows["clip"]["weight"] - 0.8) < 1e-9 and rows["texture"]["weight"] == 0
    z = (0.95 - sims["clip"].mean()) / sims["clip"].std()
    assert abs(rows["clip"]["z"] - z) < 1e-4
    # contributions add up to the fused score, and texture contributes nothing
    assert rows["texture"]["contribution"] == 0
    assert abs(e.fused - sum(r["contribution"] for r in e.rows)) < 1e-9
    # reasons: weighted retrievers first, weak texture excluded, at most MAX_REASONS
    assert e.reasons[0].startswith("similar overall content") and "very strong" in e.reasons[0]
    assert e.reasons[1].startswith("similar colour")
    assert not any("texture" in r for r in e.reasons)
    assert e.short == "content, colour" and e.text.startswith("Returned because of")
    print("ok: cosine / top_pct / z / weight / contribution, reasons ordered and filtered")

    # a result that is weak everywhere gets the fallback sentence
    sims2 = {k: v.copy() for k, v in sims.items()}
    for k in sims2:
        sims2[k][5] = sims2[k].min() - 0.1
    e2 = ex.explain_results(sims2, np.array([5]), {"clip": 1.0})[0]
    assert e2.reasons == [] and e2.text.startswith("Weak similarity")
    for bad in ({}, {"clip": 0.0}):
        try:
            ex.explain_results(sims, np.array([7]), bad)
        except ValueError:
            continue
        raise AssertionError("empty weights accepted")
    print("ok: fallback text, invalid weights rejected")

    assert ex.strength(0.5) == "very strong" and ex.strength(3) == "strong"
    assert ex.strength(15) == "moderate" and ex.strength(50) is None
    print("ok: strength bands")


def test_visuals() -> None:
    red = np.zeros((90, 120, 3), np.uint8); red[..., 2] = 220
    grey = np.full((90, 120, 3), 128, np.uint8)
    h_red, h_grey = ex.hue_histogram_image(red), ex.hue_histogram_image(grey)
    assert h_red.shape == (110, 288, 3) and h_red.dtype == np.uint8
    assert (h_red != 245).any() and (h_grey == 245).all(), "grey image must draw no bars"
    stripes = np.zeros((90, 120, 3), np.uint8); stripes[:, ::10] = 255
    e_flat, e_stripes = ex.edge_map(grey), ex.edge_map(stripes)
    assert e_flat.shape == (160, 160, 3) and e_flat.max() == 0
    assert e_stripes.mean() > 10
    print("ok: hue histogram and edge map images")


def test_engine_consistency() -> None:
    import test_pipeline as tp
    from src import retrieval as rt
    from src.dataset import FolderGallery
    from src.features import clip_encoder

    clip_encoder.encode_images = tp.fake_encode_images
    clip_encoder.encode_text = tp.fake_encode_text
    import build_index

    tmp = Path(tempfile.mkdtemp(prefix="vs_explain_"))
    try:
        data, idx = tmp / "gallery", tmp / "index"
        tp.make_gallery(data)
        build_index.main(["--dataset", f"folder:{data}", "--index-dir", str(idx)])
        eng, gal = rt.SearchEngine(idx), FolderGallery(data)
        img, _ = gal.get(3)

        sims = eng.all_similarities(img)
        assert set(sims) == set(ex.EXPLAIN_NAMES) | {"classical_concat"}
        assert all(v.shape == (eng.n_gallery,) for v in sims.values())

        # hybrid: contributions sum to the fused scores returned by the search
        for norm in ("zscore", "none"):
            res = eng.search_fused(img, {"clip": 0.7, "classical_concat": 0.3}, k=6, norm=norm, exclude_id=3)
            exps = ex.explain_results(sims, res.ids, res.weights, res.norm)
            assert np.allclose([e.fused for e in exps], res.scores, atol=1e-4), norm
            assert [e.gid for e in exps] == res.ids.tolist()
        print("ok: explanation contributions == fused scores from search_fused (zscore and none)")

        # single retriever search: weight 1 on it, others are supporting evidence
        res = eng.search_image(img, "color", k=5, exclude_id=3)
        exps = ex.explain_results(sims, res.ids, {"color": 1.0})
        assert all(abs(e.fused - next(r for r in e.rows if r["retriever"] == "color")["contribution"]) < 1e-6
                   for e in exps)
        assert all("classical_concat" not in [r["retriever"] for r in e.rows] for e in exps)
        print("ok: single-retriever explanation")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_numbers_and_reasons()
    test_visuals()
    test_engine_consistency()
    print("\nAll explain tests passed")
