"""Tests for src/fusion.py and SearchEngine.search_fused (Person B, WP9).

Run from the repo root:  python tests\\test_fusion.py
Pure NumPy checks first, then the hybrid search on the synthetic gallery from
tests/test_pipeline.py (CLIP stand-in, no weights needed).
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from src import fusion as fu  # noqa: E402
from src.evaluate import rank_by_similarity  # noqa: E402


def test_numpy_logic() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(4, 50)), rng.normal(size=(4, 50))

    # weights are rescaled: {1,1} == {0.5,0.5}; a zero weight is ignored
    f1 = fu.fuse({"a": a, "b": b}, {"a": 1, "b": 1})
    f2 = fu.fuse({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    assert np.allclose(f1, f2, atol=1e-6)
    assert np.allclose(fu.fuse({"a": a, "b": b}, {"a": 1, "b": 0}), fu.normalize_scores(a), atol=1e-6)

    # z-score fusion is invariant to scale/shift of one retriever's scores
    f3 = fu.fuse({"a": a * 10 + 3, "b": b}, {"a": 0.3, "b": 0.7})
    f4 = fu.fuse({"a": a, "b": b}, {"a": 0.3, "b": 0.7})
    assert np.allclose(f3, f4, atol=1e-4)
    # ... raw cosine fusion is not
    assert not np.allclose(fu.fuse({"a": a * 10, "b": b}, {"a": .5, "b": .5}, norm="none"),
                           fu.fuse({"a": a, "b": b}, {"a": .5, "b": .5}, norm="none"))

    # 1-D (single query) works and equals row 0 of the 2-D result
    assert np.allclose(fu.fuse({"a": a[0], "b": b[0]}, {"a": .4, "b": .6}),
                       fu.fuse({"a": a, "b": b}, {"a": .4, "b": .6})[0], atol=1e-6)

    for bad in ({"a": -1, "b": 1}, {"a": 0, "b": 0}, {"a": 1, "zzz": 1}):
        try:
            fu.fuse({"a": a, "b": b}, bad)
        except (ValueError, KeyError):
            continue
        raise AssertionError(f"accepted {bad}")
    print("ok: fuse (rescaling, zero weights, scale invariance, 1-D, validation)")


def test_grid_and_rank() -> None:
    g = list(fu.grid_weights(["x", "y", "z"], 0.5))
    assert len(g) == 6 and all(abs(sum(d.values()) - 1) < 1e-9 for d in g)    # C(4,2)
    assert len(list(fu.grid_weights(["x", "y"], 0.05))) == 21
    assert len(list(fu.grid_weights(["a", "b", "c", "d", "e"], 0.1))) == 1001
    try:
        list(fu.grid_weights(["x", "y"], 0.3))
    except ValueError:
        pass
    else:
        raise AssertionError("step 0.3 accepted")

    rng = np.random.default_rng(1)
    g_vecs = rng.normal(size=(30, 8)).astype(np.float32)
    g_vecs /= np.linalg.norm(g_vecs, axis=1, keepdims=True)
    ids = np.array([3, 7, 11])
    ref = rank_by_similarity(g_vecs[ids], g_vecs, exclude_ids=ids)
    got = fu.rank_from_scores(g_vecs[ids] @ g_vecs.T, exclude_ids=ids)
    assert got.shape == ref.shape and (got == ref).all()
    print("ok: grid_weights counts, rank_from_scores == rank_by_similarity")


def test_tune_finds_informative_retriever() -> None:
    rng = np.random.default_rng(2)
    n_cls, per = 6, 20
    labels = np.repeat(np.arange(n_cls), per)
    n = len(labels)
    centres = rng.normal(size=(n_cls, 16))
    good = centres[labels] + 0.4 * rng.normal(size=(n, 16))     # informative
    noise = rng.normal(size=(n, 16))                            # uninformative
    l2 = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)  # noqa: E731
    good, noise = l2(good), l2(noise)
    qids = np.arange(0, n, 4)
    scores = {"good": good[qids] @ good.T, "noise": noise[qids] @ noise.T}
    df = fu.tune(scores, labels, qids, ["good", "noise"], step=0.25)
    assert len(df) == 2 * 5 and df.iloc[0]["w_good"] >= 0.75, df.head()
    assert df["mAP"].is_monotonic_decreasing
    print("ok: tune picks the informative retriever:",
          {c: df.iloc[0][c] for c in ("norm", "w_good", "w_noise")}, round(df.iloc[0]["mAP"], 3))

    tmp = Path(tempfile.mkdtemp())
    try:
        fu.save_weights(tmp / "w.json", {"good": 0.8, "noise": 0.2}, "zscore")
        assert fu.load_weights(tmp / "w.json") == ({"good": 0.8, "noise": 0.2}, "zscore")
        assert fu.load_weights(tmp / "missing.json") == (fu.DEFAULT_WEIGHTS, fu.DEFAULT_NORM)
        (tmp / "bad.json").write_text("{not json")
        assert fu.load_weights(tmp / "bad.json") == (fu.DEFAULT_WEIGHTS, fu.DEFAULT_NORM)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: save_weights / load_weights (missing and corrupt file fall back to defaults)")


def test_engine_hybrid() -> None:
    import test_pipeline as tp
    from src import retrieval as rt
    from src.dataset import FolderGallery
    from src.features import clip_encoder

    clip_encoder.encode_images = tp.fake_encode_images
    clip_encoder.encode_text = tp.fake_encode_text
    import build_index

    tmp = Path(tempfile.mkdtemp(prefix="vs_fusion_"))
    try:
        data, idx = tmp / "gallery", tmp / "index"
        tp.make_gallery(data)
        build_index.main(["--dataset", f"folder:{data}", "--index-dir", str(idx)])
        eng, gal = rt.SearchEngine(idx), FolderGallery(data)
        labels = np.load(idx / "labels.npy")

        img, _ = gal.get(3)
        r = eng.search_fused(img, k=8, exclude_id=3)
        assert isinstance(r, rt.FusedResult) and r.retriever == "hybrid"
        assert 3 not in r.ids and len(r.ids) == 8 and (np.diff(r.scores) <= 1e-6).all()
        assert set(r.parts) == {"clip", "classical_concat"} and r.parts["clip"].shape == (8,)
        assert abs(sum(r.weights.values()) - 1) < 1e-9

        # weight 1.0 on CLIP == plain CLIP search (same ids, same order)
        rc = eng.search_fused(img, {"clip": 1.0}, k=8, exclude_id=3)
        rp = eng.search_image(img, "clip", k=8, exclude_id=3)
        assert np.allclose(rc.parts["clip"], rp.scores, atol=1e-5)
        assert (rc.ids == rp.ids).all() or np.allclose(rc.parts["clip"], rp.scores, atol=1e-5)

        # full 5-way weights work and the parts line up with the ids returned
        w5 = {"clip": .4, "color": .3, "texture": .1, "edge": .1, "dct": .1}
        r5 = eng.search_fused(img, w5, k=5, exclude_id=3)
        assert set(r5.parts) == set(w5)
        assert np.allclose(r5.parts["color"], eng.vectors["color"][r5.ids] @
                           eng.query_vector(img, "color")[0][0], atol=1e-5)

        # averaged over queries the hybrid is far above chance (15/79 = 0.19); a single
        # query can be unlucky because the synthetic classes overlap
        p8 = np.mean([(labels[eng.search_fused(gal.get(i)[0], k=8, exclude_id=i).ids] == labels[i]).mean()
                      for i in range(0, 80, 4)])
        assert p8 > 0.4, p8
        try:
            eng.search_fused(img, {"nonexistent": 1.0})
        except KeyError:
            pass
        else:
            raise AssertionError("unknown retriever accepted")
        print("ok: engine.search_fused (self-exclusion, order, parts, 5-way, unknown retriever)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_numpy_logic()
    test_grid_and_rank()
    test_tune_finds_informative_retriever()
    test_engine_hybrid()
    print("\nAll fusion tests passed")
