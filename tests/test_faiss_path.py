"""Exercises the real-FAISS code path of src/index.py with a strict fake `faiss` (tests/fake_faiss.py),
because most test environments only ever run the NumPy fallback.  Run: python tests/test_faiss_path.py"""

import importlib
import io
import contextlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import fake_faiss  # noqa: E402


def reload_index(with_faiss: bool):
    if with_faiss:
        sys.modules["faiss"] = fake_faiss
    else:
        sys.modules["faiss"] = None                                   # makes `import faiss` fail
    import src.index as ix
    return importlib.reload(ix)


def same_modulo_ties(a, b) -> bool:
    """Equal scores everywhere; equal ids wherever the score is unique (exact ties may be ordered differently
    by FAISS and by NumPy, and may be cut at k differently, which is not a bug)."""
    if not np.allclose(a.scores, b.scores, atol=1e-5):
        return False
    for i, sc in enumerate(a.scores):
        if np.sum(np.isclose(a.scores, sc, atol=1e-6)) == 1 and a.ids[i] != b.ids[i]:
            return False
    return True


def main() -> None:
    import test_pipeline as tp
    from src.features import clip_encoder

    clip_encoder.encode_images = tp.fake_encode_images
    clip_encoder.encode_text = tp.fake_encode_text
    tmp = Path(tempfile.mkdtemp(prefix="vs_faiss_"))
    try:
        ix = reload_index(True)
        assert ix.HAVE_FAISS
        rng = np.random.default_rng(0)
        v = rng.normal(size=(40, 16)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        # build / search accept anything the project passes (float64, non-contiguous) and return the right shapes
        index = ix.build_index(v.astype(np.float64)[:, ::1])
        assert index.ntotal == 40 and index.d == 16
        sc, ids = ix.search(index, v[:3].T.copy().T, 5)               # a transposed (F-ordered) view is made contiguous
        assert sc.shape == ids.shape == (3, 5) and (ids[:, 0] == [0, 1, 2]).all()
        sc2, ids2 = ix.search(index, v[:2], 999)                      # k above ntotal is clamped: no -1 padding leaks out
        assert ids2.shape == (2, 40) and (ids2 >= 0).all()
        print("ok: build_index / search through the faiss API (dtype + contiguity enforced, k clamped)")

        # save / load, and the stale-index guard
        p = tmp / "x.faiss"
        ix.save_index(index, p)
        assert p.exists() and ix.load_index(p, v).ntotal == 40
        assert ix.load_index(p).ntotal == 40
        assert ix.load_index(p, v[:25]).ntotal == 25                  # stale file (different N) -> rebuilt, not used
        assert ix.load_index(p, v[:, :8].copy()).d == 8               # stale file (different D) -> rebuilt
        assert ix.load_index(tmp / "missing.faiss", v).ntotal == 40
        try:
            ix.load_index(tmp / "missing.faiss")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing index without vectors accepted")
        print("ok: save / load, stale .faiss (N or D changed) is rebuilt, missing file handled")

        # the whole pipeline with faiss active: build_index script, SearchEngine, hybrid, explanations
        import build_index
        from src import retrieval as rt
        from src.dataset import FolderGallery
        data, idx = tmp / "gallery", tmp / "index"
        tp.make_gallery(data)
        with contextlib.redirect_stdout(io.StringIO()):
            build_index.main(["--dataset", f"folder:{data}", "--index-dir", str(idx)])
        assert len(list(idx.glob("*.faiss"))) == 6, sorted(p.name for p in idx.glob("*.faiss"))
        importlib.reload(rt)
        eng, gal = rt.SearchEngine(idx), FolderGallery(data)
        img, _ = gal.get(3)
        with_faiss = {r: eng.search_image(img, r, k=8, exclude_id=3) for r in ("clip", "color", "classical_concat")}
        hybrid = eng.search_fused(img, k=8, exclude_id=3)
        assert all(3 not in r.ids and len(r.ids) == 8 for r in with_faiss.values())

        # identical results with the NumPy fallback
        reload_index(False)
        importlib.reload(rt)
        import src.index as ix2
        assert not ix2.HAVE_FAISS
        eng2 = rt.SearchEngine(idx)
        for r, res in with_faiss.items():
            ref = eng2.search_image(img, r, k=8, exclude_id=3)
            assert same_modulo_ties(res, ref), r
        assert same_modulo_ties(eng2.search_fused(img, k=8, exclude_id=3), hybrid)
        print("ok: build_index + SearchEngine with faiss == NumPy fallback (3 retrievers + hybrid)")
        print("\nAll faiss-path tests passed")
    finally:
        sys.modules.pop("faiss", None)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
