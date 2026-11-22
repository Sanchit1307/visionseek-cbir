"""Tests for src/filters.py, CatalogGallery and the dataset registry (Person B, WP13b).

Run from the repo root:  python tests/test_filters.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from src import filters as fl  # noqa: E402
from src.features.classical import color_feature  # noqa: E402


def solid(h: int, s: int, v: int) -> np.ndarray:
    return cv2.cvtColor(np.full((64, 64, 3), (h, s, v), np.uint8), cv2.COLOR_HSV2BGR)


def test_hsv_colours() -> None:
    assert len(fl.bin_colour_names()) == 256
    cases = {"red": (0, 255, 255), "orange": (17, 255, 255), "yellow": (30, 255, 255),
             "green": (60, 255, 200), "cyan": (90, 255, 255), "blue": (120, 255, 255),
             "purple": (140, 255, 255), "pink": (155, 200, 255),
             "white": (0, 0, 255), "grey": (0, 0, 128), "black": (0, 0, 10)}
    assert set(cases) == set(fl.COLOUR_NAMES)
    for want, hsv in cases.items():
        sh = fl.colour_shares(color_feature(solid(*hsv))[None])[0]
        got = fl.COLOUR_NAMES[int(sh.argmax())]
        assert got == want and sh.max() > 0.95 and abs(sh.sum() - 1) < 1e-9, (want, got, sh.round(2))
    two = np.hstack([solid(0, 255, 255), solid(120, 255, 255)])
    sh = fl.colour_shares(color_feature(two)[None])[0]
    assert abs(sh[0] - 0.5) < 0.05 and abs(sh[fl.COLOUR_NAMES.index("blue")] - 0.5) < 0.05
    print("ok: HSV colour naming (11 solid colours, a two-colour image)")


def test_masks_and_results() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    shares = np.zeros((6, len(fl.COLOUR_NAMES)))
    shares[[0, 2, 4], fl.COLOUR_NAMES.index("red")] = 0.5
    assert fl.filter_mask(6) is None and fl.filter_mask(6, labels, [], shares, "any") is None
    assert fl.filter_mask(6, labels, [1]).tolist() == [False, False, True, True, False, False]
    assert fl.filter_mask(6, labels, None, shares, "red", 0.3).tolist() == [True, False, True, False, True, False]
    assert fl.filter_mask(6, labels, [0, 1], shares, "red", 0.3).tolist() == [True, False, True, False, False, False]
    for bad in (dict(colour="mauve", shares=shares), dict(colour="red"), dict(classes=[1])):
        try:
            fl.filter_mask(6, None, **bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")

    meta = pd.DataFrame({"gender": ["Men", "Men", "Women", "Women", None, "Men"],
                         "baseColour": ["Red", "Blue", "Red", "Black", "Red", None]})
    assert fl.value_counts(meta, "gender")[0] == ("Men", 3) and ("(missing)", 1) in fl.value_counts(meta, "gender")
    assert fl.metadata_mask(meta, {}) is None and fl.metadata_mask(meta, {"gender": []}) is None
    assert fl.metadata_mask(meta, {"gender": ["Men"]}).tolist() == [True, True, False, False, False, True]
    assert fl.metadata_mask(meta, {"baseColour": ["Red", "Blue"]}).tolist() == [True, True, True, False, True, False]
    both = fl.metadata_mask(meta, {"gender": ["Men", "Women"], "baseColour": ["Red"]})   # OR inside, AND across
    assert both.tolist() == [True, False, True, False, False, False]
    assert fl.metadata_mask(meta, {"baseColour": ["(missing)"]}).tolist() == [False] * 5 + [True]
    try:
        fl.metadata_mask(meta, {"nope": ["x"]})
    except KeyError:
        pass
    else:
        raise AssertionError("unknown column accepted")

    from src.retrieval import FusedResult, SearchResult
    mask = np.array([True, False, True, False, True, False])
    ids, sc = np.array([5, 4, 3, 2, 1, 0]), np.array([.9, .8, .7, .6, .5, .4], np.float32)
    r = fl.filter_result(SearchResult("clip", ids, sc, 1.0, 2.0), mask, 2)
    assert r.ids.tolist() == [4, 2] and np.allclose(r.scores, [.8, .6]) and r.extract_ms == 1.0
    f = fl.filter_result(FusedResult("hybrid", ids, sc, 1.0, 2.0, parts={"clip": sc * 2},
                                     weights={"clip": 1.0}, norm="zscore"), mask, 10)
    assert f.ids.tolist() == [4, 2, 0] and np.allclose(f.parts["clip"], [1.6, 1.2, 0.8]) and f.norm == "zscore"
    # minimum similarity: raw cosine for a plain result, CLIP cosine for a fused one
    sc = np.array([.9, .7, .5, .3], np.float32)
    plain = SearchResult("clip", np.array([4, 3, 2, 1]), sc, 1.0, 2.0)
    assert fl.min_similarity(plain, 0) is plain and fl.min_similarity(plain, -1) is plain
    assert fl.min_similarity(plain, 0.6).ids.tolist() == [4, 3] and len(fl.min_similarity(plain, 0.95).ids) == 0
    fused = FusedResult("hybrid", np.array([4, 3, 2, 1]), np.array([3., 2., 1., 0.], np.float32), 1.0, 2.0,
                        parts={"clip": np.array([.8, .4, .6, .2], np.float32), "classical_concat": np.array([.9] * 4, np.float32)},
                        weights={"clip": .9, "classical_concat": .1}, norm="none")
    ff = fl.min_similarity(fused, 0.5)                          # filters on the CLIP cosine, keeps order
    assert ff.ids.tolist() == [4, 2] and np.allclose(ff.scores, [3., 1.]) and np.allclose(ff.parts["clip"], [.8, .6])
    print("ok: class / HSV-colour masks, metadata masks (OR within, AND across, missing values), filter_result, min_similarity")


def test_catalog_gallery() -> None:
    import test_pipeline as tp
    from demo_catalog import make_catalog
    from src import dataset as ds
    from src.features import clip_encoder

    clip_encoder.encode_images = tp.fake_encode_images
    clip_encoder.encode_text = tp.fake_encode_text
    tmp = Path(tempfile.mkdtemp(prefix="vs_catalog_"))
    try:
        idx, flat, meta = make_catalog(tmp)
        g = ds.open_gallery(f"catalog:{idx}")
        labels = np.load(idx / "labels.npy")
        assert len(g) == len(meta) == 48 and g.class_names == sorted(meta.articleType.unique())
        assert all(g.class_names[labels[i]] == meta.articleType[i] for i in range(len(g)))
        img, lab = g.get(5)
        assert img.shape == (80, 60, 3) and lab == labels[5]
        assert cv2.imread(str(flat / f"{meta.id[5]}.jpg")).shape == img.shape
        d = g.describe(0)
        assert meta.productDisplayName[0] in d and "(" in d
        nan_row = int(meta.baseColour.isna().idxmax())
        assert "(" not in g.describe(nan_row)                    # no "(nan)" for a missing colour
        print("ok: CatalogGallery (rows, class names from articleType, 60x80 images, captions)")

        # variants of the layout the teammate might have written
        info = json.loads((idx / "meta.json").read_text())
        meta.drop(columns=["id"]).to_csv(idx / "metadata.csv", index=False)
        try:
            ds.open_gallery(f"catalog:{idx}")
        except ValueError as e:
            assert "id" in str(e) and "columns" in str(e)
        else:
            raise AssertionError("missing id column accepted")
        (idx / "meta.json").write_text(json.dumps({**info, "ids": [int(i) for i in meta.id]}))
        assert ds.open_gallery(f"catalog:{idx}").get(7)[0].shape == (80, 60, 3)      # ids from meta.json
        meta.to_csv(idx / "metadata.csv", index=False)

        (idx / "meta.json").write_text(json.dumps({"image_root": "/does/not/exist/images"}))
        try:
            ds.open_gallery(f"catalog:{idx}")
        except FileNotFoundError as e:
            assert "VISIONSEEK_DEMO_IMAGES" in str(e)
        else:
            raise AssertionError("missing image folder accepted")
        real_data_dir = ds.DATA_DIR
        ds.DATA_DIR = tmp / "data"                                           # default Kaggle location
        (ds.DATA_DIR / "fashion").mkdir(parents=True)
        shutil.copytree(flat, ds.DATA_DIR / "fashion" / "images")
        assert ds.open_gallery(f"catalog:{idx}").get(3)[0].shape == (80, 60, 3)    # absolute root missing -> fallback
        ds.DATA_DIR = real_data_dir
        os.environ["VISIONSEEK_DEMO_IMAGES"] = str(flat)
        assert ds.open_gallery(f"catalog:{idx}").get(0)[0].shape == (80, 60, 3)      # env override
        del os.environ["VISIONSEEK_DEMO_IMAGES"]
        shutil.copytree(flat, idx / "imgs")
        (idx / "meta.json").write_text(json.dumps({"image_root": "imgs"}))            # relative root
        assert ds.open_gallery(f"catalog:{idx}").get(0)[0].shape == (80, 60, 3)
        meta.iloc[:-3].to_csv(idx / "metadata.csv", index=False)
        try:
            ds.open_gallery(f"catalog:{idx}")
        except ValueError as e:
            assert "row i must match vector i" in str(e)
        else:
            raise AssertionError("row/vector mismatch accepted")
        meta.to_csv(idx / "metadata.csv", index=False)
        print("ok: loader errors are explicit (no id column, missing image folder, row mismatch); "
              "ids from meta.json, env override, relative image_root and the data/fashion/images fallback work")

        # filtered engine search on the catalogue index
        from src import retrieval as rt
        eng = rt.SearchEngine(idx)
        gal = ds.open_gallery(f"catalog:{idx}")
        mask = fl.metadata_mask(gal.meta, {"baseColour": ["Red", "Blue"], "gender": ["Men"]})
        img, _ = gal.get(2)
        for res in (eng.search_image(img, "color", k=eng.n_gallery, exclude_id=2),
                    eng.search_fused(img, k=eng.n_gallery, exclude_id=2)):
            out = fl.filter_result(res, mask, 6)
            assert len(out.ids) == 6 and mask[out.ids].all() and 2 not in out.ids
            assert (gal.meta.baseColour.iloc[out.ids].isin(["Red", "Blue"])).all()
        print("ok: filtered search on the catalogue index (exact colour + gender)")

        os.environ["VISIONSEEK_DEMO_INDEX"] = str(idx)
        assert list(ds.available_datasets("flowers102")) == ["Flowers-102 (benchmark)", "Fashion demo"]
        assert ds.available_datasets("flowers102")["Fashion demo"] == (f"catalog:{idx}", idx)
        os.environ["VISIONSEEK_DEMO_INDEX"] = str(tmp / "nowhere")
        assert list(ds.available_datasets("flowers102")) == ["Flowers-102 (benchmark)"]
        print("ok: available_datasets (Fashion demo only when labels.npy and metadata.csv exist)")
    finally:
        os.environ.pop("VISIONSEEK_DEMO_INDEX", None)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_hsv_colours()
    test_masks_and_results()
    test_catalog_gallery()
    print("\nAll filter tests passed")
