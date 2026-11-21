"""Headless UI smoke test for app/app.py using Streamlit's AppTest (no browser).

Run from the repo root:  python tests\\test_app.py
Uses the same synthetic gallery + CLIP stand-in as tests/test_pipeline.py, plus a synthetic
product catalogue (tests/demo_catalog.py) for the dataset switch and the metadata filters.
The synthetic gallery stands in for Flowers-102 (src.dataset.Flowers102Gallery is patched), so
the adaptive toggle can be exercised without the real dataset.
File upload is not covered by AppTest; try it by hand in the browser.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

tmp = Path(tempfile.mkdtemp(prefix="vs_app_"))
os.environ["VISIONSEEK_INDEX_DIR"] = str(tmp / "index")
os.environ["VISIONSEEK_DATASET"] = "flowers102"                   # served by the synthetic gallery below
os.environ["VISIONSEEK_DEMO_INDEX"] = str(tmp / "index_demo_cat")   # the synthetic catalogue
os.environ["VISIONSEEK_RESULTS_DIR"] = str(ROOT / "results")

import test_pipeline as tp  # noqa: E402
from src.features import clip_encoder  # noqa: E402

clip_encoder.encode_images = tp.fake_encode_images
clip_encoder.encode_text = tp.fake_encode_text
import build_index  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_adaptive_modules() -> None:
    """Person A's request: noisy queries -> bucket 'noisy' and CLIP 0.85 / classical 0.15;
    clean queries -> 0.95 / 0.05. Uses real Flowers-102 test images when the dataset is available
    (the quality thresholds were calibrated on them) and checks a share of 20 images, because a few
    real photos are legitimately a little blurry or dark."""
    import numpy as np
    try:
        from src.adaptive import adaptive_weights
        from src.quality import analyze
    except ImportError:
        print("skip: src.adaptive / src.quality not found (merge Person A's feat/adaptive first)")
        return
    try:
        from src.dataset import DATA_DIR, Flowers102Gallery
        gal = Flowers102Gallery(DATA_DIR)
        imgs = [gal.get(g)[0] for g in range(2040, 2060)]              # first 20 test images
    except Exception as exc:  # no torchvision / dataset on this machine
        print(f"skip: real Flowers-102 images not available ({type(exc).__name__}: {exc})")
        return
    from src.degrade import degrade

    def clip_share(img):
        w = adaptive_weights(analyze(img))
        return round(w["clip"] / (w["clip"] + w["classical_concat"]), 3)

    noisy = [degrade(im, "noise", 50, seed=i) for i, im in enumerate(imgs)]
    buckets = [analyze(im)["bucket"] for im in noisy]
    assert buckets.count("noisy") >= 18, buckets
    assert all(clip_share(im) == 0.85 for im, b in zip(noisy, buckets) if b == "noisy")
    clean_b = [analyze(im)["bucket"] for im in imgs]
    clean_w = [clip_share(im) for im in imgs]
    assert clean_w.count(0.95) >= 14, (clean_b, clean_w)
    print(f"ok: adaptive modules on real images: sigma-50 noise -> {buckets.count('noisy')}/20 'noisy' (0.85/0.15); "
          f"clean -> {clean_w.count(0.95)}/20 at 0.95/0.05 (buckets {sorted(set(clean_b))})")


def main() -> None:
    try:
        test_adaptive_modules()                                           # before Flowers102Gallery is patched
        import src.dataset as ds
        from demo_catalog import make_catalog

        tp.make_gallery(tmp / "gallery")
        ds.Flowers102Gallery = lambda root=None: ds.FolderGallery(tmp / "gallery")    # synthetic stand-in
        build_index.main(["--dataset", f"folder:{tmp / 'gallery'}", "--index-dir", str(tmp / "index")])
        make_catalog(tmp)                                                 # -> tmp / "index_demo_cat"

        at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=60).run()
        assert not at.exception, at.exception
        assert at.title[0].value.startswith("VisionSeek")

        # Image mode, random gallery query, every retriever
        at.radio(key="source").set_value("Random test image from the gallery").run()
        assert not at.exception, at.exception
        for name in ("clip", "classical_concat", "color", "texture", "edge", "dct", "hybrid"):
            at.selectbox(key="retriever").set_value(name).run()
            assert not at.exception, (name, at.exception)
            assert any("Top" in s.value for s in at.subheader), name
            assert any(s.value == "Why these results?" for s in at.subheader), name
        print("ok: image mode, all 6 retrievers + hybrid")

        at.selectbox(key="inspect").set_value(3).run()
        assert not at.exception, at.exception
        print("ok: explanation panel, inspect result #3")

        # Degrade + restore the query
        at.selectbox(key="kind").set_value("noise").run()
        at.selectbox(key="rest").set_value("median 3x3").run()
        assert not at.exception, at.exception
        print("ok: degrade + restore")

        # Duplicates tab: default source is a gallery image; try every edit and the gallery scan
        for edit in ("none (exact copy)", "resize", "jpeg", "crop", "brightness"):
            at.selectbox(key="dup_edit").set_value(edit).run()
            assert not at.exception, (edit, at.exception)
        at.button(key="dup_scan").click().run()
        assert not at.exception, at.exception
        print("ok: duplicates tab (all edits, gallery scan)")

        # Adaptive toggle (needs Person A's modules; the stand-in dataset is served as flowers102)
        at.selectbox(key="retriever").set_value("hybrid").run()
        if at.checkbox(key="use_adaptive").disabled:
            print("skip: adaptive toggle is disabled (Person A's src.adaptive / src.quality not merged yet)")
        else:
            at.checkbox(key="use_adaptive").check().run()
            assert not at.exception, at.exception
            assert at.slider(key="w_clip").disabled, "manual slider must be disabled while adaptive is on"
            assert any(c.value.startswith("Query quality") for c in at.caption), [c.value for c in at.caption]
            at.checkbox(key="use_adaptive").uncheck().run()
            assert not at.exception, at.exception
            print("ok: adaptive toggle (bucket + weights shown, manual slider disabled)")

        # Flowers-like filters: category and approximate colour
        at.multiselect(key="f_cls_0").set_value([1, 2]).run()
        assert not at.exception, at.exception
        assert any("Top" in s.value for s in at.subheader)
        at.multiselect(key="f_cls_0").set_value([]).run()
        at.selectbox(key="f_col_0").set_value("black").run()
        assert not at.exception, at.exception
        assert any("No gallery image matches" in w.value for w in at.warning)
        at.selectbox(key="f_col_0").set_value("any").run()
        print("ok: category filter, colour filter, empty-result warning")

        # Dataset switch -> product catalogue: exact metadata filters, adaptive off, product names
        at.selectbox(key="dataset").set_value("Fashion demo").run()
        assert not at.exception, at.exception
        assert any("Top" in s.value for s in at.subheader)
        assert at.checkbox(key="use_adaptive").disabled, "adaptive must be off for the catalogue"
        at.multiselect(key="f_baseColour_1").set_value(["Red", "Blue"]).run()
        at.multiselect(key="f_gender_1").set_value(["Men"]).run()
        assert not at.exception, at.exception
        assert any("Top" in s.value for s in at.subheader)
        at.multiselect(key="f_articleType_1").set_value(["Watches"]).run()
        at.multiselect(key="f_baseColour_1").set_value(["White"]).run()
        at.multiselect(key="f_gender_1").set_value(["Women"]).run()
        assert any("No gallery image matches" in w.value for w in at.warning)
        for key in ("f_baseColour_1", "f_gender_1", "f_articleType_1"):
            at.multiselect(key=key).set_value([]).run()
        at.selectbox(key="retriever").set_value("clip").run()
        at.selectbox(key="dup_edit").set_value("crop").run()              # duplicates tab on the catalogue
        assert not at.exception, at.exception
        at.radio(key="mode").set_value("Text").run()
        at.text_input(key="text").set_value("red watch").run()
        assert not at.exception, at.exception
        assert any("Top" in s.value for s in at.subheader)
        at.radio(key="mode").set_value("Image").run()
        at.selectbox(key="dataset").set_value("Flowers-102 (benchmark)").run()
        assert not at.exception, at.exception
        print("ok: dataset switch (catalogue: metadata filters, empty result, text search, adaptive off) and back")

        # Text mode
        at.radio(key="mode").set_value("Text").run()
        at.text_input(key="text").set_value("a red flower").run()
        assert not at.exception, at.exception
        assert any("Top" in s.value for s in at.subheader)
        print("ok: text mode")

        # Image + text mode
        at.radio(key="mode").set_value("Image + text").run()
        assert not at.exception, at.exception
        print("ok: image + text mode")
        print("\nApp smoke test passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
