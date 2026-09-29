"""Headless UI smoke test for app/app.py using Streamlit's AppTest (no browser).

Run from the repo root:  python tests\\test_app.py
Uses the same synthetic gallery + CLIP stand-in as tests/test_pipeline.py.
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
os.environ["VISIONSEEK_DATASET"] = f"folder:{tmp / 'gallery'}"
os.environ["VISIONSEEK_RESULTS_DIR"] = str(ROOT / "results")

import test_pipeline as tp  # noqa: E402
from src.features import clip_encoder  # noqa: E402

clip_encoder.encode_images = tp.fake_encode_images
clip_encoder.encode_text = tp.fake_encode_text
import build_index  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402


def main() -> None:
    try:
        tp.make_gallery(tmp / "gallery")
        build_index.main(["--dataset", f"folder:{tmp / 'gallery'}", "--index-dir", str(tmp / "index")])

        at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=60).run()
        assert not at.exception, at.exception
        assert at.title[0].value.startswith("VisionSeek")

        # Image mode, random gallery query, every retriever
        at.radio(key="source").set_value("Random test image from the gallery").run()
        assert not at.exception, at.exception
        for name in ("clip", "classical_concat", "color", "texture", "edge", "dct"):
            at.selectbox(key="retriever").set_value(name).run()
            assert not at.exception, (name, at.exception)
            assert any("Top" in s.value for s in at.subheader), name
        print("ok: image mode, all 6 retrievers")

        # Degrade + restore the query
        at.selectbox(key="kind").set_value("noise").run()
        at.selectbox(key="rest").set_value("median 3x3").run()
        assert not at.exception, at.exception
        print("ok: degrade + restore")

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
