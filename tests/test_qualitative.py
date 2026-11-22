"""Test for scripts/make_qualitative.py (Person B). Run from the repo root: python tests/test_qualitative.py"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

_TMP = Path(tempfile.mkdtemp(prefix="vs_qual_"))
os.environ["VISIONSEEK_RESULTS_DIR"] = str(_TMP / "results")          # before src.dataset is imported


def main() -> None:
    import test_pipeline as tp
    from src.features import clip_encoder

    clip_encoder.encode_images = tp.fake_encode_images
    clip_encoder.encode_text = tp.fake_encode_text
    import build_index
    import make_qualitative as mq

    for text, want in (("clean", ("clean", None)), ("noise:50", ("noise", 50)), ("jpeg:10", ("jpeg", 10))):
        assert mq.parse_condition(text) == want
    for bad in ("noise", "noise:x", "sharpen:3", "blur:"):
        try:
            mq.parse_condition(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")

    data, idx = _TMP / "gallery", _TMP / "index"
    tp.make_gallery(data)
    build_index.main(["--dataset", f"folder:{data}", "--index-dir", str(idx)])
    mq.main(["--dataset", f"folder:{data}", "--index-dir", str(idx), "--n", "2", "--k", "4",
             "--conditions", "clean,noise:25,blur:9"])
    for slug in ("clean", "noise25", "blur9"):
        p = _TMP / "results" / f"qualitative_{slug}.png"
        assert p.exists() and p.stat().st_size > 5000, p
    try:
        mq.main(["--dataset", f"folder:{data}", "--index-dir", str(idx), "--retrievers", "clip,nonexistent"])
    except SystemExit as e:
        assert "nonexistent" in str(e)
    else:
        raise AssertionError("unknown retriever accepted")
    print("ok: make_qualitative.py (3 conditions drawn, bad conditions and unknown retriever rejected)")
    print("\nAll qualitative tests passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
