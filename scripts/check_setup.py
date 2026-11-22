"""Is this clone complete? Lists missing files / packages with the fix (Person B).

Run from the repo root:   python scripts/check_setup.py
Exit code 0 = everything the app needs is there, 1 = something required is missing.
It does not import torch / streamlit, so it is fast and safe to run before anything else.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PERSON_B = [
    "app/app.py", "src/dataset.py", "src/index.py", "src/retrieval.py", "src/fusion.py",
    "src/explain.py", "src/duplicates.py", "src/filters.py", "src/features/clip_encoder.py",
    "scripts/build_index.py", "scripts/eval_clip.py", "scripts/make_comparison.py",
    "scripts/tune_fusion.py", "scripts/eval_duplicates.py", "scripts/make_qualitative.py",
]
PERSON_A = ["src/degrade.py", "src/evaluate.py", "src/features/classical.py",
            "scripts/baseline_classical.py"]
PERSON_A_ADAPTIVE = ["src/adaptive.py", "src/quality.py", "src/quality_thresholds.json",
                     "results/adaptive_weights.json"]                     # branch feat/adaptive
INDEX_REQUIRED = ["clip_image.npy", "labels.npy", "query_ids.npy"] + \
                 [f"classical_{n}.npy" for n in ("color", "texture", "edge", "dct")]
INDEX_OPTIONAL = {"phash.npy": "python scripts/eval_duplicates.py",
                  "demo/metadata.csv": "Person A: python scripts/prepare_demo_dataset.py"}
RESULTS_OPTIONAL = {"fusion_weights.json": "python scripts/tune_fusion.py",
                    "duplicate_thresholds.json": "python scripts/eval_duplicates.py",
                    "comparison_table.csv": "python scripts/make_comparison.py"}
PACKAGES_REQUIRED = {"numpy": "numpy", "cv2": "opencv-python", "skimage": "scikit-image", "pandas": "pandas",
                     "matplotlib": "matplotlib", "PIL": "pillow", "tqdm": "tqdm", "torch": "torch",
                     "torchvision": "torchvision", "open_clip": "open_clip_torch", "streamlit": "streamlit"}
PACKAGES_OPTIONAL = {"faiss": "faiss-cpu (without it the app uses an exact NumPy search, same results)"}


def main() -> int:
    missing_required = 0

    def section(title: str, names, base: Path, required: bool, hints: dict | None = None) -> None:
        nonlocal missing_required
        print(f"\n{title}")
        for n in names:
            ok = (base / n).exists()
            tag = "ok     " if ok else ("MISSING" if required else "absent ")
            hint = "" if ok else (f"   <- {hints[n]}" if hints and n in hints else "")
            print(f"  {tag} {n}{hint}")
            if required and not ok:
                missing_required += 1

    print(f"Repo: {ROOT}\nPython: {sys.version.split()[0]} (the project targets 3.10-3.12)")
    section("Person B files (unzip the consolidated package over the repo)", PERSON_B, ROOT, True)
    section("Person A files (merge Person A's PRs into main)", PERSON_A, ROOT, True)
    section("Person A adaptive fusion (optional: app runs without it, the toggle is hidden)",
            PERSON_A_ADAPTIVE, ROOT, False)
    section("index/ (gitignored: build with scripts/build_index.py or copy from a teammate)",
            INDEX_REQUIRED, ROOT / "index", True)
    section("index/ optional", list(INDEX_OPTIONAL), ROOT / "index", False, INDEX_OPTIONAL)
    section("results/ optional (the app falls back to placeholder defaults)", list(RESULTS_OPTIONAL),
            ROOT / "results", False, RESULTS_OPTIONAL)

    print("\nPython packages")
    for mod, pip in PACKAGES_REQUIRED.items():
        ok = importlib.util.find_spec(mod) is not None
        print(f"  {'ok     ' if ok else 'MISSING'} {mod}" + ("" if ok else f"   <- pip install {pip}"))
        missing_required += 0 if ok else 1
    for mod, note in PACKAGES_OPTIONAL.items():
        ok = importlib.util.find_spec(mod) is not None
        print(f"  {'ok     ' if ok else 'absent '} {mod}" + ("" if ok else f"   <- {note}"))

    print("\n" + ("Everything required is present. Start the app with:  streamlit run app/app.py"
                  if not missing_required else
                  f"{missing_required} required item(s) missing (see MISSING above). "
                  "Start the app with `streamlit run app/app.py`, not `python app/app.py`."))
    return 1 if missing_required else 0


if __name__ == "__main__":
    sys.exit(main())
