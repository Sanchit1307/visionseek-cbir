# VisionSeek: adaptive multimodal image retrieval and robust visual search

Course project (Image Processing). An image-search engine that combines classical image-processing
descriptors (HSV, LBP, edge orientation, block DCT) with CLIP embeddings, analyses the quality of the
query image, can adapt its fusion weights to that quality, explains every result, detects near-duplicates,
and is evaluated under noise, blur, JPEG and brightness/contrast damage.

Search modes: image to image, text to image, image + text to image. Two galleries: Oxford Flowers-102
(benchmark) and a small fashion-product catalogue (demo).

App features: six retrievers plus the CLIP + classical hybrid (manual or quality-adaptive weights), top-K and
minimum-similarity controls, category / colour filters (exact metadata on the catalogue), a query-quality panel,
a robustness demo (degrade and restore the query, optionally side by side with the original), a per-result
explanation ("Why these results?"), a near-duplicate tab, and a results tab with the committed experiments.

## Results at a glance
Protocol: gallery = all 8,189 Flowers-102 images, queries = 500 fixed test images (seed 42), the query is
excluded from its own results, relevant = same class. Latency is per query on CPU, from `results/`.

| Retriever | Dim | P@10 | mAP | Total ms |
|---|---|---|---|---|
| HSV colour | 256 | 0.344 | 0.155 | 1.1 |
| LBP texture | 90 | 0.122 | 0.047 | 3.0 |
| Edge orientation | 81 | 0.070 | 0.034 | 1.8 |
| Block DCT | 72 | 0.091 | 0.048 | 1.2 |
| Classical (all four) | 499 | 0.375 | 0.163 | 6.6 |
| CLIP ViT-B/32 | 512 | 0.880 | 0.624 | 106.8 |

* Hybrid (score fusion of CLIP and classical), weights tuned on the validation split: mAP 0.6251
  against 0.6242 for CLIP alone on clean queries (a negligible difference).
  A fixed 0.7 / 0.3 hybrid is worse (0.5661). On damaged queries the tuned hybrid is ahead of CLIP
  alone (0.5416 vs 0.5370 mean mAP).
* Robustness: at noise sigma 50 CLIP keeps 72% of its clean mAP, the classical concatenation 58%.
  Restoration before feature extraction (median / Gaussian filter) helps the classical descriptors under noise
  (a 5x5 median recovers 37% to 40% of the lost mAP at sigma 50 and 25) but does not help CLIP, which is
  slightly worse after filtering (`results/restoration.csv`).
* Adaptive fusion (weights chosen from the query's quality bucket) vs the fixed hybrid: no difference on clean
  queries, a small gain under noise and blur (sigma 50: +0.0134 mAP) and no measurable
  difference elsewhere; see `results/adaptive_significance.csv` for the bootstrap intervals.
* Fashion demo (8 article types, 300 queries): CLIP P@10 0.914, zero-shot type accuracy
  91.3%. The HSV colour descriptor matches the catalogue's base colour
  only 47% of the time, so the demo filters on the exact `baseColour` column.

## Quick start
Python 3.10 to 3.12. Run everything from the repository root.
```bash
python3.10 -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -c constraints.txt numpy scipy==1.13.1 pandas==2.2.2 matplotlib==3.9.2 pillow==10.4.0 tqdm
pip install -c constraints.txt opencv-python==4.10.0.84 scikit-image==0.24.0
pip install -c constraints.txt torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -c constraints.txt open_clip_torch faiss-cpu streamlit
python scripts/check_setup.py            # lists anything missing, with the fix
python scripts/build_index.py            # downloads Flowers-102 + CLIP weights, builds data/ and index/
streamlit run app/app.py                 # not: python app/app.py
```
`numpy` must stay at 1.26.4 (`constraints.txt`). `faiss-cpu` is optional: without it the app uses an exact NumPy
search with identical results. `data/` and `index/` are gitignored; copy `index/` between machines instead of
rebuilding. The full CLIP encode takes roughly 10 minutes on a laptop CPU and resumes if interrupted.

## Reproducing the experiments
Each script prints what it does and writes to `results/`. Order matters where noted.

| Step | Command | Writes |
|---|---|---|
| 1 | `python scripts/build_index.py` | `index/*.npy` (classical + CLIP vectors) |
| 2 | `python scripts/baseline_classical.py`, `python scripts/eval_clip.py` | `baseline_classical.csv`, `baseline_clip.csv` |
| 3 | `python scripts/make_comparison.py` | comparison table and plots |
| 4 | `python scripts/run_experiments.py` | `robustness.*`, `restoration.*` |
| 5 | `python scripts/run_photometric.py` | `photometric.csv` (dark / low contrast) |
| 6 | `python scripts/tune_fusion.py` | `fusion_*.csv`, `fusion_weights.json` (tuned on val) |
| 7 | `python scripts/calibrate_quality.py` | `src/quality_thresholds.json`, `quality_*.csv/png` |
| 8 | `python scripts/run_adaptive.py`, then `python scripts/bootstrap_adaptive.py` | `adaptive_*.csv/json/png` |
| 9 | `python scripts/eval_duplicates.py` | `duplicate_thresholds.json`, `duplicates_*` |
| 10 | `python scripts/make_qualitative.py` | `qualitative_*.png` (query + top-k figures) |
| 11 | `python scripts/prepare_demo_dataset.py` | `index/demo/` (needs the Kaggle fashion files) |

Tuning (fusion weights, quality thresholds, adaptive weights, duplicate thresholds) uses the validation
split or synthetic copies only; every reported number uses the 500 test queries.

## Tests
```bash
for t in tests/test_*.py; do python "$t" || break; done
```
All tests use synthetic galleries and a stand-in for CLIP, so they need no downloads, except an optional
check in `test_app.py` that uses real Flowers-102 images when they are present.

## Repository layout
```
app/app.py                     Streamlit UI: search, explanation, duplicates, results, filters
src/features/classical.py      HSV, LBP, edge and DCT descriptors
src/features/clip_encoder.py   CLIP image and text encoder (open_clip)
src/index.py, retrieval.py     FAISS (or NumPy) exact search, SearchEngine
src/fusion.py                  score-level fusion and weight tuning
src/quality.py, adaptive.py    quality analyzer, quality-adaptive weights
src/explain.py                 per-result score breakdown and reasons
src/duplicates.py              near-duplicate tiers (CLIP cosine + perceptual hash)
src/filters.py, dataset.py     category / colour filters, datasets (Flowers-102, folders, catalogue)
src/degrade.py, evaluate.py    degradations, restoration, P@K / mAP / latency
scripts/                       experiments (table above) and scripts/check_setup.py
tests/                         unit and integration tests
results/                       CSV and PNG outputs used in the report
```

## Fashion demo
`scripts/prepare_demo_dataset.py` writes `index/demo/` (vectors, `labels.npy`, `metadata.csv`, `meta.json`).
The product images are not inside it: `meta.json` stores their folder as an absolute path from the machine that
built the index. On another machine put the images in `data/fashion/images/<id>.jpg` (found automatically) or set
`VISIONSEEK_DEMO_IMAGES=/path/to/images`. The Dataset selector in the sidebar then offers "Fashion demo".
Adaptive weights are switched off there because the quality thresholds were calibrated on Flowers-102.

## Honest limitations
* The quality thresholds were calibrated on Flowers-102; a noisy or already-compressed phone photo can land in
  the wrong bucket, which is why the UI shows the bucket and the adaptive toggle is off by default.
* Adaptive fusion helps only slightly and only for noise and blur; on clean, JPEG and dark queries it equals the
  tuned hybrid.
* Duplicate thresholds come from synthetic copies of Flowers-102 images; real-world edits (rotation, heavy crops,
  overlays) were not evaluated.
* The fashion demo is a small subset of one Kaggle dataset and is used for demonstration, not for headline metrics.

## Team
Two students. Person A: classical features, evaluation, degradations, restoration, quality analyzer, adaptive
fusion, demo dataset. Person B: CLIP and FAISS, Streamlit UI, fusion tuning, explanation, duplicate detection,
filters. Branches `feat/*`, pull requests reviewed by the other person.

## Data
Oxford Flowers-102 (via `torchvision.datasets.Flowers102`) and the Kaggle "Fashion Product Images (Small)"
dataset. Check each dataset page for its licence before redistributing anything derived from it.
