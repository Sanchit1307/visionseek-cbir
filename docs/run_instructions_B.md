# Person B: Phase 1 run instructions (PowerShell, repo root, venv active)

## 0. One-time
```powershell
pip install -r requirements.txt -c constraints.txt   # if not done yet (order in requirements.txt)
python tests/test_evaluate.py          # Person A's test
python tests/test_clip_encoder.py      # logic only (fake model), no download
python tests/test_pipeline.py          # index/search plumbing on a tiny synthetic gallery
python tests/test_app.py               # headless UI smoke test
```

## 1. Build the indexes (Flowers-102)
```powershell
python scripts/build_index.py --limit 64 --index-dir index_smoke   # quick check that CLIP loads (downloads weights once)
python scripts/build_index.py                                     # full run; resumable if interrupted
```
Reuses Person A's `index/classical_*.npy` if they exist and match the gallery size.
Share `index/*.npy` with Person A through the drive (never commit them).

## 2. CLIP baseline + sanity check
```powershell
python scripts/eval_clip.py            # -> results/baseline_clip.csv, prints zero-shot accuracy
```
Expect P@10 well above the classical numbers and zero-shot accuracy around 60-70%.
If zero-shot is far lower, check the pretrained weights loaded and the class-name order.

## 3. Comparison table and plots for the slides
```powershell
python scripts/make_comparison.py      # -> results/comparison_table.csv/.md, comparison.png, robustness_heatmap.png
```

## 4. Demo app
```powershell
streamlit run app/app.py
```
Image search (6 retrievers), text search, image + text, optional degrade/restore of the query,
scores and latency, and an "Evaluation results" tab that shows whatever is in `results\`.

## For Person A: adding CLIP to the robustness experiment (two small edits in run_experiments.py)
```python
from src.retrieval import clip_query_encoder
# in load_gallery():  feats["clip"] = np.load(bc.INDEX_DIR / "clip_image.npy")
# in main():          run_experiments(images, gallery, labels, query_ids,
#                                     encoders=(classical_encoder, clip_query_encoder))
```

---
# Phase 2

## WP9: fixed hybrid fusion (CLIP + classical), tuned on the val split
```powershell
python tests/test_fusion.py            # logic + hybrid search on the synthetic gallery
python scripts/tune_fusion.py          # val-tunes weights, reports on the 500 test queries
python scripts/tune_fusion.py --full-grid   # optional: clip + colour/texture/edge/dct separately
python scripts/make_comparison.py      # (re-run) then streamlit run app/app.py -> Retriever: "Hybrid"
```
Outputs: `results/fusion_tuning.csv`, `results/fusion_weights.json` (read by the app and by Person A's
adaptive fusion), `results/fusion_comparison.csv` (CLIP / classical / fixed 0.7-0.3 / tuned, on test).
Report the tuned row as the "fixed hybrid"; do not re-tune on test queries.

## For Person A: adaptive fusion (WP10)
```python
from src.fusion import fuse, tune, grid_weights, evaluate_fusion, load_weights
# scores = {"clip": q_clip @ clip_gallery.T, "classical_concat": q_cls @ cls_gallery.T}   (Q, N) each
# fused  = fuse(scores, weights_for_bucket, norm="zscore")            # (Q, N)
# per bucket: tune(scores_val_bucket, labels, val_ids, ["clip", "classical_concat"], step=0.1)
```

## WP11: explanation panel
```powershell
python tests/test_explain.py           # numbers, reasons, visuals, contributions == fused scores
python tests/test_app.py               # includes "Why these results?" and inspect result #3
streamlit run app/app.py               # Image mode -> scroll below the grid -> "Why these results?"
```
Per result: cosine, "in top x% of the gallery", z-score, weight, contribution for CLIP / colour /
texture / edge / DCT, a plain-text reason built from the strongest signals (no language model),
plus hue histograms and gradient (edge) maps of query and match. Text-only searches have no
image to compare, so the panel is shown for image queries only.

## WP12: near-duplicate detection
```powershell
python tests/test_duplicates.py        # copies, pHash, tiers, curves, whole eval script on a synthetic gallery
python scripts/eval_duplicates.py --limit 100    # quick check (CLIP-encodes 100 x 4 copies)
python scripts/eval_duplicates.py                # full: 500 originals x 4 copies = 2,000 CLIP encodes (a few minutes)
python tests/test_app.py                         # includes the Duplicates tab
streamlit run app/app.py                         # tab "Duplicates"
```
Outputs (results\): `duplicate_thresholds.json` (read by the app), `duplicates_curve.csv`,
`duplicates_summary.csv` (precision + recall per copy type for CLIP / pHash / any / both),
`duplicates_pr.png`, `duplicates_false_positives.csv` (look at these by eye and mention them in the
report). Caches in index\: `phash.npy`, `dup_eval_cache.npz` (gitignored; pull the new .gitignore).
Thresholds come from the curves: duplicate = lowest cosine with precision >= 0.99 (`--min-precision`),
similar = best F1 for same-class retrieval. Do not edit them by hand to look better.

## WP13b: dataset switch, filters, fashion demo, adaptive toggle
Merge Person A's PRs first (feat/adaptive, feat/demo-dataset), then branch from main.
```powershell
python tests/test_filters.py      # HSV colours, metadata filters, CatalogGallery (loader errors), registry
python tests/test_app.py          # dataset switch, filters, adaptive toggle (skips parts A's modules / real data)
streamlit run app/app.py          # sidebar: Dataset = Flowers-102 / Fashion demo; Filters; Hybrid -> Adaptive
```
Fashion demo = `index/demo\` written by A's `scripts/prepare_demo_dataset.py` (vectors, labels.npy,
metadata.csv, meta.json). It appears in the Dataset selector as soon as `labels.npy` and `metadata.csv`
exist there. The 60x80 px product images are found through `image_root` in meta.json; if that path is an
absolute path from A's machine, copy the .jpg files here and set
`$env:VISIONSEEK_DEMO_IMAGES = "C:\path\to\images"` before `streamlit run`.
`index\` is gitignored: A shares `index/demo\` (including metadata.csv and meta.json) and the images by drive.
* Filters (catalogue): gender, master category, sub category, article type and exact `baseColour`
  (several values in one box = OR, different boxes = AND). Flowers-102 has no metadata: category =
  class, colour = approximate HSV name.
* Adaptive weights: Flowers-102 only (quality thresholds were calibrated there); off for the catalogue.
  The bucket and weights are shown under the results. Needs `src/adaptive.py`, `src/quality.py`,
  `results/adaptive_weights.json`, `src/quality_thresholds.json` committed (A's note).

## WP12 follow-up: why is the CLIP duplicate threshold 0.99?
```powershell
python scripts/eval_duplicates.py     # cached copies, takes seconds; now prints [1] [2] [3] diagnostics
```
[1] cosine of each copy type to its original; [2] false positives vs threshold and how many distinct
originals cause them; [3] `results/duplicates_fp_sheet.png`: closest non-original pairs to look at.

## Final steps (v3)
```bash
python scripts/make_qualitative.py        # results/qualitative_clean.png, _noise50.png, _blur15.png (for the slides)
python tests/test_qualitative.py
```
`README.md` and a corrected `.gitignore` are included in this package. The old `.gitignore` had no trailing
newline, so do NOT append lines to it with `>>` (the new line would be glued to `.env`); use the supplied file.
If you already ran that command, check `tail -3 .gitignore`: a line reading `.env*.npz` is the damage; restore
the file with `git checkout .gitignore` and unzip this package again.

## Code-complete additions (v4)
```bash
python tests/test_faiss_path.py      # real-FAISS code path with a strict fake faiss (dtype, contiguity, k padding)
python tests/test_app.py             # now also: side-by-side comparison, minimum similarity, quality panel
```
* App: **Minimum similarity** slider (sidebar; for the hybrid it is the CLIP cosine), **Also show the original
  query's results** in the robustness expander (top-5 strips, green = same class), **Query quality** expander
  (analyzer bucket and metrics, needs Person A's modules), a failed search no longer hides the other tabs.
* `src/index.py`: a stale `.faiss` file (different N or D than the `.npy` vectors) is rebuilt instead of used.
* `src/duplicates.py`: built-in defaults are the values measured on Flowers-102 (CLIP 0.99, pHash <= 7,
  similar 0.90, rule "any"); `results/duplicate_thresholds.json` overrides them once you have re-run
  `scripts/eval_duplicates.py`. That script now also works on indexes without `query_ids.npy` (the fashion demo).
