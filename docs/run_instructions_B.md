# Person B: Phase 1 run instructions (PowerShell, repo root, venv active)

## 0. One-time
```powershell
pip install -r requirements.txt -c constraints.txt   # if not done yet (order in requirements.txt)
python tests\test_evaluate.py          # Person A's test
python tests\test_clip_encoder.py      # logic only (fake model), no download
python tests\test_pipeline.py          # index/search plumbing on a tiny synthetic gallery
python tests\test_app.py               # headless UI smoke test
```

## 1. Build the indexes (Flowers-102)
```powershell
python scripts\build_index.py --limit 64 --index-dir index_smoke   # quick check that CLIP loads (downloads weights once)
python scripts\build_index.py                                     # full run; resumable if interrupted
```
Reuses Person A's `index\classical_*.npy` if they exist and match the gallery size.
Share `index\*.npy` with Person A through the drive (never commit them).

## 2. CLIP baseline + sanity check
```powershell
python scripts\eval_clip.py            # -> results\baseline_clip.csv, prints zero-shot accuracy
```
Expect P@10 well above the classical numbers and zero-shot accuracy around 60-70%.
If zero-shot is far lower, check the pretrained weights loaded and the class-name order.

## 3. Comparison table and plots for the slides
```powershell
python scripts\make_comparison.py      # -> results\comparison_table.csv/.md, comparison.png, robustness_heatmap.png
```

## 4. Demo app
```powershell
streamlit run app\app.py
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
python tests\test_fusion.py            # logic + hybrid search on the synthetic gallery
python scripts\tune_fusion.py          # val-tunes weights, reports on the 500 test queries
python scripts\tune_fusion.py --full-grid   # optional: clip + colour/texture/edge/dct separately
python scripts\make_comparison.py      # (re-run) then streamlit run app\app.py -> Retriever: "Hybrid"
```
Outputs: `results\fusion_tuning.csv`, `results\fusion_weights.json` (read by the app and by Person A's
adaptive fusion), `results\fusion_comparison.csv` (CLIP / classical / fixed 0.7-0.3 / tuned, on test).
Report the tuned row as the "fixed hybrid"; do not re-tune on test queries.

## For Person A: adaptive fusion (WP10)
```python
from src.fusion import fuse, tune, grid_weights, evaluate_fusion, load_weights
# scores = {"clip": q_clip @ clip_gallery.T, "classical_concat": q_cls @ cls_gallery.T}   (Q, N) each
# fused  = fuse(scores, weights_for_bucket, norm="zscore")            # (Q, N)
# per bucket: tune(scores_val_bucket, labels, val_ids, ["clip", "classical_concat"], step=0.1)
```
