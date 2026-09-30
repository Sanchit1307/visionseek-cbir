"""VisionSeek Streamlit UI, v1 (Person B).

Run from the repo root:
    streamlit run app\\app.py

Search modes: image -> image (choose the retriever), text -> image (CLIP),
image + text -> image (CLIP). Shows similarity scores and latency for each query.
Environment: VISIONSEEK_DATASET ('flowers102' default, or 'folder:<path>'),
VISIONSEEK_INDEX_DIR, VISIONSEEK_RESULTS_DIR.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import retrieval as rt  # noqa: E402
from src.dataset import INDEX_DIR, RESULTS_DIR, bgr_to_rgb, open_gallery  # noqa: E402
from src.degrade import degrade, restore  # noqa: E402
from src.explain import edge_map, explain_results, hue_histogram_image  # noqa: E402
from src.fusion import DEFAULT_NORM, load_weights  # noqa: E402

DATASET = os.environ.get("VISIONSEEK_DATASET", "flowers102")
NOISE_SEED = 1234
RETRIEVER_LABELS = {
    "clip": "CLIP (deep, ViT-B/32)",
    "classical_concat": "Classical: all four combined",
    "color": "Classical: colour (HSV histogram)",
    "texture": "Classical: texture (LBP)",
    "edge": "Classical: edge orientation",
    "dct": "Classical: DCT statistics",
    "hybrid": "Hybrid: CLIP + classical (score fusion)",
}
DEGRADE_LEVELS = {"noise": (10, 25, 50), "blur": (5, 9, 15), "jpeg": (30, 10)}
RESTORE_OPTIONS = {"none": None, "median 3x3": ("median", 3), "median 5x5": ("median", 5),
                   "gaussian 3x3": ("gaussian", 3), "gaussian 5x5": ("gaussian", 5)}

st.set_page_config(page_title="VisionSeek", layout="wide")


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading indexes...")
def get_engine(index_dir: str) -> rt.SearchEngine:
    return rt.SearchEngine(Path(index_dir))


@st.cache_resource(show_spinner="Opening dataset...")
def get_gallery(spec: str):
    return open_gallery(spec)


@st.cache_data(show_spinner=False, max_entries=1024)
def thumbnail(spec: str, g: int, max_side: int = 240) -> np.ndarray:
    img, _ = get_gallery(spec).get(g)
    return bgr_to_rgb(fit(img, max_side))


def fit(img_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img_bgr
    return cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def class_name(gallery, label: int) -> str:
    names = getattr(gallery, "class_names", None)
    return names[label] if names and label < len(names) else f"class {label}"


# ------------------------------------------------------------------ sidebar
engine = get_engine(str(INDEX_DIR))
if not engine.available:
    st.error(f"No indexes found in {INDEX_DIR}. Run `python scripts\\build_index.py` first.")
    st.stop()
gallery = get_gallery(DATASET)

st.title("VisionSeek: multimodal image search")

with st.sidebar:
    st.header("Search")
    mode = st.radio("Search mode", ["Image", "Text", "Image + text"], key="mode")
    if mode == "Image":
        can_fuse = {"clip", "classical_concat"} <= set(engine.available)
        retriever = st.selectbox("Retriever", engine.available + (["hybrid"] if can_fuse else []),
                                 key="retriever", format_func=lambda n: RETRIEVER_LABELS.get(n, n))
        if retriever == "hybrid":
            tuned_w, tuned_norm = load_weights(RESULTS_DIR / "fusion_weights.json")
            two_way = set(tuned_w) == {"clip", "classical_concat"}
            default_w = round(tuned_w["clip"] / sum(tuned_w.values()) / 0.05) * 0.05 if two_way else 0.7
            fuse_norm = tuned_norm if two_way else DEFAULT_NORM
            w_clip = st.slider("CLIP weight (classical = 1 - CLIP)", 0.0, 1.0,
                               float(min(1.0, max(0.0, default_w))), 0.05, key="w_clip")
            st.caption("Default = weights tuned on the validation split "
                       "(results/fusion_weights.json), or 0.7 / 0.3 if not tuned yet.")
    else:
        retriever = "clip"
        st.caption("Text queries use the CLIP index (classical descriptors have no text meaning).")
    k = st.slider("Results (top-K)", 1, 30, 10, key="k")
    w_img = 0.5
    if mode == "Image + text":
        w_img = st.slider("Weight of the image (vs text)", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    st.caption(f"Gallery: {DATASET}, {engine.n_gallery} images")
    st.caption("Indexes: " + ", ".join(engine.available))
    from src import index as _ix
    st.caption("Search backend: " + ("FAISS IndexFlatIP" if _ix.HAVE_FAISS else "NumPy exact search"))

tab_search, tab_results = st.tabs(["Search", "Evaluation results"])

# ------------------------------------------------------------------ search tab
with tab_search:
    query_img: np.ndarray | None = None
    query_gid: int | None = None          # gallery id when the query is a gallery image
    text = ""

    if mode in ("Image", "Image + text"):
        source = st.radio("Query image", ["Upload a photo", "Random test image from the gallery"],
                          horizontal=True, key="source")
        if source == "Upload a photo":
            up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if up is not None:
                query_img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                if query_img is None:
                    st.error("Could not decode that file as an image.")
        else:
            qids = np.load(INDEX_DIR / "query_ids.npy") if (INDEX_DIR / "query_ids.npy").exists() \
                else np.arange(engine.n_gallery)
            if "qid" not in st.session_state:
                st.session_state.qid = int(np.random.default_rng().choice(qids))
            if st.button("Pick another random query"):
                st.session_state.qid = int(np.random.default_rng().choice(qids))
            query_gid = st.session_state.qid
            query_img = get_gallery(DATASET).get(query_gid)[0]

        with st.expander("Degrade / restore the query (robustness demo)"):
            c1, c2, c3 = st.columns(3)
            kind = c1.selectbox("Degradation", ["none", *DEGRADE_LEVELS], key="kind")
            level = c2.selectbox("Level", DEGRADE_LEVELS.get(kind, (0,)), disabled=kind == "none",
                                 key=f"level_{kind}")
            rest = c3.selectbox("Restoration filter", list(RESTORE_OPTIONS), key="rest")
            if query_img is not None:
                if kind != "none":
                    query_img = degrade(query_img, kind, int(level), seed=NOISE_SEED)
                if RESTORE_OPTIONS[rest] is not None:
                    query_img = restore(query_img, *RESTORE_OPTIONS[rest])

    if mode in ("Text", "Image + text"):
        text = st.text_input("Text query", placeholder="e.g. a yellow flower with many petals", key="text").strip()

    ready = (mode == "Text" and text) or (mode == "Image" and query_img is not None) \
        or (mode == "Image + text" and query_img is not None and text)

    if query_img is not None and mode != "Text":
        left, _ = st.columns([1, 3])
        with left:
            cap = "Query image" + (f" (gallery #{query_gid}, {class_name(gallery, int(engine.labels[query_gid]))})"
                                   if query_gid is not None and engine.labels is not None else "")
            st.image(bgr_to_rgb(fit(query_img, 300)), caption=cap)

    if not ready:
        st.info({"Image": "Upload a photo or pick a random gallery image to search.",
                 "Text": "Type a description to search.",
                 "Image + text": "Provide both a query image and a text description."}[mode])
    elif "clip" not in engine.available and (mode != "Image" or retriever in ("clip", "hybrid")):
        st.warning("The CLIP index is missing. Run `python scripts\\build_index.py` (without --skip-clip).")
    else:
        try:
            with st.spinner("Searching..."):
                if mode == "Image" and retriever == "hybrid":
                    res = engine.search_fused(query_img, {"clip": w_clip, "classical_concat": 1.0 - w_clip},
                                              k, fuse_norm, exclude_id=query_gid)
                elif mode == "Image":
                    res = engine.search_image(query_img, retriever, k, exclude_id=query_gid)
                elif mode == "Text":
                    res = engine.search_text(text, k)
                else:
                    res = engine.search_image_text(query_img, text, k, w_img, exclude_id=query_gid)
        except Exception as exc:  # e.g. CLIP weights could not be loaded
            st.error(f"Search failed: {exc}")
            st.stop()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Feature extraction", f"{res.extract_ms:.1f} ms")
        m2.metric("Index search", f"{res.search_ms:.2f} ms")
        m3.metric("Total", f"{res.total_ms:.1f} ms")
        rel = None
        if query_gid is not None and engine.labels is not None:
            rel = engine.labels[res.ids] == engine.labels[query_gid]
            m4.metric(f"Precision@{len(res.ids)} (this query)", f"{rel.mean():.0%}")

        if retriever == "hybrid" and mode == "Image":
            st.caption("Fused score = weighted sum of per-retriever z-scores "
                       f"(CLIP {res.weights.get('clip', 0):.2f}, classical {res.weights.get('classical_concat', 0):.2f}, "
                       f"norm = {res.norm}); it is not a cosine, only the order matters.")
        st.subheader(f"Top {len(res.ids)} results  ·  {RETRIEVER_LABELS.get(res.retriever, res.retriever)}")
        exps = None                                   # per-result explanations (image queries only)
        if mode == "Image":
            sims = engine.all_similarities(query_img)
            w_used = res.weights if retriever == "hybrid" else {retriever: 1.0}
            exps = explain_results(sims, res.ids, w_used, getattr(res, "norm", DEFAULT_NORM))
        cols_per_row = 5
        for row in range(0, len(res.ids), cols_per_row):
            cols = st.columns(cols_per_row)
            for c, j in zip(cols, range(row, min(row + cols_per_row, len(res.ids)))):
                g = int(res.ids[j])
                label = int(engine.labels[g]) if engine.labels is not None else -1
                mark = "" if rel is None else ("✅ " if rel[j] else "❌ ")
                c.image(thumbnail(DATASET, g),
                        caption=f"{mark}#{j + 1}  score {res.scores[j]:.3f}\n{class_name(gallery, label)}"
                                + (f"\n↳ {exps[j].short}" if exps is not None and exps[j].short else ""))

        if exps is not None:
            st.divider()
            st.subheader("Why these results?")
            pick = st.selectbox("Inspect result", list(range(1, len(res.ids) + 1)), key="inspect",
                                format_func=lambda i: f"#{i}")
            e = exps[pick - 1]
            match_img = get_gallery(DATASET).get(e.gid)[0]
            st.write(e.text)
            table = pd.DataFrame([{
                "Retriever": r["label"], "Cosine": round(r["cosine"], 3),
                "In top % of gallery": round(r["top_pct"], 2), "z-score": round(r["z"], 2),
                "Weight": round(r["weight"], 2), "Contribution": round(r["contribution"], 3)}
                for r in e.rows])
            st.dataframe(table, hide_index=True)
            if retriever == "hybrid":
                st.caption(f"Contribution = weight x normalised score; they add up to the fused score {e.fused:.3f}.")
            else:
                st.caption("Only the selected retriever ranked this search (weight 1); "
                           "the other rows are supporting evidence.")
            qcol, mcol = st.columns(2)
            for col, title, im in ((qcol, "Query", query_img), (mcol, f"Result #{pick}", match_img)):
                col.markdown(f"**{title}**")
                col.image(bgr_to_rgb(fit(im, 260)))
                col.caption("Hue histogram (colour descriptor)")
                col.image(bgr_to_rgb(hue_histogram_image(im)))
                col.caption("Gradient magnitude (edge descriptor)")
                col.image(bgr_to_rgb(edge_map(im)))

# ------------------------------------------------------------------ results tab
with tab_results:
    st.caption("Generated by the experiment scripts (results/ folder).")
    tables = [pd.read_csv(RESULTS_DIR / f) for f in ("baseline_classical.csv", "baseline_clip.csv")
              if (RESULTS_DIR / f).exists()]
    if tables:
        st.subheader("Retriever comparison")
        st.dataframe(pd.concat(tables, ignore_index=True), hide_index=True)
    else:
        st.info("No results yet: run scripts\\baseline_classical.py and scripts\\eval_clip.py.")
    for fname, title in (("comparison.png", "Quality and latency"),
                         ("robustness.png", "Robustness to noise / blur / JPEG"),
                         ("robustness_heatmap.png", "mAP retained under degradation"),
                         ("restoration.png", "Restoration (median / Gaussian filtering)")):
        if (RESULTS_DIR / fname).exists():
            st.subheader(title)
            st.image(str(RESULTS_DIR / fname))
