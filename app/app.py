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
from src import filters as fl  # noqa: E402
from src.dataset import RESULTS_DIR, available_datasets, bgr_to_rgb, open_gallery  # noqa: E402
from src.degrade import degrade, restore  # noqa: E402
from src import duplicates as dp  # noqa: E402
from src.explain import edge_map, explain_results, hue_histogram_image  # noqa: E402
from src.fusion import DEFAULT_NORM, load_weights  # noqa: E402

try:                                    # Person A's branch feat/adaptive (merge it first)
    from src.adaptive import adaptive_weights, load_table  # noqa: E402
    from src.quality import analyze  # noqa: E402
    HAVE_ADAPTIVE = True
except ImportError:
    HAVE_ADAPTIVE = False

PRIMARY_SPEC = os.environ.get("VISIONSEEK_DATASET", "flowers102")
CATALOG_CAPTION_CHARS = 44
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


@st.cache_resource(show_spinner=False)
def get_colour_shares(index_dir: str) -> np.ndarray | None:
    """Per-image share of each named colour from the stored HSV descriptors (Flowers-102 only)."""
    eng = get_engine(index_dir)
    return fl.colour_shares(eng.vectors["color"]) if "color" in eng.vectors else None


@st.cache_data(show_spinner=False, max_entries=1024)
def thumbnail(spec: str, g: int, max_side: int = 240) -> np.ndarray:
    img, _ = get_gallery(spec).get(g)
    return bgr_to_rgb(fit(img, max_side))


@st.cache_resource(show_spinner=False)
def get_gallery_phash(index_dir: str, n: int):
    """index/phash.npy (written by scripts/eval_duplicates.py) if it matches the gallery."""
    p = Path(index_dir) / "phash.npy"
    if p.exists():
        bits = np.load(p)
        if bits.shape == (n, dp.PHASH_BITS):
            return bits
    return None


@st.cache_data(show_spinner="Scanning the gallery for duplicate pairs...", max_entries=8)
def scan_pairs(index_dir: str, thr: float) -> pd.DataFrame:      # index_dir is part of the cache key
    return dp.duplicate_pairs(engine.vectors["clip"], thr)


def fit(img_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:                       # tiny catalogue images (60 x 80 px) are shown upscaled
        if max(h, w) >= 200:
            return img_bgr
        return cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def class_name(gallery, label: int) -> str:
    names = getattr(gallery, "class_names", None)
    return names[label] if names and label < len(names) else f"class {label}"


def item_name(gallery, g: int, label: int) -> str:
    """Caption text for gallery image g: the product name for the catalogue, else the class name."""
    if hasattr(gallery, "describe"):
        t = gallery.describe(g)
        return t if len(t) <= CATALOG_CAPTION_CHARS else t[:CATALOG_CAPTION_CHARS - 1] + "…"
    return class_name(gallery, label)


# ------------------------------------------------------------------ sidebar
DATASETS = available_datasets(PRIMARY_SPEC)
with st.sidebar:
    if len(DATASETS) > 1:
        ds_label = st.selectbox("Dataset", list(DATASETS), key="dataset")
    else:
        ds_label = next(iter(DATASETS))
DATASET, INDEX_ACTIVE = DATASETS[ds_label]
ds_i = list(DATASETS).index(ds_label)          # widget keys below are per dataset
try:
    engine = get_engine(str(INDEX_ACTIVE))
    gallery = get_gallery(DATASET)
except (OSError, ValueError) as exc:           # e.g. catalogue image folder not found
    st.error(f"Could not open '{ds_label}': {exc}")
    st.stop()
if not engine.available:
    st.error(f"No indexes found in {INDEX_ACTIVE}. Run `python scripts\\build_index.py` first.")
    st.stop()

st.title("VisionSeek: multimodal image search")

with st.sidebar:
    st.header("Search")
    mode = st.radio("Search mode", ["Image", "Text", "Image + text"], key="mode")
    use_adaptive, quality = False, None
    if mode == "Image":
        can_fuse = {"clip", "classical_concat"} <= set(engine.available)
        retriever = st.selectbox("Retriever", engine.available + (["hybrid"] if can_fuse else []),
                                 key="retriever", format_func=lambda n: RETRIEVER_LABELS.get(n, n))
        if retriever == "hybrid":
            tuned_w, tuned_norm = load_weights(RESULTS_DIR / "fusion_weights.json")
            two_way = set(tuned_w) == {"clip", "classical_concat"}
            default_w = round(tuned_w["clip"] / sum(tuned_w.values()) / 0.05) * 0.05 if two_way else 0.7
            fuse_norm = tuned_norm if two_way else DEFAULT_NORM
            adaptive_ok = HAVE_ADAPTIVE and DATASET == "flowers102"
            use_adaptive = st.checkbox("Adaptive weights (by query quality)", value=False,
                                       key="use_adaptive", disabled=not adaptive_ok)
            use_adaptive = bool(use_adaptive and adaptive_ok)       # a stale True must not leak across datasets
            w_clip = st.slider("CLIP weight (classical = 1 - CLIP)", 0.0, 1.0,
                               float(min(1.0, max(0.0, default_w))), 0.05, key="w_clip",
                               disabled=use_adaptive)
            if not HAVE_ADAPTIVE:
                st.caption("Adaptive weights need Person A's src/adaptive.py and src/quality.py (branch feat/adaptive).")
            elif DATASET != "flowers102":
                st.caption("Adaptive weights are off here: the quality thresholds were calibrated on Flowers-102.")
            else:
                st.caption("Adaptive picks the CLIP / classical weights from the query's quality bucket. "
                           "Buckets were calibrated on Flowers-102, so check the bucket shown under the results.")
            st.caption("Manual default = weights tuned on the validation split "
                       "(results/fusion_weights.json), or 0.7 / 0.3 if not tuned yet.")
    else:
        retriever = "clip"
        st.caption("Text queries use the CLIP index (classical descriptors have no text meaning).")
    k = st.slider("Results (top-K)", 1, 30, 10, key="k")
    w_img = 0.5
    if mode == "Image + text":
        w_img = st.slider("Weight of the image (vs text)", 0.0, 1.0, 0.5, 0.05)

    st.header("Filters")
    meta = getattr(gallery, "meta", None)
    mask = None
    if meta is not None:                  # catalogue: exact metadata columns
        picks = {}
        for col, label in type(gallery).FILTER_COLUMNS:
            if col in meta.columns:
                counts = dict(fl.value_counts(meta, col))
                picks[col] = st.multiselect(label, list(counts), key=f"f_{col}_{ds_i}",
                                            format_func=lambda v, c=counts: f"{v} ({c[v]})")
        mask = fl.metadata_mask(meta, picks)
    else:                                 # Flowers-102: class label + approximate HSV colour
        cls_names = getattr(gallery, "class_names", None)
        sel_classes = (st.multiselect("Category", list(range(len(cls_names))), key=f"f_cls_{ds_i}",
                                      format_func=lambda i: cls_names[i]) if cls_names else [])
        shares = get_colour_shares(str(INDEX_ACTIVE))
        colour, min_share = "any", 0.2
        if shares is not None:
            colour = st.selectbox("Dominant colour (approximate)", ["any", *fl.COLOUR_NAMES], key=f"f_col_{ds_i}")
            if colour != "any":
                min_share = st.slider("Minimum share of that colour", 0.05, 0.90, 0.20, 0.05,
                                      key=f"f_share_{ds_i}")
        mask = fl.filter_mask(engine.n_gallery, engine.labels, sel_classes, shares, colour, min_share)
    if mask is not None:
        st.caption(f"{int(mask.sum())} of {engine.n_gallery} images match the filters.")

    st.divider()
    st.caption(f"Gallery: {ds_label}, {engine.n_gallery} images")
    st.caption("Indexes: " + ", ".join(engine.available))
    from src import index as _ix
    st.caption("Search backend: " + ("FAISS IndexFlatIP" if _ix.HAVE_FAISS else "NumPy exact search"))

tab_search, tab_dups, tab_results = st.tabs(["Search", "Duplicates", "Evaluation results"])

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
            qids = np.load(INDEX_ACTIVE / "query_ids.npy") if (INDEX_ACTIVE / "query_ids.npy").exists() \
                else np.arange(engine.n_gallery)
            qkey = f"qid_{ds_i}"
            if qkey not in st.session_state:
                st.session_state[qkey] = int(np.random.default_rng().choice(qids))
            if st.button("Pick another random query"):
                st.session_state[qkey] = int(np.random.default_rng().choice(qids))
            query_gid = st.session_state[qkey]
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
            cap = "Query image" + (f" (gallery #{query_gid}, {item_name(gallery, query_gid, int(engine.labels[query_gid]))})"
                                   if query_gid is not None and engine.labels is not None else "")
            st.image(bgr_to_rgb(fit(query_img, 300)), caption=cap)

    if not ready:
        st.info({"Image": "Upload a photo or pick a random gallery image to search.",
                 "Text": "Type a description to search.",
                 "Image + text": "Provide both a query image and a text description."}[mode])
    elif "clip" not in engine.available and (mode != "Image" or retriever in ("clip", "hybrid")):
        st.warning("The CLIP index is missing. Run `python scripts\\build_index.py` (without --skip-clip).")
    else:
        k_search = engine.n_gallery if mask is not None else k    # filters need the full ranking
        try:
            with st.spinner("Searching..."):
                if mode == "Image" and retriever == "hybrid":
                    w_manual = {"clip": w_clip, "classical_concat": 1.0 - w_clip}
                    w_used_in, norm_used, quality = w_manual, fuse_norm, None
                    if use_adaptive:
                        try:
                            quality = analyze(query_img)                    # BGR uint8
                            w_used_in = adaptive_weights(quality)
                            norm_used = load_table()[1]                     # must match the table
                        except ValueError:                                  # image smaller than 16 px
                            quality, w_used_in, norm_used = None, w_manual, fuse_norm
                    res = engine.search_fused(query_img, w_used_in, k_search, norm_used, exclude_id=query_gid)
                elif mode == "Image":
                    res = engine.search_image(query_img, retriever, k_search, exclude_id=query_gid)
                elif mode == "Text":
                    res = engine.search_text(text, k_search)
                else:
                    res = engine.search_image_text(query_img, text, k_search, w_img, exclude_id=query_gid)
        except Exception as exc:  # e.g. CLIP weights could not be loaded
            st.error(f"Search failed: {exc}")
            st.stop()
        res = fl.filter_result(res, mask, k) if mask is not None else res
        if len(res.ids) == 0:
            st.warning("No gallery image matches the current filters for this query.")
            res = None

        if res is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Feature extraction", f"{res.extract_ms:.1f} ms")
            m2.metric("Index search", f"{res.search_ms:.2f} ms")
            m3.metric("Total", f"{res.total_ms:.1f} ms")
            if mode == "Image" and retriever == "hybrid" and use_adaptive:
                if quality is not None:
                    st.caption(f"Query quality: **{quality['bucket']}** "
                               f"(sharpness {quality['sharpness']:.0f}, noise {quality['noise']:.1f}, "
                               f"brightness {quality['brightness']:.0f}, contrast {quality['contrast']:.0f}, "
                               f"blockiness {quality['jpeg_blockiness']:.2f}). "
                               f"Weights: CLIP {res.weights.get('clip', 0.0):.2f} / "
                               f"classical {res.weights.get('classical_concat', 0.0):.2f}.")
                else:
                    st.caption("Adaptive weights: the quality analysis was skipped (image too small); "
                               "the manual weights were used.")
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
                            caption=f"{mark}#{j + 1}  score {res.scores[j]:.3f}\n{item_name(gallery, g, label)}"
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

# ------------------------------------------------------------------ duplicates tab
TIER_MARK = {"duplicate": "🔴 duplicate", "similar": "🟡 similar"}
with tab_dups:
    st.caption("Near-duplicate check on CLIP cosine similarity (optionally with a perceptual hash). "
               "Tiers: duplicate (same photo: resized, re-compressed, cropped, brightness-shifted), "
               "similar (same kind of subject), different.")
    if "clip" not in engine.available:
        st.warning("The CLIP index is missing. Run `python scripts\\build_index.py` (without --skip-clip).")
    else:
        thr_file = RESULTS_DIR / "duplicate_thresholds.json"
        th0 = dp.load_thresholds(thr_file)
        if DATASET != "flowers102":
            st.caption("The duplicate thresholds below were tuned on Flowers-102 copies; "
                       "re-run scripts/eval_duplicates.py with this dataset before trusting them here.")
        gbits = get_gallery_phash(str(INDEX_ACTIVE), engine.n_gallery)
        st.caption("Thresholds: from the precision-recall evaluation (results/duplicate_thresholds.json)."
                   if thr_file.exists() else
                   "Thresholds are placeholders: run `python scripts\\eval_duplicates.py` to choose them from data.")
        c1, c2, c3 = st.columns(3)
        dup_cos = c1.slider("Duplicate if CLIP cosine >=", 0.50, 1.00, float(min(max(th0.dup_cos, 0.5), 1.0)),
                            0.005, key="dup_cos")
        sim_cos = c2.slider("Similar if CLIP cosine >=", 0.30, float(dup_cos),
                            float(min(th0.sim_cos, dup_cos)), 0.005, key="sim_cos")
        rules = list(dp.RULES) if gbits is not None else ["clip"]
        rule = c3.selectbox("Duplicate rule", rules, index=rules.index(th0.rule) if th0.rule in rules else 0,
                            key="dup_rule",
                            help="clip = cosine only; phash = perceptual hash only; any / both = combine them")
        ham_thr = int(th0.ham)
        if rule != "clip":
            ham_thr = st.slider("pHash: duplicate if Hamming distance <=", 0, 32, int(min(max(th0.ham, 0), 32)),
                                key="dup_ham")
        if gbits is None:
            st.caption("pHash index not found (index/phash.npy, created by scripts/eval_duplicates.py): CLIP only.")

        d_img: np.ndarray | None = None
        d_gid: int | None = None
        d_src = st.radio("Test image", ["Gallery image + synthetic edit", "Upload a photo"],
                         horizontal=True, key="dup_source")
        if d_src == "Upload a photo":
            up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="dup_upload")
            if up is not None:
                d_img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                if d_img is None:
                    st.error("Could not decode that file as an image.")
        else:
            pool = np.load(INDEX_ACTIVE / "query_ids.npy") if (INDEX_ACTIVE / "query_ids.npy").exists() \
                else np.arange(engine.n_gallery)
            dkey = f"dup_gid_{ds_i}"
            if dkey not in st.session_state:
                st.session_state[dkey] = int(np.random.default_rng().choice(pool))
            if st.button("Pick another random image", key="dup_pick"):
                st.session_state[dkey] = int(np.random.default_rng().choice(pool))
            edit = st.selectbox("Edit applied to it", ["none (exact copy)", *dp.DUP_KINDS], key="dup_edit")
            d_gid = st.session_state[dkey]
            d_img = get_gallery(DATASET).get(d_gid)[0]
            if edit in dp.DUP_KINDS:
                d_img = dp.make_duplicate(d_img, edit)

        if d_img is not None:
            left, _ = st.columns([1, 3])
            left.image(bgr_to_rgb(fit(d_img, 300)),
                       caption="Test image" + (f" (gallery #{d_gid}, {edit})" if d_gid is not None else ""))
            cos = engine.vectors["clip"] @ engine.query_vector(d_img, "clip")[0][0]
            ham = dp.hamming_matrix(dp.phash_bits(d_img)[None], gbits)[0] if gbits is not None else None
            th = dp.Thresholds(dup_cos, sim_cos, ham_thr, rule)
            tiers = dp.classify(cos, ham, th)
            matches = dp.find_matches(cos, ham, th, top=30)
            m1, m2, m3 = st.columns(3)
            m1.metric("Duplicates", int((tiers == "duplicate").sum()))
            m2.metric("Similar", int((tiers == "similar").sum()))
            m3.metric("Best cosine", f"{float(cos.max()):.3f}")
            if d_gid is not None:
                rank = int((cos > cos[d_gid]).sum()) + 1
                found = tiers[d_gid] == "duplicate"
                st.markdown(("✅ " if found else "❌ ") + f"The original (#{d_gid}) is "
                            + ("detected as a duplicate" if found else "NOT detected as a duplicate")
                            + f" (cosine {cos[d_gid]:.3f}, rank {rank}"
                            + (f", Hamming {int(ham[d_gid])}" if ham is not None else "") + ").")
            if matches.empty:
                st.info("No duplicates or similar images found at these thresholds.")
            else:
                show = matches.head(10)
                for row0 in range(0, len(show), 5):
                    cols = st.columns(5)
                    for c, (_, r) in zip(cols, show.iloc[row0:row0 + 5].iterrows()):
                        g = int(r["gid"])
                        lab = int(engine.labels[g]) if engine.labels is not None else -1
                        c.image(thumbnail(DATASET, g),
                                caption=f"{TIER_MARK[r['tier']]}\ncos {r['cos']:.3f}"
                                        + (f" · ham {int(r['hamming'])}" if ham is not None else "")
                                        + f"\n#{g} {item_name(gallery, g, lab)}")
                with st.expander("All matches (table)"):
                    st.dataframe(matches, hide_index=True)

        with st.expander("Scan the whole gallery for duplicate pairs"):
            st.caption("Every gallery pair with CLIP cosine >= the duplicate threshold above.")
            if st.button("Scan gallery", key="dup_scan"):
                pairs = scan_pairs(str(INDEX_ACTIVE), float(dup_cos))
                st.write(f"{len(pairs)} duplicate pair(s) at cosine >= {dup_cos:.3f}.")
                for _, r in pairs.head(8).iterrows():
                    a, b = int(r["i"]), int(r["j"])
                    pc = st.columns([1, 1, 3])
                    pc[0].image(thumbnail(DATASET, a), caption=f"#{a}")
                    pc[1].image(thumbnail(DATASET, b), caption=f"#{b}")
                    pc[2].write(f"cosine {r['cos']:.4f}")

# ------------------------------------------------------------------ results tab
with tab_results:
    st.caption("Benchmark results below were measured on Flowers-102, whichever dataset is selected.")
    st.caption("Generated by the experiment scripts (results/ folder).")
    tables = [pd.read_csv(RESULTS_DIR / f) for f in ("baseline_classical.csv", "baseline_clip.csv")
              if (RESULTS_DIR / f).exists()]
    if tables:
        st.subheader("Retriever comparison")
        st.dataframe(pd.concat(tables, ignore_index=True), hide_index=True)
    else:
        st.info("No results yet: run scripts\\baseline_classical.py and scripts\\eval_clip.py.")
    if (RESULTS_DIR / "fusion_comparison.csv").exists():
        st.subheader("Fixed hybrid fusion (tuned on val, reported on test)")
        st.dataframe(pd.read_csv(RESULTS_DIR / "fusion_comparison.csv"), hide_index=True)
    for fname, title in (("comparison.png", "Quality and latency"),
                         ("robustness.png", "Robustness to noise / blur / JPEG"),
                         ("robustness_heatmap.png", "mAP retained under degradation"),
                         ("restoration.png", "Restoration (median / Gaussian filtering)"),
                         ("duplicates_pr.png", "Near-duplicate detection: precision-recall")):
        if (RESULTS_DIR / fname).exists():
            st.subheader(title)
            st.image(str(RESULTS_DIR / fname))
