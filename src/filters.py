"""Result filters (Person B, WP13b).

Fashion demo : exact metadata columns from metadata.csv (gender, masterCategory, subCategory,
               articleType, baseColour). Several values in one column are OR-ed, different
               columns are AND-ed. `baseColour` is the catalogue's own label, so the colour
               filter is exact there.
Flowers-102  : no metadata, so the category filter uses the class label and the colour filter
               is derived from the stored HSV histogram (approximate: 4 saturation and 4
               value levels, so names like "pink" or "grey" are rough).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.features.classical import HSV_BINS

MISSING = "(missing)"


# ------------------------------------------------------------------ metadata (catalogue)
def _as_text(series):
    return series.astype(object).where(series.notna(), MISSING).astype(str)


def value_counts(meta, column: str) -> list[tuple[str, int]]:
    """[(value, count)] of a metadata column, most frequent first (NaN shown as '(missing)')."""
    vc = _as_text(meta[column]).value_counts()
    return [(str(v), int(c)) for v, c in vc.items()]


def metadata_mask(meta, selections: dict[str, list]) -> np.ndarray | None:
    """Boolean mask over the gallery from {column: [allowed values]}; None if nothing selected."""
    mask = None
    for col, vals in selections.items():
        if not vals:
            continue
        if col not in meta.columns:
            raise KeyError(f"metadata has no column '{col}' (columns: {list(meta.columns)})")
        m = _as_text(meta[col]).isin([str(v) for v in vals]).to_numpy()
        mask = m if mask is None else (mask & m)
    return mask


# ------------------------------------------------------------------ HSV colour (Flowers-102)
COLOUR_NAMES = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
                "white", "grey", "black")
# name of each of the 16 hue bins (OpenCV hue 0..179 = 0..358 degrees, bin width 22.5 degrees)
_HUE_NAME = ("red", "orange", "yellow", "green", "green", "green", "green", "cyan",
             "cyan", "blue", "blue", "purple", "purple", "pink", "pink", "red")


def bin_colour_names() -> list[str]:
    """Colour name of every histogram bin, in the flattened (H, S, V) order of the descriptor."""
    nh, ns, nv = HSV_BINS
    assert nh == len(_HUE_NAME), "hue bin count changed: update _HUE_NAME"
    names = []
    for h in range(nh):
        for s in range(ns):
            for v in range(nv):
                if v == 0:
                    names.append("black")                                # V < 25%
                elif s == 0:
                    names.append("white" if v == nv - 1 else "grey")     # S < 25%
                else:
                    names.append(_HUE_NAME[h])
    return names


_BIN_NAMES = np.array(bin_colour_names())


def colour_shares(color_vecs: np.ndarray) -> np.ndarray:
    """(N, 256) colour descriptors -> (N, len(COLOUR_NAMES)) share of pixels per colour name.

    classical.py stores sqrt(hist / sum) L2-normalised, so the squared vector is the share of
    pixels in each bin."""
    p = np.asarray(color_vecs, dtype=np.float64) ** 2
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return np.stack([p[:, _BIN_NAMES == n].sum(axis=1) for n in COLOUR_NAMES], axis=1)


def filter_mask(n: int, labels: np.ndarray | None = None, classes=None,
                shares: np.ndarray | None = None, colour: str | None = None,
                min_share: float = 0.2) -> np.ndarray | None:
    """Class and HSV-colour filter. Mask (True = allowed) or None when no filter is active."""
    mask = None
    if classes is not None and len(classes) > 0:
        if labels is None:
            raise ValueError("labels are needed for a category filter")
        mask = np.isin(labels, np.asarray(list(classes)))
    if colour and colour != "any":
        if colour not in COLOUR_NAMES:
            raise ValueError(f"unknown colour '{colour}', expected one of {COLOUR_NAMES}")
        if shares is None:
            raise ValueError("colour shares are needed for a colour filter")
        c = shares[:, COLOUR_NAMES.index(colour)] >= min_share
        mask = c if mask is None else (mask & c)
    if mask is not None and len(mask) != n:
        raise ValueError("filter mask length does not match the gallery")
    return mask


# ------------------------------------------------------------------ applying a mask
def filter_result(res, mask: np.ndarray, k: int):
    """Keep the first k results whose gallery id is allowed by `mask`.

    Works for SearchResult and FusedResult (the per-retriever `parts` are filtered too).
    Search with k = the whole gallery first, otherwise fewer than k results may survive."""
    keep = mask[res.ids]
    changes = {"ids": res.ids[keep][:k], "scores": res.scores[keep][:k]}
    if hasattr(res, "parts"):
        changes["parts"] = {n: v[keep][:k] for n, v in res.parts.items()}
    return dataclasses.replace(res, **changes)


def min_similarity(res, min_sim: float):
    """Drop results whose similarity is below `min_sim` (0 or less = keep everything).

    The similarity is the raw cosine: `res.scores` for a single retriever or a text query, and the CLIP
    cosine (`parts['clip']`, else the first retriever) for a fused result, whose own score is a
    weighted z-score sum and not a cosine."""
    if min_sim <= 0:
        return res
    sim = res.scores
    if getattr(res, "parts", None):
        sim = res.parts.get("clip", next(iter(res.parts.values())))
    keep = np.asarray(sim) >= min_sim
    changes = {"ids": res.ids[keep], "scores": res.scores[keep]}
    if hasattr(res, "parts"):
        changes["parts"] = {n: v[keep] for n, v in res.parts.items()}
    return dataclasses.replace(res, **changes)
