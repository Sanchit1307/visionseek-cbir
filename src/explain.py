"""Per-result explanations (Person B, WP11). No language model: everything is
computed from similarity scores and simple image statistics.

For a query and its top-k results, `explain_results` reports for every retriever
(CLIP, colour, texture, edge, DCT, and classical_concat when it was used):
    cosine        raw cosine similarity between query and result
    top_pct       share of the whole gallery that is MORE similar to the query
                  (0.5 = the result is in the top 0.5% for that retriever)
    z             standardised score over the gallery (what the fusion uses)
    weight        weight used by the search (0 if the retriever was not used)
    contribution  weight * normalised score; the contributions add up exactly to
                  the fused score returned by SearchEngine.search_fused
and turns the strongest signals into a text reason ("similar colour distribution
(top 1%)"), ordered by contribution first, then by rarity.

Visual helpers: `hue_histogram_image` and `edge_map` draw the colour histogram and
the gradient (edge) magnitude of a query / match pair for side-by-side display.
Image convention: OpenCV BGR uint8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.fusion import DEFAULT_NORM, normalize_scores

EXPLAIN_NAMES = ("clip", "color", "texture", "edge", "dct")

LABEL = {"clip": "CLIP (content)", "color": "Colour (HSV)", "texture": "Texture (LBP)",
         "edge": "Edge orientation", "dct": "DCT (frequency)",
         "classical_concat": "Classical (all four)"}
PHRASE = {"clip": "similar overall content (CLIP)", "color": "similar colour distribution",
          "texture": "similar texture pattern", "edge": "similar edge / shape orientation",
          "dct": "similar frequency content", "classical_concat": "similar classical descriptors"}
SHORT = {"clip": "content", "color": "colour", "texture": "texture", "edge": "edges",
         "dct": "frequency", "classical_concat": "classical"}

# strength bands by top-percent of the gallery
BANDS = ((1.0, "very strong"), (5.0, "strong"), (20.0, "moderate"))
MAX_REASONS = 3


def strength(top_pct: float) -> str | None:
    for limit, word in BANDS:
        if top_pct <= limit:
            return word
    return None


@dataclass
class Explanation:
    gid: int
    rows: list[dict] = field(default_factory=list)   # one dict per retriever (see module doc)
    fused: float = 0.0
    reasons: list[str] = field(default_factory=list)  # e.g. "similar colour distribution (top 1%)"
    short: str = ""                                    # e.g. "colour, edges"

    @property
    def text(self) -> str:
        if not self.reasons:
            return "Weak similarity on every descriptor (none in the top 20% of the gallery)."
        return "Returned because of " + "; ".join(self.reasons) + "."


def explain_results(sims: dict[str, np.ndarray], ids: np.ndarray, weights: dict[str, float],
                    norm: str = DEFAULT_NORM) -> list[Explanation]:
    """Explain each id in `ids`.

    sims   : {retriever: (N,) cosine of the query to the whole gallery}
             (SearchEngine.all_similarities)
    weights: weights the search actually used (any scale; rescaled to sum to 1),
             e.g. {'clip': .95, 'classical_concat': .05} or {'color': 1.0}
    norm   : normalisation the search used (src.fusion)
    """
    ids = np.asarray(ids, dtype=np.int64)
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        raise ValueError("at least one weight must be > 0")
    w = {k: (weights.get(k, 0.0) / total if weights.get(k, 0.0) > 0 else 0.0) for k in sims}
    names = [n for n in sims if n != "classical_concat" or w[n] > 0]

    normed = {n: normalize_scores(sims[n], norm) for n in names}
    zed = {n: normalize_scores(sims[n], "zscore") for n in names}
    out = []
    for g in ids:
        rows = []
        for n in names:
            s = float(sims[n][g])
            rows.append({
                "retriever": n, "label": LABEL[n], "cosine": s,
                "top_pct": float((sims[n] > s).mean() * 100.0),
                "z": float(zed[n][g]), "weight": float(w[n]),
                "contribution": float(w[n] * normed[n][g]),
            })
        fused = float(sum(r["contribution"] for r in rows))
        # weighted retrievers first (largest contribution), then supporting evidence (rarest first)
        used = sorted((r for r in rows if r["weight"] > 0), key=lambda r: -r["contribution"])
        rest = sorted((r for r in rows if r["weight"] == 0), key=lambda r: r["top_pct"])
        reasons, shorts = [], []
        for r in used + rest:
            band = strength(r["top_pct"])
            if band is None or len(reasons) >= MAX_REASONS:
                continue
            reasons.append(f"{PHRASE[r['retriever']]} ({band}, top {r['top_pct']:.1f}%)")
            shorts.append(SHORT[r["retriever"]])
        out.append(Explanation(int(g), rows, fused, reasons, ", ".join(shorts)))
    return out


# ------------------------------------------------------------------ visual helpers
def hue_histogram_image(img_bgr: np.ndarray, size: tuple[int, int] = (288, 110),
                        bins: int = 36) -> np.ndarray:
    """Hue histogram (weighted by saturation, so grey pixels don't count) drawn as bars
    coloured by hue. Returns a BGR uint8 image of `size` = (width, height)."""
    w, h = size
    hsv = cv2.cvtColor(cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2HSV)
    hist = np.bincount((hsv[..., 0].astype(np.int64) * bins // 180).ravel(),
                       weights=(hsv[..., 1] / 255.0).ravel(), minlength=bins)[:bins]
    hist = hist / max(hist.max(), 1e-9)
    canvas = np.full((h, w, 3), 245, np.uint8)
    bw = w / bins
    for i, v in enumerate(hist):
        colour = cv2.cvtColor(np.uint8([[[int((i + 0.5) * 180 / bins), 220, 235]]]),
                              cv2.COLOR_HSV2BGR)[0, 0].tolist()
        bar = int(v * (h - 2))
        if bar < 1:
            continue
        x0, x1 = int(i * bw), max(int((i + 1) * bw) - 1, int(i * bw) + 1)
        cv2.rectangle(canvas, (x0, h - 1 - bar), (x1, h - 1), colour, -1)
    return canvas


def edge_map(img_bgr: np.ndarray, size: int = 160) -> np.ndarray:
    """Sobel gradient magnitude (what the edge descriptor looks at), scaled to the
    99th percentile. Returns a BGR uint8 image, size x size."""
    g = cv2.cvtColor(cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    scale = max(float(np.percentile(mag, 99)), 1e-6)
    return cv2.cvtColor(np.clip(mag / scale * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
