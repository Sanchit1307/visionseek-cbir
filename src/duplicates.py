"""Near-duplicate detection (Person B, WP12).

Three tiers per (query, gallery image) pair:
    duplicate   exact / near-duplicate (resized, re-compressed, cropped, brightness-shifted copy)
    similar     same kind of thing but a different photo
    different
decided from the CLIP cosine and, optionally, a 64-bit perceptual hash (pHash).

Thresholds are NOT chosen by eye: scripts/eval_duplicates.py builds synthetic copies
of gallery images, sweeps every threshold, and writes results/duplicate_thresholds.json
from the precision-recall curves; `load_thresholds` reads it. The built-in defaults below are the
values that script produced on Flowers-102 (500 originals x 4 copies, duplicate precision >= 0.99),
so the app behaves sensibly before that JSON is committed; re-run the script to regenerate them.

Image convention: OpenCV BGR uint8.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.degrade import degrade

# ------------------------------------------------------------------ synthetic copies
DUP_KINDS = ("resize", "jpeg", "crop", "brightness")
RESIZE_SCALE = 0.5        # resized copy: half the width and height
JPEG_QUALITY = 30         # same quality as the robustness experiment
CROP_FRACTION = 0.10      # cropped copy: 10% of each side removed in total (5% per border)
BRIGHTNESS_SHIFT = 40     # brightness copy: +40 grey levels (saturating)


def make_duplicate(img_bgr: np.ndarray, kind: str) -> np.ndarray:
    """One synthetic copy of the image (see the constants above for the strengths)."""
    h, w = img_bgr.shape[:2]
    if kind == "resize":
        size = (max(8, int(w * RESIZE_SCALE)), max(8, int(h * RESIZE_SCALE)))
        return cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    if kind == "jpeg":
        return degrade(img_bgr, "jpeg", JPEG_QUALITY)
    if kind == "crop":
        dy, dx = int(h * CROP_FRACTION / 2), int(w * CROP_FRACTION / 2)
        return img_bgr[dy:h - dy, dx:w - dx].copy()
    if kind == "brightness":
        return cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=BRIGHTNESS_SHIFT)
    raise ValueError(f"unknown copy kind '{kind}', expected one of {DUP_KINDS}")


def make_duplicates(img_bgr: np.ndarray, kinds: tuple[str, ...] = DUP_KINDS) -> dict[str, np.ndarray]:
    return {k: make_duplicate(img_bgr, k) for k in kinds}


# ------------------------------------------------------------------ perceptual hash
PHASH_BITS = 64


def phash_bits(img_bgr: np.ndarray) -> np.ndarray:
    """64-bit pHash as a (64,) uint8 array of 0/1: grey, 32x32, DCT, the 8x8 low
    frequencies thresholded at their median (same recipe as the `imagehash` library)."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(g)[:8, :8]
    return (low > np.median(low)).ravel().astype(np.uint8)


def hamming_matrix(q_bits: np.ndarray, g_bits: np.ndarray) -> np.ndarray:
    """(Q, 64) x (N, 64) 0/1 arrays -> (Q, N) int16 Hamming distances (matrix product)."""
    a = np.asarray(q_bits, np.float32) * 2 - 1
    b = np.asarray(g_bits, np.float32) * 2 - 1
    return np.rint((PHASH_BITS - a @ b.T) / 2).astype(np.int16)


# ------------------------------------------------------------------ tiers
RULES = ("clip", "phash", "any", "both")
TIERS = ("duplicate", "similar", "different")


@dataclass(frozen=True)
class Thresholds:
    dup_cos: float = 0.99     # CLIP cosine: lowest value with precision >= 0.99 (Flowers-102 copies)
    sim_cos: float = 0.90     # best F1 for same-class retrieval (P 0.59, R 0.48)
    ham: int = 7              # pHash: duplicate if Hamming distance <= ham (precision 0.99, recall 0.78)
    rule: str = "any"         # clip | phash | any (either) | both (CLIP and pHash); "any" had the best F1


def is_duplicate(cos: np.ndarray, ham: np.ndarray | None, th: Thresholds) -> np.ndarray:
    c = np.asarray(cos) >= th.dup_cos
    if th.rule == "clip":
        return c
    if th.rule not in RULES:
        raise ValueError(f"unknown rule '{th.rule}', expected one of {RULES}")
    if ham is None:
        raise ValueError(f"rule '{th.rule}' needs pHash Hamming distances")
    p = np.asarray(ham) <= th.ham
    return {"phash": p, "any": c | p, "both": c & p}[th.rule]


def classify(cos: np.ndarray, ham: np.ndarray | None, th: Thresholds) -> np.ndarray:
    """Tier name for every pair: duplicate / similar / different."""
    dup = is_duplicate(cos, ham, th)
    return np.where(dup, "duplicate", np.where(np.asarray(cos) >= th.sim_cos, "similar", "different"))


def find_matches(cos: np.ndarray, ham: np.ndarray | None, th: Thresholds,
                 exclude_id: int | None = None, top: int = 50) -> pd.DataFrame:
    """Gallery images that are duplicates or similar to the query, best first.

    cos: (N,) CLIP cosine of the query to every gallery image; ham: (N,) or None."""
    cos = np.asarray(cos, dtype=np.float32)
    tier = classify(cos, ham, th)
    keep = tier != "different"
    if exclude_id is not None:
        keep[exclude_id] = False
    ids = np.nonzero(keep)[0]
    ids = ids[np.argsort(-cos[ids], kind="stable")][:top]
    return pd.DataFrame({"gid": ids, "cos": cos[ids],
                         "hamming": (np.asarray(ham)[ids] if ham is not None else np.nan),
                         "tier": tier[ids]})


def duplicate_pairs(vecs: np.ndarray, thr: float, chunk: int = 1024,
                    max_pairs: int = 100_000) -> pd.DataFrame:
    """All gallery pairs i < j with cosine >= thr (whole-library scan), best first."""
    v = np.asarray(vecs, dtype=np.float32)
    parts = []
    for s in range(0, len(v), chunk):
        sims = v[s:s + chunk] @ v.T
        r, c = np.nonzero(sims >= thr)
        keep = c > r + s
        r, c = r[keep], c[keep]
        if len(r) > max_pairs:
            top = np.argpartition(-sims[r, c], max_pairs - 1)[:max_pairs]
            r, c = r[top], c[top]
        parts.append(np.stack([r + s, c, sims[r, c]], axis=1))
    a = np.concatenate(parts) if parts else np.zeros((0, 3))
    df = pd.DataFrame({"i": a[:, 0].astype(np.int64), "j": a[:, 1].astype(np.int64), "cos": a[:, 2]})
    return df.sort_values("cos", ascending=False).head(max_pairs).reset_index(drop=True)


# ------------------------------------------------------------------ threshold selection
def threshold_curve(pos_scores: np.ndarray, all_scores: np.ndarray, taus: np.ndarray) -> pd.DataFrame:
    """Precision / recall of the rule "score >= tau" for every tau.

    pos_scores: score of each relevant pair (its count is the recall denominator)
    all_scores: score of EVERY pair the rule is applied to, relevant ones included
    Precision is 1.0 by convention when nothing is retrieved (recall is then 0)."""
    a = np.sort(np.asarray(all_scores, dtype=np.float64).ravel())
    p = np.sort(np.asarray(pos_scores, dtype=np.float64).ravel())
    taus = np.asarray(taus, dtype=np.float64)
    retrieved = len(a) - np.searchsorted(a, taus, side="left")
    tp = len(p) - np.searchsorted(p, taus, side="left")
    precision = np.where(retrieved > 0, tp / np.maximum(retrieved, 1), 1.0)
    recall = tp / max(len(p), 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return pd.DataFrame({"tau": taus, "precision": precision, "recall": recall, "f1": f1,
                         "retrieved": retrieved})


def pick_threshold(curve: pd.DataFrame, rule: str = "precision", min_precision: float = 0.99) -> pd.Series:
    """Operating point from a curve.

    rule 'precision': the LOWEST tau (highest recall) whose precision >= min_precision,
                      falling back to max F1 if none reaches it
    rule 'f1'       : the tau with the best F1 (ties -> the higher tau)"""
    live = curve[curve["retrieved"] > 0]
    if rule == "precision":
        ok = live[live["precision"] >= min_precision]
        if len(ok):
            return ok.sort_values("tau").iloc[0]
    elif rule != "f1":
        raise ValueError("rule must be 'precision' or 'f1'")
    return live.sort_values(["f1", "tau"], ascending=[False, False]).iloc[0]


def save_thresholds(path: Path, th: Thresholds, **meta) -> None:
    Path(path).write_text(json.dumps({**asdict(th), **meta}, indent=2), encoding="utf-8")


def load_thresholds(path: Path) -> Thresholds:
    """Tuned thresholds from `path`, or the placeholder defaults if missing / invalid."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        th = Thresholds(dup_cos=float(d["dup_cos"]), sim_cos=float(d["sim_cos"]),
                        ham=int(d["ham"]), rule=str(d["rule"]))
        if th.rule in RULES and th.sim_cos <= th.dup_cos:
            return th
    except (OSError, ValueError, KeyError):
        pass
    return Thresholds()
