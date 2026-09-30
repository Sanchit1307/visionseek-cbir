"""Image quality analyzer (Person A, WP7).

Public API (interface contract):
    analyze(img_bgr) -> {'sharpness', 'noise', 'brightness', 'contrast',
                         'jpeg_blockiness', 'bucket'}

Metrics (all on the image at native resolution; images whose longest side is
above `max_side` are first shrunk with INTER_AREA, so phone photos are handled):
    sharpness        variance of the Laplacian of the grey image (higher = sharper)
    noise            Immerkaer (1996) noise standard deviation estimate, grey levels
    brightness       mean of the HSV V channel (0-255)
    contrast         standard deviation of the grey image
    jpeg_blockiness  mean gradient across 8x8 block boundaries divided by the mean
                     gradient inside blocks (about 1.0 = no visible blocking)

Bucket (first rule that fires, in this order):
    noisy -> compressed -> dark/low-contrast -> blurred -> clean
Noise is tested first because it inflates the Laplacian variance; darkness before
blur because darkening lowers the Laplacian variance.

Thresholds are calibrated on the Flowers-102 val split by scripts/calibrate_quality.py,
which writes src/quality_thresholds.json. Without that file, uncalibrated placeholder
values are used (see is_calibrated()).

Image convention: OpenCV BGR, uint8, shape (H, W, 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

BUCKETS = ("clean", "blurred", "noisy", "compressed", "dark/low-contrast")
METRIC_NAMES = ("sharpness", "noise", "brightness", "contrast", "jpeg_blockiness")

THRESHOLDS_PATH = Path(__file__).with_name("quality_thresholds.json")

# Placeholders used only until calibrate_quality.py has been run.
DEFAULT_THRESHOLDS = {
    "noise_max": 6.0,          # noise > this          -> noisy
    "blockiness_max": 1.15,    # blockiness > this     -> compressed
    "brightness_min": 50.0,    # brightness < this     -> dark/low-contrast
    "contrast_min": 25.0,      # contrast < this       -> dark/low-contrast
    "sharpness_min": 30.0,     # sharpness < this      -> blurred
}
THRESHOLD_KEYS = tuple(DEFAULT_THRESHOLDS)

_IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
_EPS = 1e-6
_cache: dict = {}


# ----------------------------------------------------------------------------
# Thresholds
# ----------------------------------------------------------------------------
def load_thresholds(path: Path | None = None) -> dict[str, float]:
    """Calibrated thresholds from JSON if present, else the placeholders."""
    p = Path(path) if path is not None else THRESHOLDS_PATH
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: float(data[k]) for k in THRESHOLD_KEYS}
    return dict(DEFAULT_THRESHOLDS)


def is_calibrated() -> bool:
    return THRESHOLDS_PATH.exists()


def _thresholds() -> dict[str, float]:
    if "th" not in _cache:
        _cache["th"] = load_thresholds()
    return _cache["th"]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _check(img_bgr: np.ndarray) -> None:
    if (not isinstance(img_bgr, np.ndarray) or img_bgr.dtype != np.uint8
            or img_bgr.ndim != 3 or img_bgr.shape[2] != 3):
        raise ValueError("img_bgr must be a uint8 array of shape (H, W, 3)")
    if min(img_bgr.shape[:2]) < 16:
        raise ValueError("image is too small to analyse (need at least 16x16)")


def _blockiness(gray_u8: np.ndarray) -> float:
    g = gray_u8.astype(np.float64)
    dh = np.abs(np.diff(g, axis=1))                     # (H, W-1): column j -> j+1
    dv = np.abs(np.diff(g, axis=0))                     # (H-1, W)
    bh = (np.arange(dh.shape[1]) % 8) == 7              # step across a block edge
    bv = (np.arange(dv.shape[0]) % 8) == 7
    boundary = np.concatenate([dh[:, bh].ravel(), dv[bv, :].ravel()])
    interior = np.concatenate([dh[:, ~bh].ravel(), dv[~bv, :].ravel()])
    return float((boundary.mean() + _EPS) / (interior.mean() + _EPS))


def compute_metrics(img_bgr: np.ndarray, max_side: int | None = 1024) -> dict[str, float]:
    """The five raw quality metrics (no bucket)."""
    _check(img_bgr)
    h, w = img_bgr.shape[:2]
    if max_side is not None and max(h, w) > max_side:
        s = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (max(16, int(round(w * s))), max(16, int(round(h * s)))),
                             interpolation=cv2.INTER_AREA)

    gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float64)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    conv = cv2.filter2D(gray, cv2.CV_64F, _IMMERKAER_KERNEL, borderType=cv2.BORDER_REPLICATE)
    inner = np.abs(conv[1:-1, 1:-1])
    noise = float(np.sqrt(np.pi / 2.0) * inner.mean() / 6.0)

    brightness = float(img_bgr.max(axis=2).mean())      # HSV V = max(B, G, R)
    contrast = float(gray.std())
    return {"sharpness": sharpness, "noise": noise, "brightness": brightness,
            "contrast": contrast, "jpeg_blockiness": _blockiness(gray_u8)}


def assign_bucket(metrics: dict[str, float], thresholds: dict[str, float] | None = None) -> str:
    """Bucket label from raw metrics (rule order in the module docstring)."""
    th = thresholds if thresholds is not None else _thresholds()
    if metrics["noise"] > th["noise_max"]:
        return "noisy"
    if metrics["jpeg_blockiness"] > th["blockiness_max"]:
        return "compressed"
    if metrics["brightness"] < th["brightness_min"] or metrics["contrast"] < th["contrast_min"]:
        return "dark/low-contrast"
    if metrics["sharpness"] < th["sharpness_min"]:
        return "blurred"
    return "clean"


def analyze(img_bgr: np.ndarray, thresholds: dict[str, float] | None = None,
            max_side: int | None = 1024) -> dict:
    """Quality metrics plus a bucket label for a BGR uint8 image."""
    m = compute_metrics(img_bgr, max_side=max_side)
    return {**m, "bucket": assign_bucket(m, thresholds)}