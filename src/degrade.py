"""Query degradations and restoration filters (Person A).

Public API (interface contract):
    degrade(img_bgr, kind, level)  kind in {'noise', 'blur', 'jpeg'}
    restore(img_bgr, kind)         kind in {'median', 'gaussian'}

Image convention: OpenCV BGR, uint8, shape (H, W, 3).
"""

from __future__ import annotations

import cv2
import numpy as np

DEGRADATIONS = ("noise", "blur", "jpeg")
RESTORATIONS = ("median", "gaussian")


def _check(img_bgr: np.ndarray) -> None:
    if (not isinstance(img_bgr, np.ndarray) or img_bgr.dtype != np.uint8
            or img_bgr.ndim != 3 or img_bgr.shape[2] != 3):
        raise ValueError("img_bgr must be a uint8 array of shape (H, W, 3)")


def degrade(img_bgr: np.ndarray, kind: str, level: int,
            seed: int | None = None) -> np.ndarray:
    """Return a degraded copy of the image.

    kind='noise': additive Gaussian noise, `level` = sigma in grey levels
                  (e.g. 10 / 25 / 50). `seed` makes the noise reproducible.
    kind='blur' : Gaussian blur with an odd square kernel, `level` = kernel size
                  (e.g. 5 / 9 / 15); sigma is derived by OpenCV from the size.
    kind='jpeg' : JPEG re-compression, `level` = quality 1-100 (e.g. 30 / 10).
    """
    _check(img_bgr)
    if kind == "noise":
        if level < 0:
            raise ValueError("noise sigma must be >= 0")
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, float(level), img_bgr.shape).astype(np.float32)
        out = img_bgr.astype(np.float32) + noise
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    if kind == "blur":
        if level < 1 or level % 2 == 0:
            raise ValueError("blur level must be an odd kernel size >= 1")
        return cv2.GaussianBlur(img_bgr, (int(level), int(level)), 0)
    if kind == "jpeg":
        if not 1 <= level <= 100:
            raise ValueError("jpeg quality must be in 1..100")
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raise ValueError(f"unknown degradation '{kind}', expected one of {DEGRADATIONS}")


def restore(img_bgr: np.ndarray, kind: str, ksize: int = 3) -> np.ndarray:
    """Return a filtered copy (simple restoration before feature extraction).

    kind='median'  : median filter, odd `ksize` (good against impulsive/Gaussian noise).
    kind='gaussian': Gaussian smoothing, odd `ksize`.
    """
    _check(img_bgr)
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer >= 3")
    if kind == "median":
        return cv2.medianBlur(img_bgr, int(ksize))
    if kind == "gaussian":
        return cv2.GaussianBlur(img_bgr, (int(ksize), int(ksize)), 0)
    raise ValueError(f"unknown restoration '{kind}', expected one of {RESTORATIONS}")