"""Classical image-processing descriptors (Person A).

Public API (interface contract):
    extract_classical(img_bgr) -> {'color', 'texture', 'edge', 'dct'}
Each descriptor is a 1-D float32 vector with unit L2 norm, so the inner
product of two descriptors equals their cosine similarity.

Image convention: OpenCV BGR, uint8, shape (H, W, 3).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# ----------------------------------------------------------------------------
# Configuration (tunable constants)
# ----------------------------------------------------------------------------
IMG_SIZE = 128                 # every image is resized to IMG_SIZE x IMG_SIZE
HSV_BINS = (16, 4, 4)          # H, S, V bins        -> 256 dims
LBP_POINTS = 8                 # LBP neighbours
LBP_RADIUS = 1                 # LBP radius
LBP_GRID = (3, 3)              # spatial cells       -> 10 * 9 = 90 dims
EDGE_BINS = 9                  # orientation bins over [0, 180) degrees
EDGE_GRID = (3, 3)             # spatial cells       -> 9 * 9 = 81 dims
DCT_BLOCK = 8                  # block size for block-DCT
DCT_KEEP = 6                   # keep the DCT_KEEP x DCT_KEEP low-freq corner

FEATURE_NAMES = ("color", "texture", "edge", "dct")

_EPS = 1e-12


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _l2(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector to float32 (zero vector stays zero, no NaN)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if n < _EPS:
        return np.zeros(v.shape, dtype=np.float32)
    return (v / n).astype(np.float32)


def _prepare(img_bgr: np.ndarray) -> np.ndarray:
    """Validate the input and resize to IMG_SIZE x IMG_SIZE (INTER_AREA)."""
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("img_bgr must be a numpy array")
    if img_bgr.dtype != np.uint8 or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be a uint8 array of shape (H, W, 3)")
    return cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def _grid_slices(size: int, parts: int) -> list[slice]:
    edges = np.linspace(0, size, parts + 1).astype(int)
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(parts)]


def _hellinger(hist: np.ndarray) -> np.ndarray:
    """L1-normalise then square-root (Hellinger kernel): cosine of the result
    equals the Bhattacharyya coefficient, which suits histograms."""
    hist = hist.astype(np.float64)
    return np.sqrt(hist / (hist.sum() + _EPS))


# ----------------------------------------------------------------------------
# Individual descriptors (operate on the already resized BGR image)
# ----------------------------------------------------------------------------
def _color(img: np.ndarray) -> np.ndarray:
    """3-D HSV histogram (Unit 1)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # H in [0,180), S,V in [0,256)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HSV_BINS),
                        [0, 180, 0, 256, 0, 256])
    return _l2(_hellinger(hist.ravel()))


def _texture(img: np.ndarray) -> np.ndarray:
    """Uniform LBP histogram over a spatial grid (Unit 2)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp = lbp.astype(np.int64)
    n_bins = LBP_POINTS + 2  # uniform LBP yields codes 0 .. P+1
    h, w = lbp.shape
    parts = []
    for ys in _grid_slices(h, LBP_GRID[0]):
        for xs in _grid_slices(w, LBP_GRID[1]):
            cell = lbp[ys, xs].ravel()
            hist = np.bincount(cell, minlength=n_bins)[:n_bins]
            parts.append(hist / (hist.sum() + _EPS))
    return _l2(np.sqrt(np.concatenate(parts)))


def _edge(img: np.ndarray) -> np.ndarray:
    """Sobel gradient-orientation histogram, magnitude weighted (Unit 3)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3).astype(np.float64)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3).astype(np.float64)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)                 # unsigned, [0, pi)
    idx = np.minimum((ang / np.pi * EDGE_BINS).astype(np.int64), EDGE_BINS - 1)
    h, w = gray.shape
    parts = []
    for ys in _grid_slices(h, EDGE_GRID[0]):
        for xs in _grid_slices(w, EDGE_GRID[1]):
            hist = np.bincount(idx[ys, xs].ravel(),
                               weights=mag[ys, xs].ravel(),
                               minlength=EDGE_BINS)[:EDGE_BINS]
            parts.append(hist / (hist.sum() + _EPS))
    return _l2(np.sqrt(np.concatenate(parts)))


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal 1-D DCT-II matrix of size n x n."""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    m = np.cos((2 * i + 1) * k * np.pi / (2 * n))
    m[0, :] *= np.sqrt(1.0 / n)
    m[1:, :] *= np.sqrt(2.0 / n)
    return m


_D = _dct_matrix(DCT_BLOCK)


def _dct(img: np.ndarray) -> np.ndarray:
    """Block-DCT low-frequency coefficient statistics (Unit 6).

    Grayscale image -> 8x8 blocks -> 2-D DCT per block -> for the top-left
    DCT_KEEP x DCT_KEEP coefficients take mean |c| and std(c) over all blocks,
    log-compress, mean-centre (removes the common offset), L2-normalise.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) - 128.0
    nb = IMG_SIZE // DCT_BLOCK
    blocks = gray.reshape(nb, DCT_BLOCK, nb, DCT_BLOCK).transpose(0, 2, 1, 3)
    coef = np.matmul(np.matmul(_D, blocks), _D.T)            # (nb, nb, 8, 8)
    coef = coef[:, :, :DCT_KEEP, :DCT_KEEP].reshape(-1, DCT_KEEP * DCT_KEEP)
    mean_abs = np.log1p(np.abs(coef).mean(axis=0))
    std = np.log1p(coef.std(axis=0))
    v = np.concatenate([mean_abs, std])
    return _l2(v - v.mean())


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def color_feature(img_bgr: np.ndarray) -> np.ndarray:
    return _color(_prepare(img_bgr))


def texture_feature(img_bgr: np.ndarray) -> np.ndarray:
    return _texture(_prepare(img_bgr))


def edge_feature(img_bgr: np.ndarray) -> np.ndarray:
    return _edge(_prepare(img_bgr))


def dct_feature(img_bgr: np.ndarray) -> np.ndarray:
    return _dct(_prepare(img_bgr))


def extract_classical(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Extract all classical descriptors from a BGR uint8 image.

    Returns {'color': (256,), 'texture': (90,), 'edge': (81,), 'dct': (72,)},
    each float32 and L2-normalised.
    """
    img = _prepare(img_bgr)
    return {
        "color": _color(img),
        "texture": _texture(img),
        "edge": _edge(img),
        "dct": _dct(img),
    }