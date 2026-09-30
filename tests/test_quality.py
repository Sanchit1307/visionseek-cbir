"""Sanity checks for src/quality.py on synthetic images (no dataset needed).

Run from the repo root:  python tests\\test_quality.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.degrade import degrade  # noqa: E402
from src.quality import BUCKETS, DEFAULT_THRESHOLDS, analyze, compute_metrics  # noqa: E402


def make_image(h: int = 320, w: int = 320) -> np.ndarray:
    """Smooth colour gradients plus a few sharp-edged shapes (natural-ish structure)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([100 + 60 * np.sin(xx / 47.0), 120 + 50 * np.cos(yy / 39.0),
                     110 + 40 * np.sin((xx + yy) / 61.0)], axis=-1)
    img = np.clip(base, 0, 255).astype(np.uint8)
    cv2.circle(img, (110, 120), 55, (30, 200, 240), -1)
    cv2.rectangle(img, (190, 170), (280, 260), (220, 60, 40), -1)
    cv2.line(img, (20, 300), (300, 40), (255, 255, 255), 3)
    return img


def main() -> None:
    img = make_image()
    m0 = compute_metrics(img)

    # noise estimate: Immerkaer recovers the injected sigma on a mostly smooth image
    for sigma in (10, 25):
        est = compute_metrics(degrade(img, "noise", sigma, seed=0))["noise"]
        assert abs(est - sigma) < 0.25 * sigma + 1.5, (sigma, est)
    assert m0["noise"] < 3.0, m0
    print("ok: noise estimate tracks sigma (clean %.2f)" % m0["noise"])

    # sharpness falls with blur, rises with noise
    s = [compute_metrics(degrade(img, "blur", k))["sharpness"] for k in (5, 9, 15)]
    assert m0["sharpness"] > s[0] > s[1] > s[2], (m0["sharpness"], s)
    assert compute_metrics(degrade(img, "noise", 25, seed=0))["sharpness"] > m0["sharpness"]
    print("ok: sharpness ordering")

    # blockiness: about 1 on an uncompressed image, clearly higher after JPEG, q10 > q30
    b30 = compute_metrics(degrade(img, "jpeg", 30))["jpeg_blockiness"]
    b10 = compute_metrics(degrade(img, "jpeg", 10))["jpeg_blockiness"]
    assert m0["jpeg_blockiness"] < 1.15 and b10 > b30 > m0["jpeg_blockiness"] + 0.1, (m0, b30, b10)
    print("ok: blockiness %.2f -> q30 %.2f -> q10 %.2f" % (m0["jpeg_blockiness"], b30, b10))

    # brightness / contrast
    dark = (img.astype(np.float32) * 0.4).astype(np.uint8)
    md = compute_metrics(dark)
    assert md["brightness"] < 0.5 * m0["brightness"] and md["contrast"] < 0.5 * m0["contrast"]
    assert abs(compute_metrics(np.full((32, 32, 3), 200, np.uint8))["brightness"] - 200) < 1e-9
    print("ok: brightness / contrast")

    # analyze(): contract keys, valid bucket, and each degradation lands in its bucket
    r = analyze(img)
    assert set(r) == {"sharpness", "noise", "brightness", "contrast", "jpeg_blockiness", "bucket"}
    assert r["bucket"] in BUCKETS
    th = dict(DEFAULT_THRESHOLDS)
    assert analyze(img, th)["bucket"] == "clean", analyze(img, th)
    assert analyze(degrade(img, "noise", 50, seed=0), th)["bucket"] == "noisy"
    assert analyze(degrade(img, "jpeg", 10), th)["bucket"] == "compressed"
    assert analyze(degrade(img, "blur", 15), th)["bucket"] == "blurred"
    assert analyze(dark, th)["bucket"] == "dark/low-contrast"
    print("ok: analyze() keys and buckets")

    # input validation and large-image handling
    for bad in (np.zeros((10, 10), np.uint8), np.zeros((40, 40, 3), np.float32), np.zeros((8, 8, 3), np.uint8)):
        try:
            analyze(bad)
        except ValueError:
            continue
        raise AssertionError("invalid input accepted")
    big = cv2.resize(img, (2400, 2400), interpolation=cv2.INTER_LINEAR)
    assert analyze(big)["bucket"] in BUCKETS
    print("ok: validation and max_side")
    print("\nquality.py tests passed")


if __name__ == "__main__":
    main()