"""Tests for src/duplicates.py and scripts/eval_duplicates.py (Person B, WP12).

Run from the repo root:  python tests/test_duplicates.py
Pure logic first (copies, pHash, tiers, curves, pair scan), then the whole evaluation
script on the synthetic gallery from tests/test_pipeline.py with the CLIP stand-in.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

_TMP = Path(tempfile.mkdtemp(prefix="vs_dup_"))
os.environ["VISIONSEEK_RESULTS_DIR"] = str(_TMP / "results")   # before src.dataset is imported

from src import duplicates as dp  # noqa: E402


def structured(seed: int, h: int = 120, w: int = 160) -> np.ndarray:
    """A smooth, blob-like test image (photos are low-frequency; noise would make pHash flaky)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    for _ in range(12):
        c = tuple(int(v) for v in rng.integers(30, 255, 3))
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(10, 40)), c, -1)
    return cv2.GaussianBlur(img, (0, 0), 6)


def test_copies() -> None:
    img = structured(0)
    c = dp.make_duplicates(img)
    assert set(c) == set(dp.DUP_KINDS)
    assert c["resize"].shape == (60, 80, 3)
    assert c["crop"].shape == (108, 144, 3)                       # 10% of each side removed
    assert c["jpeg"].shape == img.shape and (c["jpeg"] != img).any()
    assert c["brightness"].shape == img.shape and c["brightness"].mean() > img.mean() + 10
    assert all(v.dtype == np.uint8 for v in c.values())
    assert (img == structured(0)).all(), "make_duplicates must not modify its input"
    try:
        dp.make_duplicate(img, "rotate")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kind accepted")
    print("ok: synthetic copies (shapes, strengths, input untouched)")


def test_phash() -> None:
    a, b = structured(1), structured(2)
    ha, hb = dp.phash_bits(a), dp.phash_bits(b)
    assert ha.shape == (64,) and ha.dtype == np.uint8 and set(np.unique(ha)) <= {0, 1}
    assert (dp.phash_bits(a) == ha).all()
    d_other = int(dp.hamming_matrix(ha[None], hb[None])[0, 0])
    d_copy = [int(dp.hamming_matrix(ha[None], dp.phash_bits(v)[None])[0, 0])
              for k, v in dp.make_duplicates(a).items() if k in ("resize", "jpeg", "brightness")]
    assert max(d_copy) <= 6 and d_other > max(d_copy), (d_copy, d_other)
    # matrix version == brute force
    rng = np.random.default_rng(0)
    q, g = rng.integers(0, 2, (5, 64), np.uint8), rng.integers(0, 2, (7, 64), np.uint8)
    brute = np.array([[int((x != y).sum()) for y in g] for x in q])
    assert (dp.hamming_matrix(q, g) == brute).all()
    print(f"ok: pHash (copies {d_copy} vs different image {d_other}), hamming_matrix == brute force")


def test_tiers_and_pairs() -> None:
    cos = np.array([0.99, 0.96, 0.90, 0.85, 0.5], np.float32)
    ham = np.array([0, 20, 3, 30, 40])
    th = dp.Thresholds(dup_cos=0.95, sim_cos=0.80, ham=5, rule="clip")
    assert dp.classify(cos, None, th).tolist() == ["duplicate", "duplicate", "similar", "similar", "different"]
    for rule, dups in (("phash", [True, False, True, False, False]),
                       ("any", [True, True, True, False, False]),
                       ("both", [True, False, False, False, False])):
        got = dp.is_duplicate(cos, ham, dp.Thresholds(0.95, 0.80, 5, rule))
        assert got.tolist() == dups, rule
    try:
        dp.is_duplicate(cos, None, dp.Thresholds(rule="any"))
    except ValueError:
        pass
    else:
        raise AssertionError("pHash rule without Hamming accepted")
    m = dp.find_matches(cos, None, th, exclude_id=0)
    assert m["gid"].tolist() == [1, 2, 3] and m["tier"].tolist() == ["duplicate", "similar", "similar"]
    assert dp.find_matches(cos, ham, th, top=2).shape[0] == 2

    rng = np.random.default_rng(3)
    v = rng.normal(size=(60, 8)).astype(np.float32)
    v[41] = v[7] + 0.01 * rng.normal(size=8)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    got = dp.duplicate_pairs(v, 0.98, chunk=16)
    full = v @ v.T
    want = sorted((i, j) for i in range(60) for j in range(i + 1, 60) if full[i, j] >= 0.98)
    assert sorted(zip(got["i"], got["j"])) == want and (7, 41) in want
    assert got["cos"].is_monotonic_decreasing
    print("ok: tiers, rules any/both/phash, find_matches, duplicate_pairs == brute force")


def test_curve_and_pick() -> None:
    rng = np.random.default_rng(4)
    allsc = rng.uniform(0, 1, 500)
    pos = allsc[:40]
    taus = np.array([0.0, 0.3, 0.6, 0.9, 1.1])
    c = dp.threshold_curve(pos, allsc, taus)
    for i, t in enumerate(taus):
        retrieved, tp = int((allsc >= t).sum()), int((pos >= t).sum())
        assert c.loc[i, "retrieved"] == retrieved
        assert abs(c.loc[i, "recall"] - tp / 40) < 1e-12
        assert abs(c.loc[i, "precision"] - (tp / retrieved if retrieved else 1.0)) < 1e-12
    assert c.loc[4, "precision"] == 1.0 and c.loc[4, "recall"] == 0.0   # nothing retrieved

    cur = dp.threshold_curve(np.array([0.9, 0.95, 0.97, 0.99]),
                             np.array([0.9, 0.95, 0.97, 0.99, 0.5, 0.92]), np.round(np.arange(0.4, 1.0, 0.01), 2))
    p = dp.pick_threshold(cur, "precision", 0.99)             # 0.92 is a false positive -> tau > 0.92
    assert 0.92 < p["tau"] <= 0.95 and p["precision"] >= 0.99 and abs(p["recall"] - 0.75) < 1e-9
    assert dp.pick_threshold(cur, "f1")["f1"] == cur["f1"].max()
    # unreachable precision -> falls back to max F1 instead of failing
    bad = dp.threshold_curve(np.array([0.5]), np.array([0.5, 0.9, 0.9]), np.round(np.arange(0.3, 1.0, 0.05), 2))
    assert dp.pick_threshold(bad, "precision", 0.99)["f1"] == bad["f1"].max()
    print("ok: threshold_curve == brute force, pick_threshold rules and fallback")

    tmp = Path(tempfile.mkdtemp())
    try:
        th = dp.Thresholds(0.93, 0.8, 7, "any")
        dp.save_thresholds(tmp / "t.json", th, note="x")
        assert dp.load_thresholds(tmp / "t.json") == th
        assert dp.load_thresholds(tmp / "none.json") == dp.Thresholds()
        (tmp / "bad.json").write_text('{"dup_cos": 0.5, "sim_cos": 0.9, "ham": 1, "rule": "clip"}')
        assert dp.load_thresholds(tmp / "bad.json") == dp.Thresholds()     # sim > dup is rejected
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: save / load thresholds (missing and inconsistent file -> defaults)")


def test_eval_script() -> None:
    import pandas as pd
    import test_pipeline as tp
    from src.features import clip_encoder

    clip_encoder.encode_images = tp.fake_encode_images
    clip_encoder.encode_text = tp.fake_encode_text
    import build_index
    import eval_duplicates as ed

    data, idx = _TMP / "gallery", _TMP / "index"
    tp.make_gallery(data)
    build_index.main(["--dataset", f"folder:{data}", "--index-dir", str(idx)])
    args = ["--dataset", f"folder:{data}", "--index-dir", str(idx), "--limit", "20"]
    ed.main(args)
    res = _TMP / "results"
    for f in ("duplicates_curve.csv", "duplicates_summary.csv", "duplicate_thresholds.json",
              "duplicates_pr.png", "duplicates_false_positives.csv"):
        assert (res / f).exists(), f
    assert (idx / "phash.npy").exists() and (idx / "dup_eval_cache.npz").exists()
    for f in ("duplicates_positive_scores.csv", "duplicates_fp_vs_tau.csv"):
        assert (res / f).exists(), f
    fpt = pd.read_csv(res / "duplicates_fp_vs_tau.csv")
    assert fpt["false_pos"].is_monotonic_decreasing
    assert (fpt["recall"].diff().dropna() <= 1e-12).all()
    ps = pd.read_csv(res / "duplicates_positive_scores.csv")
    assert list(ps.kind) == list(dp.DUP_KINDS) and (ps.cos_min <= ps.cos_median).all()

    curve = pd.read_csv(res / "duplicates_curve.csv")
    for (method, kind), g in curve.groupby(["method", "kind"]):
        # stricter threshold (higher cosine / smaller Hamming distance) can only lose recall
        g = g.sort_values("tau") if method == "clip" else g.sort_values("ham_max")
        d = g["recall"].diff().dropna()
        assert (d <= 1e-12).all() if method == "clip" else (d >= -1e-12).all(), (method, kind)
    assert set(curve.kind) == {"all", *dp.DUP_KINDS}

    summ = pd.read_csv(res / "duplicates_summary.csv")
    assert set(summ.rule) == set(dp.RULES) and len(summ) == 4 * 5
    th = dp.load_thresholds(res / "duplicate_thresholds.json")
    assert th.sim_cos < th.dup_cos and th.rule in dp.RULES
    # the summary row of the CLIP rule equals the curve value at the chosen tau
    row = summ[(summ.rule == "clip") & (summ.kind == "all")].iloc[0]
    cv = curve[(curve.method == "clip") & (curve.kind == "all")]
    ref = cv.iloc[(cv.tau - th.dup_cos).abs().argmin()]
    assert abs(row.recall - ref.recall) < 1e-3 and abs(row.precision - ref.precision) < 1e-3
    print("ok: eval_duplicates.py end to end; thresholds", th)

    n_before = (idx / "dup_eval_cache.npz").stat().st_mtime
    ed.main(args)                                      # second run reuses both caches
    assert (idx / "dup_eval_cache.npz").stat().st_mtime == n_before
    print("ok: cached copies and pHash reused on the second run")


if __name__ == "__main__":
    try:
        test_copies()
        test_phash()
        test_tiers_and_pairs()
        test_curve_and_pick()
        test_eval_script()
        print("\nAll duplicate tests passed")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
