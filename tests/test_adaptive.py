"""Tests for src/adaptive.py (no dataset needed).

Run from the repo root:  python tests\\test_adaptive.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import adaptive  # noqa: E402
from src.adaptive import adaptive_weights, load_table  # noqa: E402
from src.fusion import load_weights  # noqa: E402
from src.quality import BUCKETS  # noqa: E402

FIXED = load_weights(adaptive.FIXED_PATH)[0]    # what every fallback must return


def _write(path: Path, obj) -> Path:
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
    return path


def _is_fallback(tbl_norm) -> bool:
    table, _ = tbl_norm
    return set(table) == set(BUCKETS) and all(w == FIXED for w in table.values())


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        # 1. valid table in the run_adaptive.py format: {"buckets": {name: {"weights": {...}}}}
        good = _write(tmp / "good.json", {"norm": "none", "buckets": {
            "noisy": {"weights": {"clip": 0.85, "classical_concat": 0.15}, "used": "tuned"},
            "clean": {"weights": {"clip": 0.95, "classical_concat": 0.05}}}})
        table, norm = load_table(good)
        assert norm == "none"
        assert adaptive_weights({"bucket": "noisy"}, table) == {"clip": 0.85, "classical_concat": 0.15}
        assert adaptive_weights({"bucket": "clean"}, table) == {"clip": 0.95, "classical_concat": 0.05}
        print("ok: table lookup per bucket")

        # 2. unknown or missing bucket -> fixed hybrid
        assert adaptive_weights({"bucket": "blurred"}, table) == FIXED     # not in this table
        assert adaptive_weights({"bucket": "weird"}, table) == FIXED
        print("ok: unknown bucket falls back to the fixed hybrid")

        # 3. a missing 'bucket' key means clean
        assert adaptive_weights({}, table) == adaptive_weights({"bucket": "clean"}, table)
        print("ok: no bucket -> clean")

        # 4. the returned dict is a copy (callers may mutate it)
        w = adaptive_weights({"bucket": "noisy"}, table)
        w["clip"] = 0.0
        assert table["noisy"]["clip"] == 0.85
        print("ok: returns a copy")

        # 5. missing / corrupt / wrongly structured / invalid files -> fixed hybrid everywhere
        bad = {
            "missing": tmp / "nope.json",
            "not_json": _write(tmp / "a.json", "{not json"),
            "list_root": _write(tmp / "b.json", []),                            # AttributeError path
            "buckets_is_list": _write(tmp / "c.json", {"buckets": [1, 2]}),     # AttributeError path
            "entry_is_str": _write(tmp / "d.json", {"buckets": {"noisy": "x"}}),  # TypeError path
            "no_weights": _write(tmp / "e.json", {"buckets": {"noisy": {}}}),   # KeyError path
            "empty_buckets": _write(tmp / "f.json", {"buckets": {}}),
            "bad_norm": _write(tmp / "g.json", {"norm": "weird", "buckets": {
                "noisy": {"weights": {"clip": 1.0}}}}),
            "negative": _write(tmp / "h.json", {"buckets": {
                "noisy": {"weights": {"clip": -1.0, "classical_concat": 2.0}}}}),
            "zero_sum": _write(tmp / "i.json", {"buckets": {
                "noisy": {"weights": {"clip": 0.0, "classical_concat": 0.0}}}}),
            "non_numeric": _write(tmp / "j.json", {"buckets": {
                "noisy": {"weights": {"clip": "a", "classical_concat": 1}}}}),
        }
        for name, path in bad.items():
            assert _is_fallback(load_table(path)), name
        print(f"ok: {len(bad)} broken tables all fall back to the fixed hybrid")

        # 6. an invalid entry inside a passed-in table is ignored, not returned
        assert adaptive_weights({"bucket": "noisy"}, {"noisy": {"clip": 0.0, "classical_concat": 0.0}}) == FIXED
        print("ok: invalid weights in a passed table are ignored")

        # 7. the committed table (if present) loads and covers every bucket
        if adaptive.TABLE_PATH.exists():
            real, real_norm = load_table()
            assert set(real) == set(BUCKETS) and real_norm in ("zscore", "none")
            assert all(abs(sum(adaptive_weights({"bucket": b}, real).values()) - 1.0) < 1e-6 for b in BUCKETS)
            print("ok: committed results/adaptive_weights.json covers all buckets, weights sum to 1")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nadaptive.py tests passed")


if __name__ == "__main__":
    main()