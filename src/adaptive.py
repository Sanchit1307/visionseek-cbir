"""Adaptive fusion weights from the query quality bucket (Person A, WP10).

Public API (interface contract):
    adaptive_weights(quality: dict) -> dict[str, float]

`quality` is the dict returned by src.quality.analyze (only 'bucket' is used).
The weight table is written by scripts/run_adaptive.py to results/adaptive_weights.json.
Missing or corrupt table -> falls back to Person B's tuned fixed hybrid weights.

Use with the engine (norm must match the table):
    from src.adaptive import adaptive_weights, load_table
    from src.quality import analyze
    _, norm = load_table()
    res = engine.search_fused(img, adaptive_weights(analyze(img)), k=10, norm=norm)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.dataset import RESULTS_DIR
from src.fusion import DEFAULT_NORM, load_weights
from src.quality import BUCKETS

TABLE_PATH = RESULTS_DIR / "adaptive_weights.json"
FIXED_PATH = RESULTS_DIR / "fusion_weights.json"


def load_table(path: Path | None = None) -> tuple[dict[str, dict[str, float]], str]:
    """({bucket: {retriever: weight}}, norm). Falls back to the fixed hybrid for every bucket."""
    p = Path(path) if path is not None else TABLE_PATH
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        table = {b: {k: float(v) for k, v in e["weights"].items()} for b, e in d["buckets"].items()}
        return table, d.get("norm", DEFAULT_NORM)
    except (OSError, ValueError, KeyError):
        fixed, norm = load_weights(FIXED_PATH)
        return {b: dict(fixed) for b in BUCKETS}, norm


def adaptive_weights(quality: dict, table: dict | None = None) -> dict[str, float]:
    """Fusion weights for a quality dict (uses quality['bucket'])."""
    if table is None:
        table, _ = load_table()
    bucket = quality.get("bucket", "clean")
    if bucket in table:
        return dict(table[bucket])
    fixed, _ = load_weights(FIXED_PATH)
    return dict(fixed)