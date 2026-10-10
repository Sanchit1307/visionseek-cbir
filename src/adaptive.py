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
from src.fusion import DEFAULT_NORM, NORMS, load_weights
from src.quality import BUCKETS

TABLE_PATH = RESULTS_DIR / "adaptive_weights.json"
FIXED_PATH = RESULTS_DIR / "fusion_weights.json"


def _valid(w: dict) -> bool:
    """A usable weight dict: non-empty, finite-looking, non-negative, positive sum."""
    return (isinstance(w, dict) and len(w) > 0
            and all(isinstance(v, (int, float)) and v >= 0 for v in w.values())
            and sum(w.values()) > 0)


def load_table(path: Path | None = None) -> tuple[dict[str, dict[str, float]], str]:
    """({bucket: {retriever: weight}}, norm). Falls back to the fixed hybrid for every bucket
    when the file is missing, is not valid JSON, has the wrong structure, or holds invalid
    weights or an unknown normalisation."""
    p = Path(path) if path is not None else TABLE_PATH
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        table = {b: {k: float(v) for k, v in e["weights"].items()} for b, e in d["buckets"].items()}
        norm = d.get("norm", DEFAULT_NORM)
        if norm not in NORMS or not table or not all(_valid(w) for w in table.values()):
            raise ValueError("invalid adaptive table")
        return table, norm
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        fixed, norm = load_weights(FIXED_PATH)
        return {b: dict(fixed) for b in BUCKETS}, norm


def adaptive_weights(quality: dict, table: dict | None = None) -> dict[str, float]:
    """Fusion weights for a quality dict (uses quality['bucket'])."""
    if table is None:
        table, _ = load_table()
    bucket = quality.get("bucket", "clean")
    if bucket in table and _valid(table[bucket]):
        return dict(table[bucket])
    fixed, _ = load_weights(FIXED_PATH)
    return dict(fixed)