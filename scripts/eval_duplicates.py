"""Near-duplicate evaluation on synthetic copies; picks the tier thresholds (Person B, WP12).

Run from the repo root after scripts/build_index.py (needs the CLIP index):
    python scripts/eval_duplicates.py               # 500 originals x 4 copies = 2,000 queries
    python scripts/eval_duplicates.py --limit 100   # quick check

Protocol
  * originals = the saved test queries (index/query_ids.npy); each gets 4 copies:
    resized (50%), JPEG q30, 10% crop, brightness +40.
  * a copy is searched against the WHOLE gallery (its original is in it). Rule
    "score >= tau" retrieves every gallery image above tau. True positive = the
    original is retrieved; every other retrieved image is a false positive.
    precision = TP / retrieved, recall = TP / number of copies.
  * scores: CLIP cosine, and 64 - Hamming(pHash). Rules "any" / "both" combine them.
  * duplicate threshold = the lowest tau with precision >= --min-precision (0.99);
    similar threshold = best F1 for "same class" retrieval on the clean originals.
  * a gallery image that is itself a near-copy of another counts as a false positive
    here; results/duplicates_false_positives.csv lists them so you can inspect.

Writes results/duplicates_curve.csv, duplicates_summary.csv, duplicate_thresholds.json,
duplicates_pr.png, duplicates_false_positives.csv; caches index/phash.npy and
index/dup_eval_cache.npz (gitignored, safe to delete).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import duplicates as dp  # noqa: E402
from src.dataset import INDEX_DIR, RESULTS_DIR, bgr_to_rgb, open_gallery  # noqa: E402
from src.features import clip_encoder  # noqa: E402

COS_TAUS = np.round(np.arange(0.30, 1.0001, 0.0025), 4)
HAM_TAUS = np.arange(0, dp.PHASH_BITS + 1)          # tau on the score 64 - Hamming
PARAMS = (f"{dp.DUP_KINDS}|{dp.RESIZE_SCALE}|{dp.JPEG_QUALITY}|{dp.CROP_FRACTION}|{dp.BRIGHTNESS_SHIFT}")


def gallery_phash(gallery, index_dir: Path) -> np.ndarray:
    p = index_dir / "phash.npy"
    if p.exists():
        bits = np.load(p)
        if bits.shape == (len(gallery), dp.PHASH_BITS):
            return bits
    bits = np.stack([dp.phash_bits(gallery.get(g)[0]) for g in tqdm(range(len(gallery)), desc="pHash gallery")])
    np.save(p, bits)
    return bits


def encode_copies(gallery, orig_ids: np.ndarray, index_dir: Path, recompute: bool):
    """CLIP embeddings + pHash bits of every synthetic copy (images are not kept in memory)."""
    cache = index_dir / "dup_eval_cache.npz"
    if cache.exists() and not recompute:
        z = np.load(cache, allow_pickle=False)
        if (str(z["params"]) == PARAMS and len(z["orig"]) == len(orig_ids) * len(dp.DUP_KINDS)
                and np.array_equal(np.unique(z["orig"]), np.unique(orig_ids))):
            print("Synthetic copies: reusing", cache.name)
            return z["clip"], z["bits"], z["orig"], z["kind"]
    clip_l, bits_l, orig_l, kind_l = [], [], [], []
    for s in tqdm(range(0, len(orig_ids), 32), desc="Copies (CLIP + pHash)"):
        chunk = orig_ids[s:s + 32]
        imgs = [gallery.get(int(g))[0] for g in chunk]
        for ki, kind in enumerate(dp.DUP_KINDS):
            copies = [dp.make_duplicate(im, kind) for im in imgs]
            clip_l.append(clip_encoder.encode_images([bgr_to_rgb(c) for c in copies]))
            bits_l.append(np.stack([dp.phash_bits(c) for c in copies]))
            orig_l.append(chunk)
            kind_l.append(np.full(len(chunk), ki))
    out = (np.concatenate(clip_l), np.concatenate(bits_l), np.concatenate(orig_l), np.concatenate(kind_l))
    np.savez(cache, clip=out[0], bits=out[1], orig=out[2], kind=out[3], params=PARAMS)
    return out


def rule_metrics(S, ham, orig, kind_idx, th: dp.Thresholds) -> list[dict]:
    """Overall and per-kind precision / recall / F1 of a duplicate rule on the copy queries."""
    pred = dp.is_duplicate(S, ham, th)
    rows = []
    for name, sel in [("all", np.ones(len(orig), bool))] + [(k, kind_idx == i) for i, k in enumerate(dp.DUP_KINDS)]:
        p = pred[sel]
        tp = int(p[np.arange(sel.sum()), orig[sel]].sum())
        retrieved = int(p.sum())
        prec = tp / retrieved if retrieved else 1.0
        rec = tp / int(sel.sum())
        rows.append({"rule": th.rule, "kind": name, "dup_cos": th.dup_cos, "ham": th.ham,
                     "precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "retrieved": retrieved})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="flowers102")
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N originals")
    ap.add_argument("--min-precision", type=float, default=0.99)
    ap.add_argument("--recompute", action="store_true", help="ignore cached copy embeddings")
    ap.add_argument("--inspect-tau", type=float, default=0.95,
                    help="CLIP cosine used for the false-positive contact sheet (default 0.95)")
    args = ap.parse_args(argv)

    d = args.index_dir
    for f in ("clip_image.npy", "labels.npy"):
        if not (d / f).exists():
            raise SystemExit(f"Missing {d / f}. Run scripts/build_index.py first.")
    G, labels = np.load(d / "clip_image.npy"), np.load(d / "labels.npy")
    if (d / "query_ids.npy").exists():
        qids = np.load(d / "query_ids.npy")
    else:                                  # e.g. the fashion demo index: draw a seeded sample of originals
        qids = np.sort(np.random.default_rng(42).choice(len(G), size=min(300, len(G)), replace=False))
        print(f"No query_ids.npy in {d}: using a seeded sample of {len(qids)} originals.")
    orig_ids = qids[:args.limit] if args.limit else qids
    gallery = open_gallery(args.dataset)
    print(f"{len(orig_ids)} originals x {len(dp.DUP_KINDS)} copies against a gallery of {len(G)}")

    gbits = gallery_phash(gallery, d)
    cclip, cbits, orig, kind_idx = encode_copies(gallery, orig_ids, d, args.recompute)
    S = cclip @ G.T                                        # (M, N) CLIP cosine
    ham = dp.hamming_matrix(cbits, gbits)                  # (M, N) Hamming distance
    hs = (dp.PHASH_BITS - ham).astype(np.float32)          # pHash "score"

    # ---- precision-recall curves: CLIP cosine and pHash, overall and per copy kind
    curves = []
    for method, score, taus in (("clip", S, COS_TAUS), ("phash", hs, HAM_TAUS)):
        for name, sel in [("all", slice(None))] + [(k, kind_idx == i) for i, k in enumerate(dp.DUP_KINDS)]:
            sub, o = score[sel], orig[sel]
            pos = sub[np.arange(len(o)), o]
            c = dp.threshold_curve(pos, sub, taus)
            c.insert(0, "kind", name)
            c.insert(0, "method", method)
            c["ham_max"] = (dp.PHASH_BITS - c["tau"]) if method == "phash" else np.nan  # tau = 64 - Hamming
            curves.append(c)
    curve = pd.concat(curves, ignore_index=True)
    c_clip = curve[(curve.method == "clip") & (curve.kind == "all")].reset_index(drop=True)
    c_ph = curve[(curve.method == "phash") & (curve.kind == "all")].reset_index(drop=True)
    dup_row = dp.pick_threshold(c_clip, "precision", args.min_precision)
    ph_row = dp.pick_threshold(c_ph, "precision", args.min_precision)
    dup_cos, ham_thr = float(dup_row["tau"]), int(dp.PHASH_BITS - ph_row["tau"])   # lowest score = loosest

    # ---- "similar" threshold: same-class retrieval on the clean originals
    q = G[orig_ids]
    sims = q @ G.T
    self_mask = np.zeros_like(sims, dtype=bool)
    self_mask[np.arange(len(orig_ids)), orig_ids] = True
    same = (labels[orig_ids][:, None] == labels[None, :]) & ~self_mask
    sim_curve = dp.threshold_curve(sims[same], sims[~self_mask], COS_TAUS)
    sim_row = dp.pick_threshold(sim_curve, "f1")
    sim_cos = float(sim_row["tau"])
    if sim_cos >= dup_cos:
        print(f"WARNING: similar threshold {sim_cos} >= duplicate threshold {dup_cos}; lowering it.")
        sim_cos = round(dup_cos - 0.05, 4)

    # ---- the four rules at their operating points
    summary = []
    for rule in dp.RULES:
        summary += rule_metrics(S, ham, orig, kind_idx, dp.Thresholds(dup_cos, sim_cos, ham_thr, rule))
    summary = pd.DataFrame(summary)
    overall = summary[summary.kind == "all"].set_index("rule")
    best_rule = max(dp.RULES, key=lambda r: (round(overall.loc[r, "f1"], 4), r == "clip"))
    th = dp.Thresholds(dup_cos, sim_cos, ham_thr, best_rule)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve.round(5).to_csv(RESULTS_DIR / "duplicates_curve.csv", index=False)
    summary.round(4).to_csv(RESULTS_DIR / "duplicates_summary.csv", index=False)
    dp.save_thresholds(RESULTS_DIR / "duplicate_thresholds.json", th,
                       n_originals=int(len(orig_ids)), min_precision=args.min_precision,
                       dup_precision=round(float(dup_row["precision"]), 4),
                       dup_recall=round(float(dup_row["recall"]), 4),
                       sim_f1=round(float(sim_row["f1"]), 4),
                       sim_precision=round(float(sim_row["precision"]), 4),
                       sim_recall=round(float(sim_row["recall"]), 4))

    # ---- false positives at the duplicate threshold (inspect these by eye)
    r, c = np.nonzero(S >= dup_cos)
    fp = c != orig[r]
    r, c = r[fp], c[fp]
    order = np.argsort(-S[r, c])[:100]
    pd.DataFrame({"copy_kind": [dp.DUP_KINDS[i] for i in kind_idx[r[order]]], "orig_id": orig[r[order]],
                  "other_id": c[order], "cos": S[r[order], c[order]].round(4),
                  "same_class": labels[orig[r[order]]] == labels[c[order]]}
                 ).to_csv(RESULTS_DIR / "duplicates_false_positives.csv", index=False)

    plot(curve, sim_curve, dup_row, ph_row, sim_row, RESULTS_DIR / "duplicates_pr.png")

    print(f"\nDuplicate threshold (CLIP cosine, precision >= {args.min_precision}): {dup_cos}"
          f"   -> precision {dup_row['precision']:.4f}, recall {dup_row['recall']:.4f}")
    print(f"pHash threshold (max Hamming distance, same rule): {ham_thr}"
          f"   -> precision {ph_row['precision']:.4f}, recall {ph_row['recall']:.4f}")
    print(f"Similar threshold (best F1 for same-class retrieval): {sim_cos}"
          f"   -> P {sim_row['precision']:.3f}, R {sim_row['recall']:.3f}, F1 {sim_row['f1']:.3f}")
    show = summary.pivot(index="rule", columns="kind", values="recall").reindex(list(dp.RULES))
    show.insert(0, "precision", overall["precision"].reindex(list(dp.RULES)))
    print("\nPrecision (overall) and recall per copy kind at those thresholds:")
    print(show[["precision", "all", *dp.DUP_KINDS]].round(3).to_string())
    print(f"\nChosen rule (best overall F1, ties -> simplest): {best_rule}")
    print(f"False positives at the threshold: {int(fp.sum())} (see duplicates_false_positives.csv)")
    diagnostics(S, ham, orig, kind_idx, labels, gallery, args.inspect_tau)
    print(f"Saved to {RESULTS_DIR}")


def diagnostics(S, ham, orig, kind_idx, labels, gallery, inspect_tau: float) -> None:
    """Why does the CLIP threshold sit where it does? Three checks, all printed and saved.

    1. how close are the copies to their originals (CLIP cosine, pHash Hamming) per copy type
    2. false positives vs tau: how many, how many DISTINCT originals cause them, same class?
       (few originals + same class = the gallery itself holds near-identical photos)
    3. a contact sheet of the closest non-original pairs at --inspect-tau to look at by eye
    """
    import cv2

    rows_ = np.arange(len(orig))
    pos_cos, pos_ham = S[rows_, orig], ham[rows_, orig]
    q = []
    for i, k in enumerate(dp.DUP_KINDS):
        m = kind_idx == i
        q.append({"kind": k, "cos_median": np.median(pos_cos[m]), "cos_p10": np.percentile(pos_cos[m], 10),
                  "cos_min": pos_cos[m].min(), "ham_median": np.median(pos_ham[m]),
                  "ham_p90": np.percentile(pos_ham[m], 90)})
    qdf = pd.DataFrame(q)
    qdf.round(4).to_csv(RESULTS_DIR / "duplicates_positive_scores.csv", index=False)
    print("\n[1] Copy -> original similarity (how close a true duplicate really is):")
    print(qdf.round(3).to_string(index=False))

    out = []
    for tau in (0.90, 0.93, 0.95, 0.97, 0.98, 0.985, 0.99):
        r, c = np.nonzero(S >= tau)
        fp = c != orig[r]
        tp = int((c == orig[r]).sum())
        n_fp = int(fp.sum())
        pairs = np.unique(np.stack([orig[r[fp]], c[fp]], 1), axis=0) if n_fp else np.zeros((0, 2), int)
        out.append({"tau": tau, "recall": tp / len(orig), "precision": tp / max(tp + n_fp, 1),
                    "false_pos": n_fp, "distinct_pairs": len(pairs),
                    "distinct_originals": len(np.unique(pairs[:, 0])) if n_fp else 0,
                    "same_class_share": float((labels[pairs[:, 0]] == labels[pairs[:, 1]]).mean()) if n_fp else np.nan})
    odf = pd.DataFrame(out)
    odf.round(4).to_csv(RESULTS_DIR / "duplicates_fp_vs_tau.csv", index=False)
    print("\n[2] False positives vs CLIP threshold (recall counts all copies):")
    print(odf.round(3).to_string(index=False))

    r, c = np.nonzero(S >= inspect_tau)
    fp = c != orig[r]
    if fp.any():
        pairs = {}
        for rr, cc in zip(r[fp], c[fp]):
            key = (int(orig[rr]), int(cc))
            pairs[key] = max(pairs.get(key, -1.0), float(S[rr, cc]))
        top = sorted(pairs.items(), key=lambda kv: -kv[1])[:12]
        tiles = []
        for (a, b), cs in top:
            ims = [cv2.resize(gallery.get(g)[0], (200, 160), interpolation=cv2.INTER_AREA) for g in (a, b)]
            tile = np.hstack(ims)
            same = labels[a] == labels[b]
            cv2.putText(tile, f"#{a} | #{b}  cos {cs:.3f}  {'same class' if same else 'DIFF class'}",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(RESULTS_DIR / "duplicates_fp_sheet.png"), sheet)
        print(f"\n[3] Contact sheet of the {len(top)} closest non-original pairs at cosine >= {inspect_tau}: "
              "results/duplicates_fp_sheet.png (left = original, right = the other gallery image).")
    else:
        print(f"\n[3] No false positives at cosine >= {inspect_tau}: no contact sheet.")


def plot(curve, sim_curve, dup_row, ph_row, sim_row, path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for method, label in (("clip", "CLIP cosine"), ("phash", "pHash Hamming")):
        c = curve[(curve.method == method) & (curve.kind == "all")].sort_values("recall")
        ax[0].plot(c["recall"], c["precision"], label=label)
    ax[0].plot(dup_row["recall"], dup_row["precision"], "ko", label=f"CLIP tau={dup_row['tau']:.3f}")
    ax[0].plot(ph_row["recall"], ph_row["precision"], "ks", label=f"pHash <= {int(dp.PHASH_BITS - ph_row['tau'])}")
    ax[0].set(xlabel="recall", ylabel="precision", title="Duplicate detection: precision-recall (all copies)",
              ylim=(0, 1.02))
    ax[0].legend(fontsize=8)
    for kind in dp.DUP_KINDS:
        c = curve[(curve.method == "clip") & (curve.kind == kind)]
        ax[1].plot(c["tau"], c["recall"], label=kind)
    ax[1].axvline(dup_row["tau"], color="k", linestyle="--", linewidth=1)
    ax[1].set(xlabel="CLIP cosine threshold", ylabel="recall", title="Recall by copy type (CLIP)", xlim=(0.5, 1.0))
    ax[1].legend(fontsize=8)
    ax[2].plot(sim_curve["tau"], sim_curve["precision"], label="precision")
    ax[2].plot(sim_curve["tau"], sim_curve["recall"], label="recall")
    ax[2].plot(sim_curve["tau"], sim_curve["f1"], label="F1")
    ax[2].axvline(sim_row["tau"], color="k", linestyle="--", linewidth=1)
    ax[2].set(xlabel="CLIP cosine threshold", title="'Similar' tier: same-class retrieval", xlim=(0.3, 1.0))
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
