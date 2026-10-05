"""Evaluation harness. Writes data/evaluation/metrics.json, which the UI shows verbatim.

Three things are measured, each with its caveat recorded next to the number:
  1. Change detection, pixel level, on the real LEVIR-CD sample pairs (labels are human-drawn).
  2. Scenario suite: does the temporal logic date the change correctly and suppress false alarms?
  3. Retrieval: robustness to image perturbations, and label coherence of neighbours.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import analysis, change, config, db, imaging
from .embedding import DescriptorEmbedder
from .index import VectorIndex


def levir_pixel_metrics() -> list[dict]:
    rows = []
    for name in sorted(p.name for p in (config.RAW / "levir" / "A").glob("*.png")):
        a = np.asarray(Image.open(config.RAW / "levir" / "A" / name).convert("RGB"))
        b = np.asarray(Image.open(config.RAW / "levir" / "B" / name).convert("RGB"))
        t = np.asarray(Image.open(config.RAW / "levir" / "label" / name).convert("L")) > 127
        m = change.analyze_pair(a, b, 0.5).mask
        tp, fp, fn = int((m & t).sum()), int((m & ~t).sum()), int((~m & t).sum())
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        tn = int(m.size - tp - fp - fn)
        rows.append({"pair": name, "fpr": round(fp / max(fp + tn, 1), 3), "precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(2 * p * r / max(p + r, 1e-9), 3), "iou": round(tp / max(tp + fp + fn, 1), 3),
                     "truth_frac": round(float(t.mean()), 3), "pred_frac": round(float(m.mean()), 3)})
    return rows


def scenario_suite() -> list[dict]:
    rows = []
    with db.session() as s:
        sites = [(x.site_id, x.name, x.scenario, x.expected or {}) for x in s.query(db.Site).order_by(db.Site.site_id)]
    for sid, name, scen, exp in sites:
        with db.session() as s:
            ev = s.query(db.ChangeEvent).filter(db.ChangeEvent.site_id == sid).all()
        sup = sorted([e for e in ev if e.status == "supported"], key=lambda e: e.first_supported_date)
        got = sup[0].first_supported_date if sup else None
        want = exp.get("first_change")
        ok_date = got == want
        ok_type = True if not want else (sup and sup[0].change_type == exp.get("type"))
        rejected_ok = True
        for d in exp.get("rejected", []):
            if any(e.status == "supported" and e.first_supported_date == d for e in ev):
                rejected_ok = False
        rows.append({"site": sid, "name": name, "scenario": scen, "expected_date": want, "got_date": got,
                     "expected_type": exp.get("type"), "got_type": sup[0].change_type if sup else None,
                     "false_alarm_dates": [e.first_supported_date for e in sup if e.first_supported_date != want],
                     "pass": bool(ok_date and ok_type and rejected_ok)})
    return rows


def _augment(rgb: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    f = rgb.astype(np.float32)
    if kind == "flip":
        return rgb[:, ::-1].copy()
    if kind == "rot90":
        return np.rot90(rgb).copy()
    if kind == "brightness":
        return np.clip(f * rng.uniform(0.88, 1.12), 0, 255).astype(np.uint8)
    if kind == "noise":
        return np.clip(f + rng.normal(0, 4, f.shape), 0, 255).astype(np.uint8)
    if kind == "shift":
        return np.roll(rgb, (int(rng.integers(-6, 7)), int(rng.integers(-6, 7))), (0, 1))
    raise ValueError(kind)


def _label(attrs: dict) -> str:
    cand = {"water": attrs["water"], "veg": attrs["veg"], "bare": attrs["bare"], "urban": attrs["urban"]}
    return max(cand, key=cand.get)


def retrieval_metrics(index: VectorIndex, emb: DescriptorEmbedder, n: int = 120, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    vecs, ids, payloads = index.all_vectors()
    with db.session() as s:
        tiles = {t.point_id: (t.scene_id, t.window, t.attrs, t.quality) for t in s.query(db.Tile).all()}
    good = [i for i in ids if tiles[i][3] > 0.8]
    pick = list(rng.choice(good, size=min(n, len(good)), replace=False))
    from . import ingest
    cache: dict[str, tuple] = {}
    out = {}
    for kind in ("flip", "rot90", "brightness", "noise", "shift"):
        ranks = []
        for pid in pick:
            sid, win, attrs, _ = tiles[pid]
            if sid not in cache:
                cache[sid] = ingest.read_scene_rgb(sid)
            rgb, valid = cache[sid]
            c, r, w, h = win
            tile = rgb[r:r + h, c:c + w]
            q = emb.embed_tile(_augment(tile, kind, rng), valid[r:r + h, c:c + w])
            res = index.search(q, None, limit=50)
            ranks.append(next((k + 1 for k, x in enumerate(res) if x["id"] == pid), 999))
        ranks = np.asarray(ranks)
        out[kind] = {"recall@1": round(float((ranks <= 1).mean()), 3), "recall@5": round(float((ranks <= 5).mean()), 3),
                     "recall@10": round(float((ranks <= 10).mean()), 3)}
    # label coherence of neighbours (labels are rule-derived, so this is optimistic)
    labels = {i: _label(tiles[i][2]) for i in ids}
    arr = np.asarray(vecs)
    p10, base = [], []
    classes = np.asarray([labels[i] for i in ids])
    prior = {c: float((classes == c).mean()) for c in set(classes)}
    for pid in pick:
        k = ids.index(pid)
        sims = arr @ arr[k]
        sims[k] = -9
        nn = np.argsort(-sims)[:10]
        p10.append(float((classes[nn] == classes[k]).mean()))
        base.append(prior[classes[k]])
    # latency
    times = []
    for pid in pick[:40]:
        t = time.perf_counter()
        index.search(arr[ids.index(pid)], None, limit=20)
        times.append((time.perf_counter() - t) * 1000)
    return {"n_queries": len(pick), "index_size": len(ids), "perturbation_recall": out,
            "label_coherence_p@10": round(float(np.mean(p10)), 3),
            "label_coherence_random_baseline": round(float(np.mean(base)), 3),
            "search_ms_median": round(float(np.median(times)), 1),
            "search_ms_p95": round(float(np.percentile(times, 95)), 1),
            "caveats": ["Perturbation recall asks whether a distorted copy of a tile finds the original; "
                        "it measures robustness, not semantic quality.",
                        "Label coherence uses cover classes from the same colour rules that feed the "
                        "descriptor, so it overstates semantic accuracy. No human-labelled retrieval "
                        "benchmark is available offline."]}


def _dir_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 1)


def system_metrics(index: VectorIndex) -> dict:
    import os
    import platform
    from sqlalchemy import func
    with db.session() as s:
        runs = s.query(db.ProcessingRun).filter(db.ProcessingRun.kind == "ingest").all()
        area = s.query(func.sum(db.Scene.width * db.Scene.height * db.Scene.resolution_m ** 2)).scalar() or 0
        scenes = s.query(db.Scene).count()
        tiles = s.query(db.Tile).count()
    secs = [r.stats["seconds"] for r in runs if r.stats]
    tl = [r.stats["tiles"] for r in runs if r.stats]
    return {"scenes": scenes, "tiles": tiles, "indexed_area_km2": round(area / 1e6, 1),
            "ingest_seconds_total": round(sum(secs), 1),
            "ingest_seconds_per_scene": round(float(np.mean(secs)), 1) if secs else None,
            "ingest_ms_per_tile": round(1000 * sum(secs) / max(sum(tl), 1), 1),
            "storage_mb": {"processed_imagery": _dir_mb(config.PROCESSED), "vector_index": _dir_mb(Path(config.QDRANT_PATH)),
                           "raw_inputs": _dir_mb(config.RAW)},
            "hardware": {"cpu_cores": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version(),
                         "gpu": "none used"}}


def run(index: VectorIndex, emb: DescriptorEmbedder) -> dict:
    lev = levir_pixel_metrics()
    suite = scenario_suite()
    ret = retrieval_metrics(index, emb)
    mean = {k: round(float(np.mean([r[k] for r in lev])), 3) for k in ("precision", "recall", "f1", "iou", "fpr")}
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": config.PIPELINE_VERSION, "embedder": emb.name,
        "levir_pixel": lev, "levir_pixel_mean": mean,
        "levir_caveat": "Thresholds were tuned on these same 3 sample pairs, so the numbers are optimistic. "
                        "The full LEVIR-CD test set (not downloadable here) is the right place to measure.",
        "scenarios": suite,
        "scenario_summary": {"passed": sum(r["pass"] for r in suite), "total": len(suite)},
        "scenario_caveat": "Regression suite: the pipeline was tuned until these cases passed, and several "
                           "epochs are synthetic. It shows the logic works as designed, not field accuracy.",
        "retrieval": ret,
        "system": system_metrics(index),
    }
    (config.EVAL_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    return out
