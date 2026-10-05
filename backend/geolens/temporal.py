"""Multi-temporal reasoning over an ordered series of observations of one site.

Finds the earliest observation at which a change is *supported*: it must exceed the change
threshold, come from usable imagery, be well registered, persist in later observations, and
reach the confidence threshold.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import change, imaging

DEFAULT_SERIES_PARAMS = {
    "conf_threshold": 0.55,
    "min_changed_frac": 0.002,
    "min_usable_frac": 0.70,
    "max_cloud": 0.25,
    "max_haze": 0.60,
}


def _quality_row(o: dict, q: dict, usable: bool, reason: str | None) -> dict:
    return {"obs_id": o["id"], "date": o["date"], "sensor": o["sensor"], "usable": usable,
            "exclusion": reason, "cloud_frac": round(q["cloud_frac"], 3),
            "shadow_frac": round(q["shadow_frac"], 3), "haze": round(q["haze"], 3),
            "usable_frac": round(q["usable_frac"], 3), "valid_frac": round(q["valid_frac"], 3),
            "synthetic": bool(o.get("synthetic")), "note": o.get("note")}


def _persistence(i: dict, j_aligned: np.ndarray, region: np.ndarray, later: list[dict],
                 valid_i: np.ndarray) -> float | None:
    """Fraction of changed pixels that, in later usable observations, still look like the
    'after' state rather than reverting to the 'before' state."""
    if not later or region.sum() == 0:
        return None
    base = imaging.to_float(i["rgb"])
    jn, _ = change.normalise(i["rgb"], j_aligned, valid_i)
    votes = []
    for k in later:
        reg = change.register(i["rgb"], k["rgb"], valid_i, k["valid"])
        kn, _ = change.normalise(i["rgb"], reg["aligned"], valid_i & reg["valid"])
        d_before = np.abs(kn - base).mean(-1)
        d_after = np.abs(kn - jn).mean(-1)
        ok = region & reg["valid"] & k["usable_mask"]
        if ok.sum() < 5:
            continue
        votes.append(float((d_after[ok] < d_before[ok]).mean()))
    return float(np.mean(votes)) if votes else None


def analyze_series(observations: list[dict], gsd: float, params: dict | None = None,
                   pair_params: dict | None = None) -> dict[str, Any]:
    sp = {**DEFAULT_SERIES_PARAMS, **(params or {})}
    obs = sorted(observations, key=lambda o: o["date"])
    h, w = obs[0]["rgb"].shape[:2]
    timeline: list[dict] = []
    for o in obs:
        o.setdefault("valid", np.ones((h, w), bool))
        q = imaging.assess_quality(o["rgb"], o["valid"])
        reason = None
        if q["cloud_frac"] > sp["max_cloud"]:
            reason = f"cloud cover {q['cloud_frac']:.0%} exceeds {sp['max_cloud']:.0%}"
        elif q["usable_frac"] < sp["min_usable_frac"]:
            reason = f"only {q['usable_frac']:.0%} usable pixels (cloud {q['cloud_frac']:.0%})"
        elif q["haze"] > sp["max_haze"]:
            reason = f"heavy haze ({q['haze']:.2f})"
        o["usable"] = reason is None
        o["usable_mask"] = q["usable_mask"]
        o["quality"] = q
        timeline.append(_quality_row(o, q, o["usable"], reason))

    usable = [o for o in obs if o["usable"]]
    transitions: list[dict] = []
    rejected: list[dict] = []

    def run_pair(a: dict, b: dict, later: list[dict]):
        res = change.analyze_pair(a["rgb"], b["rgb"], gsd, a["valid"], b["valid"], pair_params)
        n_total = res.mask.size
        rec: dict[str, Any] = {
            "before_id": a["id"], "after_id": b["id"], "before_date": a["date"], "after_date": b["date"],
            "changed_frac": res.stats["changed_frac"], "regions": len(res.regions),
            "registration": {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in res.registration.items()},
            "normalisation": res.normalisation, "has_change": False, "supported": False,
            "persistence": None, "confidence": 0.0, "factors": None, "reasons": [], "type": None,
        }
        if res.mask.sum() / n_total >= sp["min_changed_frac"] and res.regions:
            rec["has_change"] = True
            idx = list(range(len(res.regions)))
            reg_union = np.zeros_like(res.mask)
            for r in res.regions:
                pass
            from scipy import ndimage as ndi  # local import keeps module import light
            lab, _ = ndi.label(res.mask)
            reg_union = res.mask
            pers = _persistence(a, res.aligned_after, reg_union, later, a["valid"])
            by_area: dict[str, float] = {}
            for r in res.regions:
                by_area[r["type"]] = by_area.get(r["type"], 0) + r["area_px"]
            dom = max(by_area, key=by_area.get)
            veg_only = all(r["veg_only"] for r in res.regions)
            factors, reasons = change.pair_factors(res, idx, a["date"], b["date"], a["sensor"],
                                                   b["sensor"], pers, list(by_area), veg_only)
            conf = change.combine_confidence(factors)
            rec.update({"persistence": pers, "factors": factors, "reasons": reasons,
                        "confidence": conf, "type": dom,
                        "type_label": change.CHANGE_TYPES[dom],
                        "area_m2": float(sum(r["area_px"] for r in res.regions) * gsd * gsd),
                        "area_frac": res.stats["changed_frac"],
                        "score": float(np.average([r["mean_score"] for r in res.regions],
                                                  weights=[r["area_px"] for r in res.regions])),
                        "type_areas": {k: float(v * gsd * gsd) for k, v in by_area.items()}})
        return res, rec

    for idx in range(len(usable) - 1):
        a, b = usable[idx], usable[idx + 1]
        later = usable[idx + 2:]
        res, rec = run_pair(a, b, later)
        rec["supported"] = bool(rec["has_change"] and rec["confidence"] >= sp["conf_threshold"])
        if rec["has_change"] and not rec["supported"]:
            rec["reasons"].append(f"confidence {rec['confidence']:.2f} below threshold "
                                  f"{sp['conf_threshold']:.2f}")
        rec["_res"] = res
        transitions.append(rec)

    # audit: candidate changes that touch an excluded observation (for false-alarm accounting)
    for pos, o in enumerate(obs):
        if o["usable"]:
            continue
        prev = next((x for x in reversed(obs[:pos]) if x["usable"]), None)
        if prev is None:
            continue
        res, rec = run_pair(prev, o, [])
        if rec["has_change"]:
            rec["supported"] = False
            rec["reasons"].insert(0, f"observation {o['date']} excluded: "
                                  f"{next(t['exclusion'] for t in timeline if t['obs_id'] == o['id'])}")
            rec["confidence"] = min(rec["confidence"], 0.2)
            rec["rejected_candidate"] = True
            rejected.append(rec)

    first = next((t for t in transitions if t["supported"]), None)
    return {"timeline": timeline, "transitions": transitions, "rejected": rejected,
            "first_supported": first, "observations": obs, "params": sp}
