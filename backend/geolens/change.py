"""Pairwise change analysis: registration, radiometric normalisation, change scoring,
segmentation and rule-based classification.

Raw image difference is only one input. Every stage records what it did so the confidence model
(and the analyst) can see why a change was or was not believed.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.registration import phase_cross_correlation

from . import imaging

CHANGE_TYPES = {
    "construction": "Construction",
    "clearance": "Clearance / removal",
    "water_expansion": "Water-extent expansion",
    "water_reduction": "Water-extent reduction",
    "road": "Road development",
    "other": "Other / unknown change",
}

DEFAULT_PARAMS = {
    "z_threshold": 3.0,
    "abs_min": 0.10,          # minimum smoothed colour difference (0..1) to count as change
    "min_area_m2": None,      # defaults derived from GSD
    "max_shift_px": 8.0,
    "smooth_sigma": None,
}


def _min_area_px(gsd: float, min_area_m2: float | None) -> int:
    if min_area_m2:
        return max(2, int(min_area_m2 / (gsd * gsd)))
    if gsd >= 10:
        return 3
    if gsd >= 1:
        return 12
    return 60


# ----------------------------------------------------------------------------- registration
def _highpass(g: np.ndarray, valid: np.ndarray) -> np.ndarray:
    g = np.where(valid, g, g[valid].mean() if valid.any() else 0)
    hp = g - ndi.gaussian_filter(g, 6)
    hp[~valid] = 0
    return hp


def _ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    if m.sum() < 50:
        return 0.0
    x, y = a[m], b[m]
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-9 else 0.0


def register(ref: np.ndarray, mov: np.ndarray, valid_ref: np.ndarray, valid_mov: np.ndarray,
             max_shift: float = 8.0) -> dict[str, Any]:
    """Estimate and remove a translation between two acquisitions of the same footprint."""
    v = valid_ref & valid_mov
    g1 = _highpass(imaging.gray(ref), v)
    g2 = _highpass(imaging.gray(mov), v)
    ncc_before = _ncc(g1, g2, v)
    try:
        shift, _, _ = phase_cross_correlation(g1, g2, upsample_factor=10, normalization=None)
    except Exception:
        shift = np.zeros(2)
    shift = np.asarray(shift, float)
    failed = bool(np.hypot(*shift) > max_shift)
    if failed:
        shift = np.zeros(2)
    def _apply(sh):
        al = np.stack([ndi.shift(mov[..., c].astype(np.float32), sh, order=1, mode="nearest")
                       for c in range(3)], -1)
        al = np.clip(al, 0, 255).astype(np.uint8)
        va = ndi.shift(valid_mov.astype(np.uint8), sh, order=0, mode="constant", cval=0).astype(bool)
        g2 = _highpass(imaging.gray(al), valid_ref & va)
        return al, va, g2, _ncc(g1, g2, valid_ref & va)

    aligned, valid_a, g2a, ncc_after = _apply(shift)
    if np.hypot(*shift) > 0 and ncc_after < ncc_before + 0.02:
        # the estimated shift did not improve agreement: structure changed, not geometry
        shift = np.zeros(2)
        aligned, valid_a, g2a, ncc_after = _apply(shift)
    margin = int(np.ceil(np.abs(shift).max())) + 2 if np.abs(shift).max() > 0 else 0
    if margin:
        # pixels near the edge were extrapolated or interpolated by the shift; never trust them
        valid_a = valid_a.copy()
        valid_a[:margin, :] = False
        valid_a[-margin:, :] = False
        valid_a[:, :margin] = False
        valid_a[:, -margin:] = False
    vv = valid_ref & valid_a
    try:
        resid, _, _ = phase_cross_correlation(g1, g2a, upsample_factor=10, normalization=None)
        resid_px = float(np.hypot(*resid))
    except Exception:
        resid_px = 0.0
    # With strongly different content the correlation is low whatever the geometry. In that case the
    # alignment cannot be verified; that is a mild penalty, not evidence of misregistration.
    verified = ncc_after >= 0.30
    if not verified:
        resid_px = 0.0
        quality = 0.7 * (0.5 if failed else 1.0)
    else:
        quality = float(np.clip(ncc_after, 0, 1) ** 0.5 * np.exp(-resid_px / 1.0) * (0.5 if failed else 1.0))
    return {
        "aligned": aligned, "valid": valid_a,
        "info": {"shift_dy": float(shift[0]), "shift_dx": float(shift[1]), "residual_px": resid_px,
                 "ncc_before": ncc_before, "ncc_after": ncc_after, "failed": failed,
                 "verified": bool(verified), "quality": quality},
    }


# ----------------------------------------------------------------------------- normalisation
def normalise(before: np.ndarray, after: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-band robust gain/offset so `after` matches `before` on pixels that did not change
    (iteratively re-estimated, in the spirit of IR-MAD without the eigen-decomposition)."""
    b = imaging.to_float(before)
    a = imaging.to_float(after)
    mask = valid.copy()
    gains, offsets = np.ones(3), np.zeros(3)
    out = a.copy()
    for _ in range(3):
        if mask.sum() < 100:
            break
        for c in range(3):
            x, y = a[..., c][mask], b[..., c][mask]
            qx = np.subtract(*np.percentile(x, [75, 25]))
            qy = np.subtract(*np.percentile(y, [75, 25]))
            g = float(np.clip(qy / max(qx, 1e-3), 0.6, 1.6))
            o = float(np.median(y) - g * np.median(x))
            gains[c], offsets[c] = g, o
            out[..., c] = np.clip(a[..., c] * g + o, 0, 1)
        d = np.abs(out - b).mean(-1)
        mask = valid & (d < np.percentile(d[valid], 80))
    return out, {"gain": gains.round(3).tolist(), "offset": offsets.round(3).tolist()}


# ----------------------------------------------------------------------------- change score
def _robust_sigma(x: np.ndarray, floor: float) -> tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, max(mad, floor)


def change_score(before: np.ndarray, after_norm: np.ndarray, valid: np.ndarray, gsd: float,
                 smooth: float | None = None) -> dict[str, np.ndarray]:
    b = imaging.to_float(before) if before.dtype == np.uint8 else before
    a = after_norm
    d = a - b
    mag = np.sqrt((d ** 2).sum(-1) / 3.0)
    med, sig = _robust_sigma(mag[valid], 0.02)
    z_spec = (mag - med) / sig
    tb = imaging.local_std(0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2], 5)
    ta = imaging.local_std(0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2], 5)
    ds = np.abs(ta - tb)
    med2, sig2 = _robust_sigma(ds[valid], 0.01)
    z_str = (ds - med2) / sig2
    s = smooth if smooth is not None else (1.0 if gsd >= 10 else 1.5 if gsd >= 1 else 2.0)
    score = ndi.gaussian_filter(np.clip(0.65 * z_spec + 0.35 * z_str, -5, 20), s)
    mag_s = ndi.gaussian_filter(mag, s)
    return {"score": score.astype(np.float32), "mag": mag_s.astype(np.float32), "diff": d}


# ----------------------------------------------------------------------------- classification
def _fractions(rgb: np.ndarray, region: np.ndarray, gsd: float = 30.0) -> dict[str, float]:
    m = imaging.cover_masks(rgb)
    if gsd < 2:
        m["built"] = imaging.hires_built(rgb)
    gy = imaging.gray(rgb)
    busy = imaging.local_std(gy, 5) > 0.06
    n = max(int(region.sum()), 1)
    out = {k: float((v & region).sum() / n) for k, v in m.items()}
    out["urban"] = float((((m["built"] | m["bare"]) & busy) & region).sum() / n)
    out["brightness"] = float(gy[region].mean()) if region.any() else 0.0
    return out


def classify_region(before: np.ndarray, after: np.ndarray, region: np.ndarray, props: Any,
                    gsd: float) -> dict[str, Any]:
    fb, fa = _fractions(before, region, gsd), _fractions(after, region, gsd)
    major = float(getattr(props, "axis_major_length", None) or getattr(props, "major_axis_length", 0.0))
    minor = float(max(getattr(props, "axis_minor_length", None) or getattr(props, "minor_axis_length", 1.0), 1.0))
    elong = major / minor
    built_a = max(fa["built"], fa["urban"])
    built_b = max(fb["built"], fb["urban"])
    veg_loss = fb["veg"] - fa["veg"]
    reasons: list[str] = []
    ctype, cscore = "other", 0.35

    if fa["water"] > 0.55 and fb["water"] < 0.25:
        ctype, cscore = "water_expansion", min(1.0, 0.5 + fa["water"] - fb["water"])
        reasons.append(f"water cover {fb['water']:.0%} -> {fa['water']:.0%}")
    elif fb["water"] > 0.55 and fa["water"] < 0.25:
        ctype, cscore = "water_reduction", min(1.0, 0.5 + fb["water"] - fa["water"])
        reasons.append(f"water cover {fb['water']:.0%} -> {fa['water']:.0%}")
    elif elong >= 4.0 and minor <= max(3.0, 12.0 / gsd) and fb["water"] < 0.3 and fa["water"] < 0.3 \
            and (fa["built"] + fa["bare"]) > 0.4:
        ctype, cscore = "road", min(1.0, 0.45 + 0.05 * elong)
        reasons.append(f"linear feature, elongation {elong:.1f}")
    elif built_a > 0.30 and built_a - built_b > 0.20:
        ctype, cscore = "construction", min(1.0, 0.4 + built_a - built_b)
        reasons.append(f"built-up cover {built_b:.0%} -> {built_a:.0%}")
    elif (veg_loss > 0.30 and fa["veg"] < 0.35) or (built_b - built_a > 0.25 and built_a < 0.30):
        ctype, cscore = "clearance", min(1.0, 0.4 + max(veg_loss, built_b - built_a))
        reasons.append(f"vegetation {fb['veg']:.0%} -> {fa['veg']:.0%}"
                       if veg_loss > 0.3 else f"built-up {built_b:.0%} -> {built_a:.0%}")
    else:
        reasons.append("no rule matched the before/after cover change")
    veg_only = ctype in ("other", "clearance") \
        and (fa["built"] - fb["built"]) < 0.12 and (fa["urban"] - fb["urban"]) < 0.12 \
        and abs(fa["water"] - fb["water"]) < 0.15
    return {"type": ctype, "class_score": float(cscore), "before": fb, "after": fa,
            "elongation": float(elong), "rule": "; ".join(reasons), "veg_only": bool(veg_only)}


# ----------------------------------------------------------------------------- pair analysis
@dataclass
class PairResult:
    mask: np.ndarray
    score: np.ndarray
    regions: list[dict]
    before_q: dict
    after_q: dict
    registration: dict
    normalisation: dict
    aligned_after: np.ndarray
    valid: np.ndarray
    stats: dict = field(default_factory=dict)


def analyze_pair(before: np.ndarray, after: np.ndarray, gsd: float,
                 valid_before: np.ndarray | None = None, valid_after: np.ndarray | None = None,
                 params: dict | None = None, do_register: bool = True) -> PairResult:
    p = {**DEFAULT_PARAMS, **(params or {})}
    h, w = before.shape[:2]
    vb = np.ones((h, w), bool) if valid_before is None else valid_before
    va = np.ones((h, w), bool) if valid_after is None else valid_after
    qb = imaging.assess_quality(before, vb)
    qa = imaging.assess_quality(after, va)

    if do_register:
        reg = register(before, after, vb, va, p["max_shift_px"])
        aligned, va2, reg_info = reg["aligned"], reg["valid"], reg["info"]
    else:
        aligned, va2 = after, va
        reg_info = {"shift_dy": 0.0, "shift_dx": 0.0, "residual_px": 0.0, "ncc_before": 1.0,
                    "ncc_after": 1.0, "failed": False, "verified": True, "quality": 1.0}

    valid = vb & va2
    usable = valid & qb["usable_mask"] & ~ndi.shift(qa["cloud_mask"].astype(np.uint8),
                                                    [reg_info["shift_dy"], reg_info["shift_dx"]],
                                                    order=0).astype(bool)
    usable &= ~ndi.shift(((qa["usable_mask"] == 0) & va).astype(np.uint8),
                         [reg_info["shift_dy"], reg_info["shift_dx"]], order=0).astype(bool)
    norm_after, norm_info = normalise(before, aligned, usable if usable.sum() > 200 else valid)
    sc = change_score(before, norm_after, usable if usable.sum() > 200 else valid, gsd, p["smooth_sigma"])

    if gsd < 2:
        # sub-2 m imagery: pixel differencing cannot separate new buildings from tone changes in
        # lawns and trees, so compare man-made-surface masks instead (post-classification change)
        built_b, built_a = imaging.hires_built(before), imaging.hires_built(aligned)
        gain = built_a & ~ndi.binary_dilation(built_b, iterations=2)
        loss = built_b & ~ndi.binary_dilation(built_a, iterations=2)
        mask = (gain | loss) & usable
        mask = ndi.binary_opening(mask, iterations=2)
        mask = ndi.binary_closing(mask, iterations=2)
        # evidence map = local agreement of the man-made-surface change (0..1)
        sc["score"] = ndi.gaussian_filter(mask.astype(np.float32), 2.0)
        mode = "postclass"
    else:
        mode = "spectral"
        mask = (sc["score"] > p["z_threshold"]) & (sc["mag"] > p["abs_min"]) & usable
        mask = ndi.binary_closing(mask, iterations=1)
    mask = morphology.remove_small_objects(mask, max_size=_min_area_px(gsd, p["min_area_m2"]) - 1)
    lab, n = ndi.label(mask)
    regions: list[dict] = []
    for rp in measure.regionprops(lab, intensity_image=sc["score"]):
        reg_mask = lab == rp.label
        cls = classify_region(before, aligned, reg_mask, rp, gsd)
        r0, c0, r1, c1 = rp.bbox
        regions.append({
            "label": int(rp.label), "area_px": int(rp.area), "area_m2": float(rp.area * gsd * gsd),
            "bbox": [int(c0), int(r0), int(c1), int(r1)],
            "centroid": [float(rp.centroid[1]), float(rp.centroid[0])],
            "mean_score": float(getattr(rp, "intensity_mean", None) if getattr(rp, "intensity_mean", None) is not None else rp.mean_intensity), "solidity": float(rp.solidity), **cls,
        })
    vm_b = imaging.cover_masks(before)["veg"] & usable
    vm_a = imaging.cover_masks(aligned)["veg"] & usable
    veg = vm_b & vm_a
    other = usable & ~ndi.binary_dilation(vm_b | vm_a, iterations=2) & ~mask
    fb_, fa_ = imaging.to_float(before), imaging.to_float(aligned)
    if veg.sum() > 30 and other.sum() > 30:
        v_shift = float(np.linalg.norm(fa_[veg].mean(0) - fb_[veg].mean(0)))
        o_shift = float(np.linalg.norm(fa_[other].mean(0) - fb_[other].mean(0)))
    else:
        v_shift = o_shift = 0.0
    stats = {"veg_shift": v_shift, "nonveg_shift": o_shift, "mode": mode, "changed_px": int(mask.sum()), "changed_frac": float(mask.mean()),
             "usable_frac": float(usable.mean()), "regions": len(regions),
             "z_threshold": p["z_threshold"], "abs_min": p["abs_min"]}
    return PairResult(mask, sc["score"], regions, _strip(qb), _strip(qa), reg_info, norm_info,
                      aligned, valid, stats)


def _strip(q: dict) -> dict:
    return {k: v for k, v in q.items() if not k.endswith("_mask")}


# ----------------------------------------------------------------------------- confidence
def season_delta(d1: str, d2: str) -> float:
    """0 = same time of year, 1 = opposite season (circular day-of-year distance)."""
    a, b = dt.date.fromisoformat(d1), dt.date.fromisoformat(d2)
    delta = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return float(min(delta, 365 - delta) / 182.5)


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def pair_factors(res: PairResult, region_idx: list[int], d_before: str, d_after: str,
                 sensor_before: str, sensor_after: str, persistence: float | None,
                 types: list[str], veg_only: bool = False) -> tuple[dict, list[str]]:
    regs = [res.regions[i] for i in region_idx]
    area = sum(r["area_px"] for r in regs)
    zmean = float(np.average([r["mean_score"] for r in regs], weights=[r["area_px"] for r in regs]))
    reasons: list[str] = []
    if res.stats.get("mode") == "postclass":
        evidence = sigmoid((zmean - 0.55) / 0.08) * (0.75 + 0.25 * min(1.0, area / 400.0))
    else:
        evidence = sigmoid((zmean - 3.4) / 1.0) * (0.75 + 0.25 * min(1.0, area / 40.0))
    q = float(np.sqrt(max(min(res.before_q["usable_frac"], res.after_q["usable_frac"]), 0.0)))
    if res.before_q["cloud_frac"] > 0.1 or res.after_q["cloud_frac"] > 0.1:
        reasons.append(f"cloud cover {max(res.before_q['cloud_frac'], res.after_q['cloud_frac']):.0%} "
                       "masked out of the comparison")
    haze = max(res.before_q["haze"], res.after_q["haze"])
    if haze > 0.5:
        q *= 1 - 0.5 * (haze - 0.5)
        reasons.append(f"haze index {haze:.2f}")
    reg = res.registration["quality"]
    if res.registration["residual_px"] > 0.8 or res.registration["failed"]:
        reasons.append(f"registration residual {res.registration['residual_px']:.1f} px"
                       + (" (alignment failed, shift not applied)" if res.registration["failed"] else ""))
    if abs(res.registration["shift_dx"]) > 0.6 or abs(res.registration["shift_dy"]) > 0.6:
        reasons.append(f"scene shifted by ({res.registration['shift_dx']:+.1f}, "
                       f"{res.registration['shift_dy']:+.1f}) px and re-aligned")
    if persistence is None:
        temporal = 0.8
        reasons.append("latest transition: no later observation to confirm it persists")
    else:
        temporal = float(np.clip(0.15 + 0.85 * persistence, 0, 1))
        if persistence < 0.6:
            reasons.append(f"change does not persist in later observations ({persistence:.0%} agreement)")
    sensor = 1.0 if sensor_before == sensor_after else 0.8
    if sensor != 1.0:
        reasons.append(f"different sensors ({sensor_before} vs {sensor_after})")
    sd = season_delta(d_before, d_after)
    season = 1.0 - 0.1 * sd
    phen = max(0.0, res.stats.get("veg_shift", 0.0) - res.stats.get("nonveg_shift", 0.0))
    area_veg = sum(r["area_px"] for r in regs if r["veg_only"]) / max(area, 1)
    if sd > 0.25 and (phen > 0.03 or area_veg >= 0.7):
        strength = max(min(1.0, phen / 0.06), area_veg if area_veg >= 0.7 else 0.0)
        season = 1.0 - 0.8 * sd * strength
        reasons.append(f"observations {sd * 6:.0f} months apart in the seasonal cycle; vegetation shifts "
                       f"{phen:.2f} more than other surfaces, so the difference is treated as phenology")
    cls = float(np.average([r["class_score"] for r in regs], weights=[r["area_px"] for r in regs]))
    f = {"evidence": evidence, "quality": q, "registration": reg, "temporal": temporal,
         "sensor": sensor, "season": season, "classification": 0.5 + 0.5 * cls}
    return {k: round(float(v), 3) for k, v in f.items()}, reasons


def combine_confidence(f: dict) -> float:
    w = {"evidence": 3.0, "temporal": 2.0, "quality": 1.5, "registration": 1.5, "season": 1.5,
         "sensor": 0.5, "classification": 1.0}
    logsum = sum(w[k] * np.log(max(f[k], 1e-3)) for k in w) / sum(w.values())
    conf = float(np.exp(logsum))
    if min(f["quality"], f["registration"], f["temporal"], f["season"]) < 0.3:
        conf = min(conf, 0.35)
    return round(conf, 3)
