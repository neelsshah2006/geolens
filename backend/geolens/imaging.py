"""Pixel-level image analysis shared by retrieval, quality control and change detection.

All functions work on uint8 RGB arrays (H, W, 3). Landsat digital numbers are mapped to uint8
with one fixed stretch (config.LANDSAT8_STRETCH) so every scene lives in the same value space.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import local_binary_pattern

from . import config


# --------------------------------------------------------------------------- basics
def stretch_dn(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = []
    for arr, key in ((r, "R"), (g, "G"), (b, "B")):
        lo, hi = config.LANDSAT8_STRETCH[key]
        out.append(np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1))
    return (np.stack(out, -1) * 255 + 0.5).astype(np.uint8)


def to_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def gray(rgb: np.ndarray) -> np.ndarray:
    f = to_float(rgb)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def hsv_arrays(rgb: np.ndarray):
    f = to_float(rgb)
    mx = f.max(-1)
    mn = f.min(-1)
    d = mx - mn
    sat = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return mx, sat, d


def local_std(img: np.ndarray, size: int = 5) -> np.ndarray:
    m = ndi.uniform_filter(img, size)
    m2 = ndi.uniform_filter(img * img, size)
    return np.sqrt(np.clip(m2 - m * m, 0, None))


# --------------------------------------------------------------------------- land cover
def cover_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Rule-based per-pixel classes for true-colour imagery (no NIR/SWIR available).

    The rules use only red/green/blue relationships, which is all that is guaranteed to exist
    in ordinary RGB composites. They are deliberately coarse; they feed interpretable tile
    attributes and the change classifier, not a land-cover map.
    """
    f = to_float(rgb)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    v, sat, _ = hsv_arrays(rgb)
    water = ((b - r) > 0.10) & (b >= g - 0.03) & (sat > 0.25) & (v < 0.85)
    veg = (g > r + 0.05) & (g > b + 0.05) & (sat > 0.20) & ~water
    dark = (v < 0.17) & ~water
    built = (sat < 0.20) & (v > 0.45) & ~water
    bare = (r >= g - 0.02) & (r >= b - 0.02) & (sat >= 0.10) & (v > 0.30) & ~water & ~built
    return {"water": water, "veg": veg, "bare": bare, "built": built, "dark": dark}


def hires_built(rgb: np.ndarray) -> np.ndarray:
    """Man-made surfaces in sub-2 m imagery: desaturated, not shadow-dark, locally smooth, not
    greenish. Roofs, roads and driveways all qualify. Parameters were picked on the three LEVIR-CD
    sample pairs (see docs/EVALUATION.md) so treat the resulting scores as optimistic."""
    f = to_float(rgb)
    v, sat, _ = hsv_arrays(rgb)
    smooth = local_std(gray(rgb), 7) < 0.14
    greenish = (f[..., 1] > f[..., 0] + 0.03) & (f[..., 1] > f[..., 2] + 0.03)
    return (sat < 0.16) & (v > 0.15) & (v < 0.8) & ~greenish & smooth


def tile_attributes(rgb: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    """Interpretable per-tile attributes used for filtering, re-ranking and explanations."""
    if valid is None:
        valid = np.ones(rgb.shape[:2], bool)
    n = max(int(valid.sum()), 1)
    masks = cover_masks(rgb)
    gy = gray(rgb)
    tex = local_std(gy, 5)
    out = {k: float((m & valid).sum() / n) for k, m in masks.items()}
    out["brightness"] = float(gy[valid].mean()) if valid.any() else 0.0
    out["texture"] = float(tex[valid].mean()) if valid.any() else 0.0
    # built-up evidence: bright, desaturated *and* busy; smooth bright surfaces are cloud or sand
    busy = tex > 0.045
    out["built_busy"] = float((masks["built"] & busy & valid).sum() / n)
    # urban fabric = bright/pinkish surfaces broken up at pixel scale (roofs, streets, yards);
    # bare farmland has the same colours but is smooth, which is what separates the two
    urban_px = (masks["built"] | masks["bare"]) & (local_std(gy, 5) > 0.07)
    out["urban"] = float((ndi.binary_closing(urban_px, iterations=2) & valid).sum() / n)
    cl = cloud_mask(rgb)
    out["cloud"] = float((cl & valid).sum() / n)
    out["valid"] = float(valid.mean())
    return out


# --------------------------------------------------------------------------- quality
def cloud_mask(rgb: np.ndarray) -> np.ndarray:
    """Bright, desaturated, smooth, spatially extensive pixels. Dense urban roofs are busy and
    small-scale, so they are excluded by the texture and blob-size conditions."""
    v, sat, _ = hsv_arrays(rgb)
    gy = gray(rgb)
    smooth = local_std(gy, 7) < 0.035
    cand = (v > 0.72) & (sat < 0.16) & smooth
    cand = ndi.binary_opening(cand, iterations=1)
    lab, n = ndi.label(cand)
    if n == 0:
        return np.zeros_like(cand)
    sizes = ndi.sum(cand, lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= 150))
    return ndi.binary_dilation(keep, iterations=2)


def haze_score(rgb: np.ndarray, valid: np.ndarray | None = None) -> float:
    """0 = crisp, 1 = heavily hazed. Haze lifts the dark end of the histogram and lowers contrast."""
    f = to_float(rgb)
    gy = gray(rgb)
    if valid is not None and valid.any():
        gy = gy[valid]
        f = f[valid]
    else:
        f = f.reshape(-1, 3)
        gy = gy.ravel()
    p2 = np.percentile(gy, 2)
    spread = np.percentile(gy, 98) - p2
    blue_lift = float(np.percentile(f[:, 2], 2))
    return float(np.clip(0.6 * np.clip((p2 - 0.12) / 0.3, 0, 1) + 0.4 * np.clip(1 - spread / 0.55, 0, 1)
                         + 0.0 * blue_lift, 0, 1))


def assess_quality(rgb: np.ndarray, valid: np.ndarray | None = None) -> dict:
    if valid is None:
        valid = np.ones(rgb.shape[:2], bool)
    cl = cloud_mask(rgb) & valid
    dark_cluster = ndi.binary_dilation(cl, iterations=12) & (gray(rgb) < 0.14) & valid  # shadows next to cloud
    usable = valid & ~cl & ~dark_cluster
    haze = haze_score(rgb, valid)
    return {
        "valid_frac": float(valid.mean()),
        "cloud_frac": float(cl.sum() / max(valid.sum(), 1)),
        "shadow_frac": float(dark_cluster.sum() / max(valid.sum(), 1)),
        "haze": haze,
        "usable_frac": float(usable.mean()),
        "usable_mask": usable,
        "cloud_mask": cl,
    }


# --------------------------------------------------------------------------- descriptor
_LBP_P, _LBP_R = 8, 1


def descriptor(rgb: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Hand-built remote-sensing descriptor (colour, spectral relations, texture, coarse layout)."""
    f = to_float(rgb)
    gy = gray(rgb)
    v, sat, _ = hsv_arrays(rgb)
    parts: list[np.ndarray] = []

    # colour statistics
    parts.append(f.reshape(-1, 3).mean(0))
    parts.append(f.reshape(-1, 3).std(0))
    # joint-ish histograms: value, saturation, blue-red balance
    parts.append(np.histogram(v, bins=8, range=(0, 1))[0] / v.size)
    parts.append(np.histogram(sat, bins=8, range=(0, 1))[0] / sat.size)
    parts.append(np.histogram(f[..., 2] - f[..., 0], bins=8, range=(-0.4, 0.4))[0] / v.size)
    parts.append(np.histogram(f[..., 1] - f[..., 0], bins=8, range=(-0.4, 0.4))[0] / v.size)

    # texture: gradient magnitude / orientation
    gx = ndi.sobel(gy, axis=1) / 8.0
    gyy = ndi.sobel(gy, axis=0) / 8.0
    mag = np.hypot(gx, gyy)
    ang = (np.arctan2(gyy, gx) % np.pi)
    parts.append(np.histogram(mag, bins=8, range=(0, 0.25))[0] / mag.size)
    hist_o, _ = np.histogram(ang, bins=8, range=(0, np.pi), weights=mag)
    parts.append(hist_o / max(hist_o.sum(), 1e-6))
    parts.append(np.array([mag.mean(), (mag > 0.08).mean()]))
    # LBP texture
    lbp = local_binary_pattern((gy * 255).astype(np.uint8), _LBP_P, _LBP_R, method="uniform")
    parts.append(np.histogram(lbp, bins=_LBP_P + 2, range=(0, _LBP_P + 2))[0] / lbp.size)
    # multi-scale local variance
    parts.append(np.array([local_std(gy, 3).mean(), local_std(gy, 9).mean(), local_std(gy, 21).mean()]))

    # coarse 2x2 layout (brightness, blue-red)
    h, w = gy.shape
    for sl_y in (slice(0, h // 2), slice(h // 2, h)):
        for sl_x in (slice(0, w // 2), slice(w // 2, w)):
            parts.append(np.array([gy[sl_y, sl_x].mean(), (f[sl_y, sl_x, 2] - f[sl_y, sl_x, 0]).mean()]))

    # cover fractions
    m = cover_masks(rgb)
    parts.append(np.array([m[k].mean() for k in ("water", "veg", "bare", "built", "dark")]))
    return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])


DESCRIPTOR_DIM = None  # resolved lazily


def descriptor_dim() -> int:
    global DESCRIPTOR_DIM
    if DESCRIPTOR_DIM is None:
        DESCRIPTOR_DIM = int(descriptor(np.zeros((config.TILE_SIZE, config.TILE_SIZE, 3), np.uint8)).size)
    return DESCRIPTOR_DIM
