"""Builds the demo sites.

Real imagery supplies every pixel of ground truth content. What is synthetic is *time*: no
public dataset reachable offline ships dense multi-date series with known outcomes, so epochs are
derived from real images and every derived epoch is flagged `synthetic` in the database and UI.

Sites
  * LEVIR-CD chips (real before/after pairs, hand-labelled change masks). The two real epochs are
    kept as is; extra epochs injecting cloud, misregistration, haze are synthetic.
  * Landsat 8 windows (real 30 m scene). Epochs after the first are synthetic variations
    (seasonal tone shift, cloud, shift, pasted built-up patch).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from scipy import ndimage as ndi

from . import config, db, imaging, ingest

RNG = np.random.default_rng(26227)
SITE_DIR = config.PROCESSED / "sites"


# --------------------------------------------------------------------------- degradations
def add_cloud(rgb: np.ndarray, cover: float = 0.5, seed: int = 1) -> np.ndarray:
    r = np.random.default_rng(seed)
    h, w = rgb.shape[:2]
    noise = ndi.gaussian_filter(r.standard_normal((h, w)), max(h, w) / 9)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    thr = np.quantile(noise, 1 - cover)
    alpha = np.clip((noise - thr) / 0.08, 0, 1)
    alpha = ndi.gaussian_filter(alpha, 3) * 0.97
    cloud = np.array([244, 246, 248], np.float32)
    out = rgb.astype(np.float32) * (1 - alpha[..., None]) + cloud * alpha[..., None]
    # shadow, displaced down-right
    sh = ndi.shift(alpha, (h * 0.05, w * 0.05), order=1)
    out *= (1 - 0.45 * np.clip(sh - alpha, 0, 1))[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_haze(rgb: np.ndarray, amount: float = 0.45) -> np.ndarray:
    f = rgb.astype(np.float32)
    veil = np.array([200, 205, 215], np.float32)
    return np.clip(f * (1 - amount) + veil * amount, 0, 255).astype(np.uint8)


def shift_image(rgb: np.ndarray, dy: float, dx: float) -> np.ndarray:
    return np.stack([ndi.shift(rgb[..., c].astype(np.float32), (dy, dx), order=1, mode="nearest")
                     for c in range(3)], -1).clip(0, 255).astype(np.uint8)


def seasonal(rgb: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Dry-season look: vegetation loses green and brightens toward brown, everything else barely moves."""
    f = rgb.astype(np.float32)
    masks = imaging.cover_masks(rgb)
    veg = ndi.gaussian_filter(masks["veg"].astype(np.float32), 1.0)[..., None]
    brown = np.array([150, 125, 85], np.float32)
    f = f * (1 - strength * 0.7 * veg) + brown * (strength * 0.7 * veg)
    return np.clip(f * 1.03, 0, 255).astype(np.uint8)


def radiometric_drift(rgb: np.ndarray, gain=(1.08, 1.04, 0.96), off=(6, 4, -3)) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * np.array(gain) + np.array(off), 0, 255).astype(np.uint8)


def noise(rgb: np.ndarray, sigma: float = 3.0, seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed)
    return np.clip(rgb.astype(np.float32) + r.normal(0, sigma, rgb.shape), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- storage
def save_obs(site_id: str, tag: str, rgb: np.ndarray) -> str:
    d = SITE_DIR / site_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{tag}.png"
    Image.fromarray(rgb).save(p, optimize=True)
    return p.relative_to(config.PROCESSED).as_posix()


def _site(s, site_id: str, **kw) -> db.Site:
    old = s.get(db.Site, site_id)
    if old:
        for o in list(old.observations):
            s.delete(o)
        s.delete(old)
        s.flush()
    site = db.Site(site_id=site_id, **kw)
    s.add(site)
    return site


def _obs(s, site: db.Site, tag: str, date: str, rgb: np.ndarray, sensor: str, synthetic: bool,
         note: str, scene_id: str | None = None) -> None:
    path = save_obs(site.site_id, tag, rgb)
    s.add(db.Observation(site_id=site.site_id, scene_id=scene_id, date=date, image_path=path,
                         sensor=sensor, synthetic=synthetic, note=note))


# --------------------------------------------------------------------------- LEVIR sites
LEVIR = {
    "levir-test2-a": ("test_2_0000_0000.png", "Housing development, chip A", 30.2672, -97.7431),
    "levir-test2-b": ("test_2_0000_0512.png", "Housing development, chip B", 30.2672, -97.7431),
    "levir-test7": ("test_7_0256_0512.png", "Subdivision with new access road", 30.2672, -97.7431),
}


def _levir_arrays(name: str):
    r = config.RAW / "levir"
    a = np.array(Image.open(r / "A" / name).convert("RGB"))
    b = np.array(Image.open(r / "B" / name).convert("RGB"))
    t = np.array(Image.open(r / "label" / name).convert("L")) > 127
    return a, b, t


def build_levir_sites() -> list[str]:
    ids = []
    with db.session() as s:
        for sid, (name, title, *_ ) in LEVIR.items():
            a, b, truth = _levir_arrays(name)
            tp = SITE_DIR / sid
            tp.mkdir(parents=True, exist_ok=True)
            Image.fromarray((truth * 255).astype(np.uint8)).save(tp / "truth.png")
            site = _site(s, sid, name=title, kind="levir-chip", georeferenced=False, polygon=None,
                         lon=None, lat=None, gsd_m=0.5, scenario="real-pair",
                         note="LEVIR-CD sample (Chen & Shi 2020). Chips carry no coordinates. Capture "
                              "dates are not published per chip (dataset spans 2002-2018); epochs below "
                              "use nominal dates and are not acquisition dates.",
                         truth_path=(tp / "truth.png").relative_to(config.PROCESSED).as_posix(),
                         expected={"first_change": "2018-06-01", "type": "construction",
                                   "truth_frac": float(truth.mean())})
            _obs(s, site, "e0", "2008-06-01", a, "aerial/VHR", False, "LEVIR-CD image A (real, earlier epoch)")
            _obs(s, site, "e1", "2018-06-01", b, "aerial/VHR", False, "LEVIR-CD image B (real, later epoch)")
            ids.append(sid)

        # synthetic series built from the real pair: demonstrates false-alarm suppression
        a, b, truth = _levir_arrays("test_2_0000_0000.png")
        site = _site(s, "levir-series-cloud", name="Housing development, with cloudy interim pass",
                     kind="levir-chip", georeferenced=False, gsd_m=0.5, scenario="cloud-then-real-change",
                     note="Real before/after pair plus synthetic interim epochs: one heavily clouded, "
                          "one hazy. The clouded epoch should NOT be reported as the change date.",
                     truth_path=(SITE_DIR / "levir-test2-a" / "truth.png").relative_to(config.PROCESSED).as_posix(),
                     expected={"first_change": "2016-06-01", "type": "construction",
                               "rejected": ["2013-06-01", "2015-06-01"]})
        _obs(s, site, "e0", "2008-06-01", a, "aerial/VHR", False, "real, earlier epoch")
        _obs(s, site, "e1", "2013-06-01", add_cloud(a, 0.55, seed=11), "aerial/VHR", True,
             "synthetic: 55% cloud over earlier epoch")
        _obs(s, site, "e2", "2015-06-01", add_haze(a, 0.55), "aerial/VHR", True, "synthetic: dense haze")
        _obs(s, site, "e3", "2016-06-01", noise(b, 2, 3), "aerial/VHR", True,
             "later epoch (real image B with sensor noise)")
        _obs(s, site, "e4", "2018-06-01", b, "aerial/VHR", False, "real, later epoch")
        ids.append("levir-series-cloud")

        a, b, truth = _levir_arrays("test_7_0256_0512.png")
        site = _site(s, "levir-series-shift", name="Subdivision, misregistered interim pass",
                     kind="levir-chip", georeferenced=False, gsd_m=0.5, scenario="misregistration-then-real-change",
                     note="Synthetic interim epochs of the unchanged earlier scene: shifted by 5 px and "
                          "radiometrically drifted. These must not be reported as change.",
                     truth_path=(SITE_DIR / "levir-test7" / "truth.png").relative_to(config.PROCESSED).as_posix(),
                     expected={"first_change": "2018-06-01", "type": "construction",
                               "rejected": ["2011-06-01", "2014-06-01"]})
        _obs(s, site, "e0", "2008-06-01", a, "aerial/VHR", False, "real, earlier epoch")
        _obs(s, site, "e1", "2011-06-01", shift_image(a, 4, -3), "aerial/VHR", True, "synthetic: 5 px offset")
        _obs(s, site, "e2", "2014-06-01", radiometric_drift(shift_image(a, -2, 3)), "aerial/VHR", True,
             "synthetic: 3.6 px offset plus radiometric drift")
        _obs(s, site, "e3", "2018-06-01", b, "aerial/VHR", False, "real, later epoch")
        ids.append("levir-series-shift")
    return ids


# --------------------------------------------------------------------------- Landsat sites
def _landsat_window(scene_id: str, col: int, row: int, size: int = 192):
    rgb, valid = ingest.read_scene_rgb(scene_id, (col, row, size, size))
    return rgb, valid


def _paste_urban(rgb: np.ndarray, src_rgb: np.ndarray, y: int, x: int, h: int, w: int) -> np.ndarray:
    out = rgb.copy()
    patch = src_rgb[:h, :w]
    m = np.zeros(out.shape[:2], bool)
    m[y:y + h, x:x + w] = True
    soft = ndi.gaussian_filter(m.astype(np.float32), 1.2)[..., None]
    canvas = out.astype(np.float32)
    big = np.zeros_like(canvas)
    big[y:y + h, x:x + w] = patch
    canvas = canvas * (1 - soft) + big * soft
    return canvas.clip(0, 255).astype(np.uint8)


def build_landsat_sites() -> list[str]:
    scene = "LC08_L1TP_224078_20200518_20200518_01_RT"
    with db.session() as s:
        if not s.get(db.Scene, scene):
            return []
        sc = s.get(db.Scene, scene)
        tr = sc.qa
    # choose windows from the real tile statistics: greenfield base, dense-urban donor, shore window
    with db.session() as s:
        tiles = s.query(db.Tile).filter(db.Tile.scene_id == scene).all()
        rows = [(t.row, t.col, t.attrs) for t in tiles]
    donor_rc = max(rows, key=lambda r: r[2]["urban"] * (1 - r[2]["water"]))
    greens = [r for r in rows if r[2]["urban"] < 0.08 and r[2]["water"] < 0.02 and r[2]["cloud"] < 0.01
              and (r[2]["veg"] + r[2]["bare"]) > 0.85]
    greens.sort(key=lambda r: (r[0], r[1]))
    base_rc = greens[len(greens) // 3]
    farm_rc = greens[(2 * len(greens)) // 3]
    shore = [r for r in rows if 0.25 < r[2]["water"] < 0.6]
    shore_rc = shore[len(shore) // 2]
    T = config.TILE_SIZE
    urban = _landsat_window(scene, base_rc[1] * T, base_rc[0] * T, 192)
    farm = _landsat_window(scene, farm_rc[1] * T, farm_rc[0] * T, 192)
    shore_w = _landsat_window(scene, shore_rc[1] * T, shore_rc[0] * T, 192)
    donor, _ = _landsat_window(scene, donor_rc[1] * T, donor_rc[0] * T, 192)
    ids = []
    with db.session() as s:
        sc = s.get(db.Scene, scene)
        left, top = 0.0, 0.0
        with rasterio.open(config.PROCESSED / sc.cog_path) as src:
            def lonlat(col, row):
                from rasterio.warp import transform
                x, y = src.transform * (col, row)
                lo, la = transform(src.crs, "EPSG:4326", [x], [y])
                return lo[0], la[0]
            c1 = lonlat(base_rc[1] * T + 96, base_rc[0] * T + 96)
            c2 = lonlat(farm_rc[1] * T + 96, farm_rc[0] * T + 96)
            c3 = lonlat(shore_rc[1] * T + 96, shore_rc[0] * T + 96)

        # 1) real urban growth simulated: built-up patch appears at epoch 3, cloud at epoch 2
        base, bv = urban
        site = _site(s, "ls-urban-growth", name="Cropland near Ciudad del Este: simulated urban growth",
                     kind="landsat-window", georeferenced=True, lon=c1[0], lat=c1[1], gsd_m=30.0,
                     scenario="cloud-then-growth",
                     note="Landsat 8 window (real, 2020-05-18). Later epochs are synthetic: a dense "
                          "a patch of urban fabric (copied from the city in the same scene) is added from 2022 "
                          "and a 2021 pass is cloud-covered. Expected: change dated 2022, not 2021.",
                     expected={"first_change": "2022-05-18", "type": "construction",
                               "rejected": ["2021-05-18"]})
        grown = _paste_urban(base, donor, 70, 80, 40, 52)
        _obs(s, site, "e0", "2020-05-18", base, "OLI", False, "real Landsat 8 window", scene)
        _obs(s, site, "e1", "2021-05-18", add_cloud(base, 0.6, seed=4), "OLI", True, "synthetic: cloud-covered pass")
        _obs(s, site, "e2", "2022-05-18", noise(radiometric_drift(grown, (1.03, 1.02, 1.0), (2, 2, 0)), 1.5, 7),
             "OLI", True, "synthetic: built-up patch added")
        _obs(s, site, "e3", "2023-05-18", noise(grown, 1.5, 9), "OLI", True, "synthetic: change persists")
        ids.append(site.site_id)

        # 2) seasonal vegetation only: must be suppressed
        fbase, fv = farm
        site = _site(s, "ls-seasonal", name="Cropland: seasonal change only",
                     kind="landsat-window", georeferenced=True, lon=c2[0], lat=c2[1], gsd_m=30.0,
                     scenario="seasonal-only",
                     note="Landsat 8 cropland window. The later epoch is synthetic and differs only by a "
                          "dry-season vegetation tone shift six months after the first. A good system "
                          "should not call this construction or clearance.",
                     expected={"first_change": None, "rejected": ["2020-11-18"]})
        _obs(s, site, "e0", "2020-05-18", fbase, "OLI", False, "real Landsat 8 window", scene)
        _obs(s, site, "e1", "2020-11-18", seasonal(fbase, 0.55), "OLI", True, "synthetic: dry-season shift")
        ids.append(site.site_id)

        # 3) misregistration only: no change
        base = shore_w[0]
        site = _site(s, "ls-shift-only", name="Reservoir shore: registration offset only",
                     kind="landsat-window", georeferenced=True, lon=c3[0], lat=c3[1], gsd_m=30.0,
                     scenario="misregistration-only",
                     note="Same window, later epoch offset by 2 px and radiometrically drifted. No real "
                          "change: a correct pipeline re-aligns and reports nothing.",
                     expected={"first_change": None})
        _obs(s, site, "e0", "2020-05-18", base, "OLI", False, "real Landsat 8 window", scene)
        _obs(s, site, "e1", "2021-05-18", radiometric_drift(shift_image(base, 2, -2)), "OLI", True,
             "synthetic: 2 px offset and drift")
        ids.append(site.site_id)
    return ids


def build_all() -> dict:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    a = build_levir_sites()
    b = build_landsat_sites()
    return {"levir": a, "landsat": b}
