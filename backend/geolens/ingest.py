"""Geospatial ingestion: validate -> COG -> web overlay -> tile -> assess -> embed -> index.

New scenes are inserted next to what is already indexed; nothing is rebuilt. The embedding scaler
is frozen at first build so vectors from different ingest runs stay comparable.
"""
from __future__ import annotations

import datetime as dt
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
import rasterio.shutil
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, transform as warp_transform
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from sqlalchemy import func

from . import config, db, imaging
from .embedding import DescriptorEmbedder
from .index import VectorIndex, date_ord

LANDSAT_RE = re.compile(r"(LC0[89]|LE07|LT05)_(\w{4})_(\d{3})(\d{3})_(\d{8})_")
PLATFORM = {"LC08": "Landsat 8", "LC09": "Landsat 9", "LE07": "Landsat 7", "LT05": "Landsat 5"}


class IngestError(Exception):
    pass


# --------------------------------------------------------------------------- helpers
def parse_landsat_name(name: str) -> dict[str, Any] | None:
    m = LANDSAT_RE.search(name)
    if not m:
        return None
    prefix, level, path, row, ymd = m.groups()
    return {"platform": PLATFORM[prefix], "sensor": "OLI/TIRS" if prefix in ("LC08", "LC09") else "ETM+/TM",
            "level": level, "path": path, "row": row,
            "date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}",
            "scene_id": re.sub(r"_B\d+.*$", "", Path(name).stem)}


def _poly(transform: Affine, crs: Any, col: int, row: int, w: int, h: int):
    xs = [col, col + w, col + w, col]
    ys = [row, row, row + h, row + h]
    px = [transform * (x, y) for x, y in zip(xs, ys)]
    lon, lat = warp_transform(crs, "EPSG:4326", [p[0] for p in px], [p[1] for p in px])
    ring = [[float(a), float(b)] for a, b in zip(lon, lat)]
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}, (min(lon), min(lat), max(lon), max(lat))


def validate_raster(src: rasterio.DatasetReader) -> None:
    if src.crs is None:
        raise IngestError("raster has no coordinate reference system")
    if src.transform == Affine.identity():
        raise IngestError("raster is not georeferenced (identity transform)")
    if src.count < 3:
        raise IngestError(f"expected at least 3 bands (R,G,B), found {src.count}")
    b = src.bounds
    if not all(np.isfinite([b.left, b.bottom, b.right, b.top])):
        raise IngestError("raster bounds are not finite")


def scene_dir(kind: str, scene_id: str) -> Path:
    d = config.PROCESSED / kind / scene_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel(p: Path) -> str:
    return p.relative_to(config.PROCESSED).as_posix()


def read_scene_rgb(scene_id: str, window: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8, valid bool) for a scene or a (col, row, w, h) window of it."""
    with db.session() as s:
        sc = s.get(db.Scene, scene_id)
        cog, dtype = config.PROCESSED / sc.cog_path, (sc.qa or {}).get("dtype", "uint16")
    with rasterio.open(cog) as src:
        win = Window(*window) if window else None
        arr = src.read([1, 2, 3], window=win)
    return dn_to_rgb(arr, dtype)


def dn_to_rgb(arr: np.ndarray, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    valid = (arr > 0).all(0)
    if dtype == "uint8":
        rgb = np.moveaxis(arr, 0, -1).astype(np.uint8)
    else:
        rgb = imaging.stretch_dn(arr[0], arr[1], arr[2])
    rgb[~valid] = 0
    return rgb, valid


# --------------------------------------------------------------------------- overlay
def make_overlay(rgb: np.ndarray, valid: np.ndarray, transform: Affine, crs: Any, out: Path) -> list[float]:
    h, w = rgb.shape[:2]
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    dst_t, dw, dh = calculate_default_transform(crs, "EPSG:4326", w, h, left=min(left, right),
                                                bottom=min(top, bottom), right=max(left, right),
                                                top=max(top, bottom))
    scale = min(1.0, 2200 / max(dw, dh))
    if scale < 1:
        dst_t = dst_t * Affine.scale(1 / scale)
        dw, dh = int(dw * scale), int(dh * scale)
    chans = []
    for c in range(3):
        dst = np.zeros((dh, dw), np.uint8)
        reproject(rgb[..., c], dst, src_transform=transform, src_crs=crs, dst_transform=dst_t,
                  dst_crs="EPSG:4326", resampling=Resampling.bilinear)
        chans.append(dst)
    a = np.zeros((dh, dw), np.uint8)
    reproject(valid.astype(np.uint8) * 255, a, src_transform=transform, src_crs=crs,
              dst_transform=dst_t, dst_crs="EPSG:4326", resampling=Resampling.nearest)
    Image.fromarray(np.dstack(chans + [a])).save(out, format="WEBP", quality=88, method=4)
    west, north = dst_t * (0, 0)
    east, south = dst_t * (dw, dh)
    return [float(west), float(south), float(east), float(north)]


# --------------------------------------------------------------------------- scene processing
def _next_point_id(s) -> int:
    m = s.query(func.max(db.Tile.point_id)).scalar()
    return int(m or 0) + 1


def process_scene(dn: np.ndarray, transform: Affine, crs: Any, meta: dict[str, Any], *,
                  index: VectorIndex, embedder: DescriptorEmbedder, replace: bool = False,
                  progress: Callable[[str, float], None] | None = None,
                  run_id: str | None = None) -> dict[str, Any]:
    """dn: (3, H, W) array in the sensor's native values. meta needs scene_id/platform/sensor/date."""
    t0 = time.time()
    note = progress or (lambda *_: None)
    scene_id = meta["scene_id"]
    dtype = "uint8" if dn.dtype == np.uint8 else "uint16"
    with db.session() as s:
        exists = s.get(db.Scene, scene_id) is not None
    if exists and not replace:
        raise IngestError(f"scene {scene_id} is already indexed (use replace to re-ingest it)")
    if exists:
        remove_scene(scene_id, index)

    _, h, w = dn.shape
    rgb, valid = dn_to_rgb(dn, dtype)
    note("writing cloud-optimised GeoTIFF", 0.05)
    out_dir = scene_dir("cog", scene_id)
    tmp = out_dir / "tmp.tif"
    prof = {"driver": "GTiff", "height": h, "width": w, "count": 3, "dtype": dn.dtype, "crs": crs,
            "transform": transform, "tiled": True, "blockxsize": 256, "blockysize": 256}
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(dn)
    cog_path = out_dir / f"{scene_id}.tif"
    rasterio.shutil.copy(tmp, cog_path, driver="COG", compress="deflate", blocksize=256, overview_resampling="average")
    tmp.unlink()

    note("building map overlay", 0.15)
    ov_dir = scene_dir("overlays", scene_id)
    overlay_path = ov_dir / "overlay.webp"
    overlay_bounds = make_overlay(rgb, valid, transform, crs, overlay_path)
    bbox = [float(v) for v in transform_bounds(crs, "EPSG:4326", *_bounds(transform, h, w))]

    T = config.TILE_SIZE
    tile_dir = scene_dir("tiles", scene_id)
    gsd = float(abs(transform.a))
    windows = [(c, r) for r in range(0, h - T + 1, T) for c in range(0, w - T + 1, T)]
    records: list[dict[str, Any]] = []
    raws: list[np.ndarray] = []
    note("tiling and assessing quality", 0.2)
    for n, (c, r) in enumerate(windows):
        t_rgb = rgb[r:r + T, c:c + T]
        t_valid = valid[r:r + T, c:c + T]
        if t_valid.mean() < config.MIN_TILE_VALID:
            continue
        q = imaging.assess_quality(t_rgb, t_valid)
        attrs = imaging.tile_attributes(t_rgb, t_valid)
        attrs["haze"] = q["haze"]
        quality = float(np.clip(q["usable_frac"] * (1 - 0.5 * q["haze"]), 0, 1))
        geom, (x0, y0, x1, y1) = _poly(transform, crs, c, r, T, T)
        tid = f"{scene_id}_r{r // T:02d}_c{c // T:02d}"
        Image.fromarray(t_rgb).save(tile_dir / f"{tid}.png", optimize=True)
        raws.append(embedder.raw(t_rgb, t_valid))
        records.append({"tile_id": tid, "row": r // T, "col": c // T, "window": [c, r, T, T],
                        "polygon": geom, "minx": x0, "miny": y0, "maxx": x1, "maxy": y1,
                        "lon": (x0 + x1) / 2, "lat": (y0 + y1) / 2,
                        "image_path": rel(tile_dir / f"{tid}.png"), "attrs": attrs, "quality": quality})
        if n % 25 == 0:
            note("tiling and assessing quality", 0.2 + 0.4 * n / max(len(windows), 1))
    if not records:
        raise IngestError("no tile had enough valid pixels")

    note("embedding tiles", 0.65)
    raw_arr = np.vstack(raws)
    if not embedder.fitted:
        embedder.fit(raw_arr)
    vecs = embedder.transform(raw_arr)

    note("inserting into index and database", 0.85)
    date = meta["date"]
    with db.session() as s:
        pid = _next_point_id(s)
        s.add(db.Scene(
            scene_id=scene_id, title=meta.get("title", scene_id), sensor=meta["sensor"],
            platform=meta["platform"], acquisition_date=date, crs=str(crs), resolution_m=gsd,
            width=w, height=h, bbox_wgs84=bbox, source_path=meta.get("source_path"),
            cog_path=rel(cog_path), overlay_path=rel(overlay_path), origin=meta.get("origin", ""),
            license=meta.get("license", ""), synthetic=bool(meta.get("synthetic")),
            georeferenced=True, parent_scene=meta.get("parent_scene"),
            qa={"dtype": dtype, "overlay_bounds": overlay_bounds, "tiles": len(records),
                "valid_frac": float(valid.mean()), "raster_crs": str(crs)}))
        points = []
        for rec, vec in zip(records, vecs):
            s.add(db.Tile(scene_id=scene_id, point_id=pid, embedder=embedder.name, **rec))
            points.append({"id": pid, "vector": vec, "payload": tile_payload(rec, meta, scene_id)})
            pid += 1
        index.upsert(points)
        s.add(db.ProcessingRun(
            id=run_id or f"run-{uuid.uuid4().hex[:8]}", kind="ingest", scene_id=scene_id,
            pipeline_version=config.PIPELINE_VERSION, model_version=embedder.name,
            params={"tile_size": T, "min_tile_valid": config.MIN_TILE_VALID, "replace": replace},
            status="done", finished=db.utcnow(),
            stats={"tiles": len(records), "seconds": round(time.time() - t0, 2),
                   "area_km2": round(len(records) * (T * gsd / 1000) ** 2, 1)}))
    note("done", 1.0)
    return {"scene_id": scene_id, "tiles": len(records), "seconds": round(time.time() - t0, 2)}


def _bounds(transform: Affine, h: int, w: int):
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (w, h)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def tile_payload(rec: dict, meta: dict, scene_id: str) -> dict:
    a = rec["attrs"]
    return {"tile_id": rec["tile_id"], "scene_id": scene_id, "sensor": meta["sensor"],
            "date": meta["date"], "date_ord": date_ord(meta["date"]), "quality": rec["quality"],
            "synthetic": bool(meta.get("synthetic")),
            "center": {"lon": rec["lon"], "lat": rec["lat"]},
            **{k: round(float(a[k]), 4) for k in ("water", "veg", "bare", "built", "urban", "dark",
                                                  "texture", "brightness", "cloud")}}


def remove_scene(scene_id: str, index: VectorIndex) -> None:
    index.delete_scene(scene_id)
    with db.session() as s:
        for t in s.query(db.Tile).filter(db.Tile.scene_id == scene_id).all():
            s.delete(t)
        sc = s.get(db.Scene, scene_id)
        if sc:
            s.delete(sc)
    for kind in ("tiles", "cog", "overlays"):
        d = config.PROCESSED / kind / scene_id
        if d.exists():
            shutil.rmtree(d)


# --------------------------------------------------------------------------- entry points
def ingest_landsat_bands(bands: dict[str, Path], *, index: VectorIndex, embedder: DescriptorEmbedder,
                         replace: bool = False, progress=None, run_id: str | None = None) -> dict:
    """bands: {'R': B4 path, 'G': B3 path, 'B': B2 path} of a Landsat 8 scene."""
    info = parse_landsat_name(bands["R"].name)
    if not info:
        raise IngestError("cannot parse Landsat product name")
    arrays, transform, crs = [], None, None
    for k in ("R", "G", "B"):
        with rasterio.open(bands[k]) as src:
            if src.count != 1:
                raise IngestError(f"{bands[k].name}: expected a single-band raster")
            validate_raster_single(src)
            arrays.append(src.read(1))
            if transform is None:
                transform, crs = src.transform, src.crs
            elif src.transform != transform:
                raise IngestError("band rasters are not on the same grid")
    meta = {"scene_id": info["scene_id"], "title": f"{info['platform']} path {info['path']} row {info['row']}",
            "platform": info["platform"], "sensor": info["sensor"], "date": info["date"],
            "source_path": ";".join(str(bands[k].name) for k in "RGB"),
            "origin": "USGS/NASA Landsat 8 OLI, Level-1 (bands 4,3,2)", "license": "Public domain (USGS)"}
    return process_scene(np.stack(arrays), transform, crs, meta, index=index, embedder=embedder,
                         replace=replace, progress=progress, run_id=run_id)


def validate_raster_single(src: rasterio.DatasetReader) -> None:
    if src.crs is None or src.transform == Affine.identity():
        raise IngestError(f"{Path(src.name).name}: raster is not georeferenced")


def ingest_geotiff(path: Path, *, index: VectorIndex, embedder: DescriptorEmbedder,
                   date: str | None = None, sensor: str = "unknown", platform: str = "unknown",
                   scene_id: str | None = None, replace: bool = False, progress=None,
                   run_id: str | None = None) -> dict:
    """Any georeferenced GeoTIFF/COG with R,G,B in bands 1-3 (uint8 or uint16)."""
    with rasterio.open(path) as src:
        validate_raster(src)
        if src.dtypes[0] not in ("uint8", "uint16"):
            raise IngestError(f"unsupported data type {src.dtypes[0]} (uint8 or uint16 expected)")
        dn = src.read([1, 2, 3])
        transform, crs = src.transform, src.crs
        tags = src.tags()
    info = parse_landsat_name(path.name)
    if info:
        date = date or info["date"]
        sensor = info["sensor"] if sensor == "unknown" else sensor
        platform = info["platform"] if platform == "unknown" else platform
    if not date:
        stamp = tags.get("TIFFTAG_DATETIME")
        date = stamp[:10].replace(":", "-") if stamp else dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
    meta = {"scene_id": scene_id or re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem), "title": path.stem,
            "platform": platform, "sensor": sensor, "date": date, "source_path": path.name,
            "origin": "user-supplied GeoTIFF", "license": "as supplied"}
    if dn.dtype == np.uint16 and info is None:
        # unknown 16-bit product: derive a stretch from the data itself and keep it with the scene
        lo = np.percentile(dn[dn > 0], 2) if (dn > 0).any() else 0
        hi = np.percentile(dn[dn > 0], 98) if (dn > 0).any() else 1
        dn = np.clip((dn.astype(np.float32) - lo) / max(hi - lo, 1) * 254 + 1, 1, 255).astype(np.uint8) \
            * (dn > 0).astype(np.uint8)
        meta["origin"] += " (16-bit values stretched to 8-bit on ingest)"
    return process_scene(dn, transform, crs, meta, index=index, embedder=embedder, replace=replace,
                         progress=progress, run_id=run_id)
