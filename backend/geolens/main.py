"""GeoLens API + static front end. Run: uvicorn geolens.main:app --port 8000"""
from __future__ import annotations

import io
import json
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy import func

from . import analysis, config, db, ingest, search
from .change import CHANGE_TYPES
from .embedding import DescriptorEmbedder
from .index import VectorIndex

FRONTEND = config.ROOT / "frontend"
state: dict[str, Any] = {}
JOBS: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    state["index"] = VectorIndex()
    state["embedder"] = DescriptorEmbedder()
    yield
    state["index"].close()


app = FastAPI(title="GeoLens", version="0.1.0", lifespan=lifespan)
app.mount("/files", StaticFiles(directory=str(config.PROCESSED)), name="files")


def furl(path: str | None) -> str | None:
    return f"/files/{path}" if path else None


# ------------------------------------------------------------------------ overview
@app.get("/api/overview")
def overview():
    with db.session() as s:
        scenes = s.query(db.Scene).count()
        tiles = s.query(db.Tile).count()
        sites = s.query(db.Site).count()
        by_status = dict(s.query(db.ChangeEvent.status, func.count()).group_by(db.ChangeEvent.status).all())
        reviews = dict(s.query(db.AnalystReview.decision, func.count()).group_by(db.AnalystReview.decision).all())
        area = s.query(func.sum(db.Scene.width * db.Scene.height * db.Scene.resolution_m * db.Scene.resolution_m)).scalar() or 0
    return {"scenes": scenes, "tiles": tiles, "indexed_vectors": state["index"].count(), "sites": sites,
            "changes": by_status, "reviews": reviews, "area_km2": round(area / 1e6, 1),
            "embedder": state["embedder"].name, "pipeline": config.PIPELINE_VERSION,
            "change_method": config.CHANGE_METHOD, "tile_size_px": config.TILE_SIZE}


@app.get("/api/scenes")
def scenes():
    with db.session() as s:
        out = []
        for sc in s.query(db.Scene).order_by(db.Scene.acquisition_date):
            d = db.row_dict(sc, exclude=("cog_path", "source_path"))
            d["overlay_url"] = furl(sc.overlay_path)
            d["overlay_bounds"] = (sc.qa or {}).get("overlay_bounds")
            out.append(d)
        return out


@app.delete("/api/scenes/{scene_id}")
def delete_scene(scene_id: str):
    with db.session() as s:
        if not s.get(db.Scene, scene_id):
            raise HTTPException(404, "unknown scene")
    ingest.remove_scene(scene_id, state["index"])
    return {"ok": True, "indexed_vectors": state["index"].count()}


@app.get("/api/scenes/{scene_id}/tiles")
def scene_tiles(scene_id: str):
    with db.session() as s:
        rows = s.query(db.Tile).filter(db.Tile.scene_id == scene_id).all()
        return {"type": "FeatureCollection", "features": [{
            "type": "Feature", "geometry": t.polygon,
            "properties": {"point_id": t.point_id, "tile_id": t.tile_id, "quality": t.quality, **{
                k: round(v, 3) for k, v in (t.attrs or {}).items() if k in
                ("water", "veg", "bare", "urban", "dark", "cloud", "texture")}}} for t in rows]}


# ------------------------------------------------------------------------ search
class Filters(BaseModel):
    scene_ids: Optional[list[str]] = None
    sensors: Optional[list[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_quality: Optional[float] = None
    bbox: Optional[list[float]] = None
    exclude_synthetic: bool = False


class SearchBody(BaseModel):
    text: Optional[str] = None
    tile_id: Optional[str] = None
    point_id: Optional[int] = None
    filters: Filters = Field(default_factory=Filters)
    limit: int = 40
    per_scene_cap: Optional[int] = None


def _decorate(hits: list[dict]) -> list[dict]:
    ids = [h["point_id"] for h in hits]
    with db.session() as s:
        rows = {t.point_id: t for t in s.query(db.Tile).filter(db.Tile.point_id.in_(ids)).all()} if ids else {}
        scenes = {sc.scene_id: sc for sc in s.query(db.Scene).all()}
    out = []
    for rank, h in enumerate(hits, 1):
        t = rows.get(h["point_id"])
        if not t:
            continue
        sc = scenes[t.scene_id]
        out.append({
            "rank": rank, "point_id": t.point_id, "tile_id": t.tile_id, "score": round(h["score"], 4),
            "similarity": round(h["similarity"], 4), "attribute_score": round(h["attribute_score"], 4),
            "matched": h["matched"], "image_url": furl(t.image_path), "polygon": t.polygon,
            "lon": t.lon, "lat": t.lat, "scene_id": t.scene_id, "scene_title": sc.title,
            "date": sc.acquisition_date, "sensor": sc.sensor, "quality": round(t.quality, 3),
            "attrs": {k: round(v, 3) for k, v in (t.attrs or {}).items()
                      if k in ("water", "veg", "bare", "urban", "dark", "cloud", "texture")},
            "synthetic": bool(sc.synthetic)})
    return out


def _run_search(text, vector, flt, limit, cap):
    f = {k: v for k, v in flt.items() if v not in (None, False, [])}
    res = search.hybrid_search(state["index"], state["embedder"], text=text, vector=vector, flt=f,
                               limit=limit * 3 if cap else limit)
    res["hits"] = search.diversify(res["hits"], per_scene_cap=cap)[:limit]
    return {"mode": res["mode"], "intent": res["intent"], "notes": res["notes"], "total": len(res["hits"]),
            "results": _decorate(res["hits"])}


@app.post("/api/search")
def do_search(body: SearchBody):
    vec = None
    flt = body.filters.model_dump()
    if body.tile_id or body.point_id:
        pid = body.point_id
        if pid is None:
            with db.session() as s:
                t = s.query(db.Tile).filter(db.Tile.tile_id == body.tile_id).first()
                if not t:
                    raise HTTPException(404, "unknown tile")
                pid = t.point_id
        vec = state["index"].get_vector(pid)
        if vec is None:
            raise HTTPException(404, "tile has no vector")
        flt["exclude_ids"] = [pid]
    if not body.text and vec is None:
        raise HTTPException(400, "provide text, tile_id or an image")
    return _run_search(body.text, vec, flt, body.limit, body.per_scene_cap)


@app.post("/api/search/upload")
async def search_upload(file: UploadFile = File(...), text: str = Form(""), limit: int = Form(40),
                        exclude_synthetic: bool = Form(False)):
    try:
        im = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not read image")
    vec = state["embedder"].embed_image(np.asarray(im))
    return _run_search(text or None, vec, {"exclude_synthetic": exclude_synthetic}, limit, None)


# ------------------------------------------------------------------------ sites / change
def _site_summary(s, site: db.Site) -> dict:
    ev = s.query(db.ChangeEvent).filter(db.ChangeEvent.site_id == site.site_id).all()
    sup = [e for e in ev if e.status == "supported"]
    first = min(sup, key=lambda e: e.first_supported_date) if sup else None
    return {"site_id": site.site_id, "name": site.name, "kind": site.kind, "scenario": site.scenario,
            "georeferenced": site.georeferenced, "lon": site.lon, "lat": site.lat, "gsd_m": site.gsd_m,
            "observations": len(site.observations),
            "date_range": [site.observations[0].date, site.observations[-1].date] if site.observations else None,
            "synthetic_epochs": sum(1 for o in site.observations if o.synthetic),
            "supported": len(sup), "suppressed": sum(1 for e in ev if e.status == "suppressed"),
            "first_change": ({"date": first.first_supported_date, "type": first.change_type,
                              "confidence": first.confidence, "id": first.id} if first else None),
            "expected": site.expected}


@app.get("/api/sites")
def sites():
    with db.session() as s:
        return [_site_summary(s, x) for x in s.query(db.Site).order_by(db.Site.site_id)]


@app.get("/api/sites/{site_id}")
def site_detail(site_id: str):
    with db.session() as s:
        site = s.get(db.Site, site_id)
        if not site:
            raise HTTPException(404, "unknown site")
        out = _site_summary(s, site)
        out["note"] = site.note
        out["truth_url"] = furl(site.truth_path)
        out["observations_detail"] = [{
            "id": o.id, "date": o.date, "sensor": o.sensor, "synthetic": bool(o.synthetic),
            "note": o.note, "image_url": furl(o.image_path), "quality": o.quality} for o in site.observations]
        out["events"] = [_event_brief(e) for e in s.query(db.ChangeEvent).filter(
            db.ChangeEvent.site_id == site_id).order_by(db.ChangeEvent.window_start, db.ChangeEvent.first_supported_date)]
        return out


def _review_state(s, event_id: str) -> dict | None:
    r = s.query(db.AnalystReview).filter(db.AnalystReview.change_event_id == event_id) \
        .order_by(db.AnalystReview.id.desc()).first()
    return {"decision": r.decision, "comment": r.comment, "analyst_id": r.analyst_id,
            "timestamp": r.timestamp.isoformat() + "Z"} if r else None


def _event_brief(e: db.ChangeEvent) -> dict:
    with db.session() as s:
        rv = _review_state(s, e.id)
    return {"id": e.id, "site_id": e.site_id, "type": e.change_type, "type_label": CHANGE_TYPES.get(e.change_type),
            "status": e.status, "confidence": e.confidence, "area_m2": e.area_m2, "area_frac": e.area_frac,
            "before_date": e.window_start, "after_date": e.first_supported_date or _after_date(e),
            "reasons": e.reasons or [], "review": rv,
            "thumb": furl((e.evidence or {}).get("overlay"))}


def _after_date(e: db.ChangeEvent) -> str | None:
    with db.session() as s:
        o = s.get(db.Observation, e.after_obs)
        return o.date if o else None


@app.get("/api/changes")
def changes(status: Optional[str] = None, type: Optional[str] = None, min_conf: float = 0.0,
            review: Optional[str] = None, site_id: Optional[str] = None):
    with db.session() as s:
        q = s.query(db.ChangeEvent)
        if status:
            q = q.filter(db.ChangeEvent.status == status)
        if type:
            q = q.filter(db.ChangeEvent.change_type == type)
        if site_id:
            q = q.filter(db.ChangeEvent.site_id == site_id)
        q = q.filter(db.ChangeEvent.confidence >= min_conf).order_by(db.ChangeEvent.confidence.desc())
        rows = q.all()
        names = {x.site_id: x.name for x in s.query(db.Site).all()}
    out = []
    for e in rows:
        b = _event_brief(e)
        b["site_name"] = names.get(e.site_id)
        if review == "pending" and b["review"]:
            continue
        if review in ("confirmed", "rejected", "flagged") and (not b["review"] or b["review"]["decision"] != review):
            continue
        out.append(b)
    return out


@app.get("/api/changes/export.geojson")
def export_geojson(status: str = "supported"):
    feats = []
    with db.session() as s:
        for e in s.query(db.ChangeEvent).filter(db.ChangeEvent.status == status):
            site = s.get(db.Site, e.site_id)
            rv = _review_state(s, e.id)
            geom = None
            if site and site.georeferenced and site.lon is not None:
                geom = {"type": "Point", "coordinates": [site.lon, site.lat]}
            feats.append({"type": "Feature", "geometry": geom, "properties": {
                "id": e.id, "site": e.site_id, "type": e.change_type, "confidence": e.confidence,
                "area_m2": e.area_m2, "after": e.first_supported_date, "before": e.window_start,
                "review": rv["decision"] if rv else None, "method": e.method,
                "pipeline": e.pipeline_version}})
    return JSONResponse({"type": "FeatureCollection", "features": feats},
                        headers={"Content-Disposition": "attachment; filename=geolens_changes.geojson"})


@app.get("/api/changes/{event_id}")
def change_detail(event_id: str):
    with db.session() as s:
        e = s.get(db.ChangeEvent, event_id)
        if not e:
            raise HTTPException(404, "unknown change event")
        site = s.get(db.Site, e.site_id)
        before = s.get(db.Observation, e.before_obs)
        after = s.get(db.Observation, e.after_obs)
        reviews = [{"decision": r.decision, "comment": r.comment, "analyst_id": r.analyst_id,
                    "timestamp": r.timestamp.isoformat() + "Z"} for r in
                   s.query(db.AnalystReview).filter(db.AnalystReview.change_event_id == event_id)
                   .order_by(db.AnalystReview.id)]
        ev = e.evidence or {}
        return {
            **{k: v for k, v in db.row_dict(e).items() if k not in ("evidence",)},
            "type_label": CHANGE_TYPES.get(e.change_type), "site_name": site.name, "gsd_m": site.gsd_m,
            "before": {"date": before.date, "sensor": before.sensor, "synthetic": bool(before.synthetic),
                       "note": before.note},
            "after": {"date": after.date, "sensor": after.sensor, "synthetic": bool(after.synthetic),
                      "note": after.note},
            "images": {k: furl(v) for k, v in ev.items()},
            "reviews": reviews,
            "site_note": site.note, "expected": site.expected}


class ReviewBody(BaseModel):
    decision: str
    comment: str = ""
    analyst_id: str = "analyst"


@app.post("/api/changes/{event_id}/review")
def review(event_id: str, body: ReviewBody):
    if body.decision not in ("confirmed", "rejected", "flagged"):
        raise HTTPException(400, "decision must be confirmed, rejected or flagged")
    with db.session() as s:
        if not s.get(db.ChangeEvent, event_id):
            raise HTTPException(404, "unknown change event")
        s.add(db.AnalystReview(change_event_id=event_id, decision=body.decision,
                               comment=body.comment, analyst_id=body.analyst_id))
    return {"ok": True}


class AnalyzeBody(BaseModel):
    conf_threshold: Optional[float] = None
    z_threshold: Optional[float] = None
    max_cloud: Optional[float] = None


@app.post("/api/sites/{site_id}/analyze")
def analyze(site_id: str, body: AnalyzeBody):
    sp = {k: v for k, v in {"conf_threshold": body.conf_threshold, "max_cloud": body.max_cloud}.items() if v is not None}
    pp = {"z_threshold": body.z_threshold} if body.z_threshold is not None else None
    try:
        return analysis.analyze_site(site_id, sp or None, pp)
    except KeyError:
        raise HTTPException(404, "unknown site")


@app.get("/api/sites/{site_id}/timeline")
def timeline(site_id: str):
    with db.session() as s:
        site = s.get(db.Site, site_id)
        if not site:
            raise HTTPException(404, "unknown site")
        return [{"date": o.date, "sensor": o.sensor, "synthetic": bool(o.synthetic), **(o.quality or {})}
                for o in site.observations]


# ------------------------------------------------------------------------ provenance / runs
def site_geo(s, site_id: str):
    site = s.get(db.Site, site_id)
    return bool(site and site.georeferenced)


@app.get("/api/export/{event_id}")
def export_case(event_id: str):
    """One analyst case as a self-contained JSON record: result, evidence links, provenance, decisions."""
    case = change_detail(event_id)
    case["provenance"] = provenance(event_id)
    return JSONResponse(case, headers={"Content-Disposition": f"attachment; filename=geolens_{event_id}.json"})


class SimilarBody(BaseModel):
    limit: int = 30


@app.post("/api/sites/{site_id}/similar")
def similar_sites(site_id: str, body: SimilarBody):
    with db.session() as s:
        site = s.get(db.Site, site_id)
        if not site:
            raise HTTPException(404, "unknown site")
        if not site.georeferenced or (site.gsd_m or 0) < 10:
            raise HTTPException(400, "This site is a 0.5 m aerial chip; the index holds 30 m Landsat tiles, "
                                     "so similarity between them would not mean anything.")
        obs = [o for o in site.observations if not o.synthetic] or list(site.observations)
        path, lon, lat = obs[-1].image_path, site.lon, site.lat
    rgb = np.asarray(Image.open(config.PROCESSED / path).convert("RGB"))
    t = config.TILE_SIZE
    if min(rgb.shape[:2]) >= t:      # match the scale of an indexed tile
        y, x = (rgb.shape[0] - t) // 2, (rgb.shape[1] - t) // 2
        rgb = rgb[y:y + t, x:x + t]
    vec = state["embedder"].embed_image(rgb)
    res = _run_search(None, vec, {}, body.limit + 12, None)
    res["results"] = [r for r in res["results"] if abs(r["lat"] - lat) > 0.03 or abs(r["lon"] - lon) > 0.03][:body.limit]
    res["total"] = len(res["results"])
    for i, r in enumerate(res["results"], 1):
        r["rank"] = i
    res["notes"] = ["tiles within about 3 km of the site were left out so you see other places"]
    res["mode"] = "site"
    return res


@app.get("/api/provenance/{event_id}")
def provenance(event_id: str):
    with db.session() as s:
        e = s.get(db.ChangeEvent, event_id)
        if not e:
            raise HTTPException(404, "unknown change event")
        before, after = s.get(db.Observation, e.before_obs), s.get(db.Observation, e.after_obs)
        scenes = {}
        for o in (before, after):
            sc = s.get(db.Scene, o.scene_id) if o.scene_id else None
            if sc:
                scenes[sc.scene_id] = {k: v for k, v in db.row_dict(sc).items() if k in (
                    "scene_id", "platform", "sensor", "acquisition_date", "crs", "resolution_m", "origin", "license", "source_path")}
        runs = [db.row_dict(r) for r in s.query(db.ProcessingRun).filter(
            db.ProcessingRun.scene_id == e.site_id).order_by(db.ProcessingRun.started.desc()).limit(3)]
        reviews = [{"decision": r.decision, "analyst_id": r.analyst_id, "comment": r.comment,
                    "timestamp": r.timestamp.isoformat() + "Z"} for r in
                   s.query(db.AnalystReview).filter(db.AnalystReview.change_event_id == event_id)]
        return {"event": event_id, "site": e.site_id, "method": e.method, "pipeline_version": e.pipeline_version,
                "params": e.params, "created_at": e.created_at.isoformat() + "Z",
                "inputs": [{"role": "before", "observation": before.id, "date": before.date,
                            "file": before.image_path, "synthetic": bool(before.synthetic), "note": before.note},
                           {"role": "after", "observation": after.id, "date": after.date, "file": after.image_path,
                            "synthetic": bool(after.synthetic), "note": after.note}],
                "registration": (e.transitions or {}).get("registration"),
                "normalisation": (e.transitions or {}).get("normalisation"),
                "confidence_factors": e.factors, "reasons": e.reasons, "runs": runs, "reviews": reviews,
                "source_scenes": list(scenes.values()), "embedder": state["embedder"].name,
                "georeferenced": bool(site_geo(s, e.site_id))}


# ------------------------------------------------------------------------ ingestion jobs
def _job_progress(job_id: str):
    def cb(msg: str, frac: float):
        JOBS[job_id].update(message=msg, progress=round(frac, 2))
    return cb


def _run_ingest(job_id: str, path: Path, date, sensor, platform, replace):
    JOBS[job_id].update(status="running")
    try:
        res = ingest.ingest_geotiff(path, index=state["index"], embedder=state["embedder"], date=date,
                                    sensor=sensor or "unknown", platform=platform or "unknown",
                                    replace=replace, progress=_job_progress(job_id), run_id=job_id)
        JOBS[job_id].update(status="done", result=res, progress=1.0, message="indexed")
    except ingest.IngestError as ex:
        JOBS[job_id].update(status="failed", message=str(ex))
    except Exception as ex:  # noqa
        traceback.print_exc()
        JOBS[job_id].update(status="failed", message=f"{type(ex).__name__}: {ex}")


@app.post("/api/ingest")
async def ingest_upload(background: BackgroundTasks, file: UploadFile = File(...), date: str = Form(""),
                        sensor: str = Form(""), platform: str = Form(""), replace: bool = Form(False)):
    name = Path(file.filename or "upload.tif").name
    if not name.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "upload a GeoTIFF (.tif)")
    folder = config.INCOMING / uuid.uuid4().hex[:8]     # own folder keeps the file name (and scene id) stable
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    dest.write_bytes(await file.read())
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    JOBS[job_id] = {"id": job_id, "file": name, "status": "queued", "progress": 0.0, "message": "queued"}
    background.add_task(_run_ingest, job_id, dest, date or None, sensor, platform, replace)
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]


@app.get("/api/runs")
def runs(limit: int = 20):
    with db.session() as s:
        return [db.row_dict(r) for r in s.query(db.ProcessingRun).order_by(db.ProcessingRun.started.desc()).limit(limit)]


# ------------------------------------------------------------------------ evaluation
@app.get("/api/evaluation")
def evaluation():
    p = config.EVAL_DIR / "metrics.json"
    if not p.exists():
        return {"available": False}
    return {"available": True, **json.loads(p.read_text())}


@app.get("/api/health")
def health():
    return {"ok": True}


# ------------------------------------------------------------------------ front end
if FRONTEND.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND / "assets")), name="assets") if (FRONTEND / "assets").exists() else None

    @app.get("/")
    def index_page():
        return FileResponse(FRONTEND / "index.html")

    @app.get("/{name}.{ext}")
    def static_root(name: str, ext: str):
        p = FRONTEND / f"{name}.{ext}"
        if p.exists() and p.parent == FRONTEND:
            return FileResponse(p)
        raise HTTPException(404)
