"""Runs the temporal analysis for a site and persists change events plus evidence imagery."""
from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
from PIL import Image

from . import change, config, db, temporal

EVID = config.PROCESSED / "evidence"


def _load_obs(site: db.Site) -> list[dict]:
    out = []
    for o in site.observations:
        rgb = np.asarray(Image.open(config.PROCESSED / o.image_path).convert("RGB"))
        valid = (rgb.sum(-1) > 0)
        out.append({"id": o.id, "date": o.date, "sensor": o.sensor or "unknown", "rgb": rgb,
                    "valid": valid, "synthetic": bool(o.synthetic), "note": o.note})
    return out


def _overlay(rgb: np.ndarray, mask: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    edge = mask & ~__import__("scipy.ndimage", fromlist=["x"]).binary_erosion(mask, iterations=1)
    tint = np.array([255, 90, 40], np.float32)
    out[mask] = out[mask] * 0.62 + tint * 0.38
    out[edge] = tint
    return out.clip(0, 255).astype(np.uint8)


def _next_event_id(s) -> str:
    n = s.query(db.ChangeEvent).count() + 1
    while s.get(db.ChangeEvent, f"GL-{n:05d}"):
        n += 1
    return f"GL-{n:05d}"


def analyze_site(site_id: str, params: dict | None = None, pair_params: dict | None = None) -> dict[str, Any]:
    t0 = time.time()
    with db.session() as s:
        site = s.get(db.Site, site_id)
        if site is None:
            raise KeyError(site_id)
        gsd = site.gsd_m or 1.0
        obs = _load_obs(site)
        georef = site.georeferenced
        site_poly = site.polygon
    if len(obs) < 2:
        raise ValueError("a site needs at least two observations")

    default_pair = {} if gsd < 2 else {"z_threshold": 3.2, "abs_min": 0.06}
    pp = {**default_pair, **(pair_params or {})}
    result = temporal.analyze_series(obs, gsd, params, pp)

    out_dir = EVID / site_id
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {o["id"]: o for o in result["observations"]}
    events: list[dict] = []
    first = result["first_supported"]

    with db.session() as s:
        for e in s.query(db.ChangeEvent).filter(db.ChangeEvent.site_id == site_id).all():
            for r in s.query(db.AnalystReview).filter(db.AnalystReview.change_event_id == e.id).all():
                s.delete(r)
            s.delete(e)
        s.flush()

        def store(rec: dict, status: str, res_obj) -> dict:
            eid = _next_event_id(s)
            a, b = by_id[rec["before_id"]], by_id[rec["after_id"]]
            res = res_obj if res_obj is not None else change.analyze_pair(
                a["rgb"], b["rgb"], gsd, a["valid"], b["valid"], pp)
            paths = {}
            for tag, img in (("before", a["rgb"]), ("after", res.aligned_after),
                             ("overlay", _overlay(res.aligned_after, res.mask, res.regions))):
                p = out_dir / f"{eid}_{tag}.png"
                Image.fromarray(img).save(p, optimize=True)
                paths[tag] = p.relative_to(config.PROCESSED).as_posix()
            m = out_dir / f"{eid}_mask.png"
            Image.fromarray((res.mask * 255).astype(np.uint8)).save(m)
            paths["mask"] = m.relative_to(config.PROCESSED).as_posix()
            regions = [{k: v for k, v in r.items() if k not in ("before", "after")}
                       | {"before": {k: round(v, 3) for k, v in r["before"].items()},
                          "after": {k: round(v, 3) for k, v in r["after"].items()}}
                       for r in res.regions]
            ev = db.ChangeEvent(
                id=eid, site_id=site_id, change_type=rec.get("type") or "other", status=status,
                confidence=rec["confidence"], score=rec.get("score"), area_m2=rec.get("area_m2"),
                area_frac=rec.get("area_frac"), first_supported_date=rec["after_date"] if status == "supported" else None,
                window_start=rec["before_date"], before_obs=rec["before_id"], after_obs=rec["after_id"],
                factors=rec.get("factors"), reasons=rec.get("reasons"),
                transitions={"persistence": rec.get("persistence"),
                             "registration": rec["registration"], "normalisation": rec["normalisation"],
                             "type_areas": rec.get("type_areas")},
                regions=regions, evidence=paths, method=config.CHANGE_METHOD,
                params={"series": result["params"], "pair": pp}, pipeline_version=config.PIPELINE_VERSION)
            s.add(ev)
            return {"id": eid, "status": status, "before_date": rec["before_date"],
                    "after_date": rec["after_date"], "confidence": rec["confidence"],
                    "type": rec.get("type")}

        for rec in result["transitions"]:
            if not rec["has_change"]:
                continue
            if rec["supported"]:
                status = "supported"
            else:
                status = "suppressed"
            events.append(store(rec, status, rec.get("_res")))
        for rec in result["rejected"]:
            events.append(store(rec, "suppressed", None))
        s.add(db.ProcessingRun(
            id=f"run-{uuid.uuid4().hex[:8]}", kind="change", scene_id=site_id,
            pipeline_version=config.PIPELINE_VERSION, model_version=config.CHANGE_METHOD,
            params={"series": result["params"], "pair": pp}, status="done", finished=db.utcnow(),
            stats={"events": len(events), "seconds": round(time.time() - t0, 2)}))
        site = s.get(db.Site, site_id)
        for row in result["timeline"]:
            o = s.get(db.Observation, row["obs_id"])
            o.quality = row

    return {"site_id": site_id, "events": events, "timeline": result["timeline"],
            "first_supported": ({"date": first["after_date"], "type": first["type"],
                                 "confidence": first["confidence"]} if first else None),
            "seconds": round(time.time() - t0, 2)}
