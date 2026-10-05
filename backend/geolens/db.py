"""Relational store. SQLite by default; set GEOLENS_DB to a PostgreSQL URL to move to Postgres.

Geometry is kept as GeoJSON text plus numeric bounding-box columns so spatial filters work
without an extension. On PostgreSQL the same columns can be mirrored into PostGIS geometry.
"""
from __future__ import annotations

import datetime as dt
import json
from contextlib import contextmanager
from typing import Any

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String,
                        Text, create_engine, event)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from . import config

Base = declarative_base()


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Scene(Base):
    __tablename__ = "scenes"
    scene_id = Column(String, primary_key=True)
    title = Column(String)
    sensor = Column(String)
    platform = Column(String)
    acquisition_date = Column(String)               # ISO date
    crs = Column(String)
    resolution_m = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    bbox_wgs84 = Column(JSON)                       # [minx, miny, maxx, maxy]
    source_path = Column(String)
    cog_path = Column(String)
    overlay_path = Column(String)
    origin = Column(String)                         # dataset / provider
    license = Column(String)
    synthetic = Column(Boolean, default=False)
    georeferenced = Column(Boolean, default=True)
    parent_scene = Column(String)
    qa = Column(JSON)
    created_at = Column(DateTime, default=utcnow)


class Tile(Base):
    __tablename__ = "tiles"
    tile_id = Column(String, primary_key=True)
    scene_id = Column(String, ForeignKey("scenes.scene_id"), index=True)
    row = Column(Integer)
    col = Column(Integer)
    window = Column(JSON)                           # [col_off, row_off, w, h]
    polygon = Column(JSON)                          # GeoJSON geometry (WGS84)
    minx = Column(Float)
    miny = Column(Float)
    maxx = Column(Float)
    maxy = Column(Float)
    lon = Column(Float)
    lat = Column(Float)
    image_path = Column(String)
    attrs = Column(JSON)
    quality = Column(Float)
    point_id = Column(Integer, unique=True)         # vector id in the index
    cluster = Column(Integer)
    embedder = Column(String)


class Site(Base):
    """A location that has several observations over time."""
    __tablename__ = "sites"
    site_id = Column(String, primary_key=True)
    name = Column(String)
    kind = Column(String)                           # landsat-window | levir-chip
    georeferenced = Column(Boolean, default=True)
    polygon = Column(JSON)
    lon = Column(Float)
    lat = Column(Float)
    gsd_m = Column(Float)
    note = Column(Text)
    scenario = Column(String)                       # synthetic scenario label, if any
    truth_path = Column(String)                     # ground-truth mask, if any
    expected = Column(JSON)                         # known outcome for validation scenarios
    observations = relationship("Observation", back_populates="site", order_by="Observation.date")


class Observation(Base):
    __tablename__ = "observations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    site_id = Column(String, ForeignKey("sites.site_id"), index=True)
    scene_id = Column(String, ForeignKey("scenes.scene_id"))
    date = Column(String)
    image_path = Column(String)
    sensor = Column(String)
    synthetic = Column(Boolean, default=False)
    note = Column(String)
    quality = Column(JSON)
    site = relationship("Site", back_populates="observations")


class ChangeEvent(Base):
    __tablename__ = "change_events"
    id = Column(String, primary_key=True)           # GL-000123
    site_id = Column(String, ForeignKey("sites.site_id"), index=True)
    change_type = Column(String)
    status = Column(String)                         # supported | suppressed | no_change
    confidence = Column(Float)
    score = Column(Float)
    area_m2 = Column(Float)
    area_frac = Column(Float)
    first_supported_date = Column(String)
    window_start = Column(String)                   # last observation before the change
    before_obs = Column(Integer)
    after_obs = Column(Integer)
    factors = Column(JSON)
    reasons = Column(JSON)
    transitions = Column(JSON)
    regions = Column(JSON)
    evidence = Column(JSON)                         # paths to before/after/overlay
    method = Column(String)
    params = Column(JSON)
    pipeline_version = Column(String)
    created_at = Column(DateTime, default=utcnow)


class AnalystReview(Base):
    __tablename__ = "analyst_reviews"
    id = Column(Integer, primary_key=True, autoincrement=True)
    change_event_id = Column(String, ForeignKey("change_events.id"), index=True)
    decision = Column(String)                       # confirmed | rejected | flagged
    comment = Column(Text)
    analyst_id = Column(String)
    timestamp = Column(DateTime, default=utcnow)


class ProcessingRun(Base):
    __tablename__ = "processing_runs"
    id = Column(String, primary_key=True)
    kind = Column(String)                           # ingest | change | index
    scene_id = Column(String)
    pipeline_version = Column(String)
    model_version = Column(String)
    params = Column(JSON)
    status = Column(String)                         # queued | running | done | failed
    message = Column(Text)
    stats = Column(JSON)
    started = Column(DateTime, default=utcnow)
    finished = Column(DateTime)


_engine = create_engine(config.DB_URL, connect_args={"check_same_thread": False}
                        if config.DB_URL.startswith("sqlite") else {})

if config.DB_URL.startswith("sqlite"):
    @event.listens_for(_engine, "connect")
    def _pragma(conn, _):  # pragma: no cover
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(_engine)


@contextmanager
def session():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def row_dict(obj: Any, exclude: tuple[str, ...] = ()) -> dict:
    out = {}
    for c in obj.__table__.columns:
        if c.name in exclude:
            continue
        v = getattr(obj, c.name)
        if isinstance(v, dt.datetime):
            v = v.isoformat() + "Z"
        out[c.name] = v
    return out
