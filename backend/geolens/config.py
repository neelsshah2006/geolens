"""Central configuration. Everything is local; nothing here points at a network service."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("GEOLENS_HOME", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
INCOMING = DATA / "incoming"
EVAL_DIR = DATA / "evaluation"
MODELS = ROOT / "models"
STATIC = ROOT / "backend" / "static"

DB_URL = os.environ.get("GEOLENS_DB", f"sqlite:///{(DATA / 'geolens.db').as_posix()}")
QDRANT_PATH = os.environ.get("GEOLENS_QDRANT", str(DATA / "qdrant"))
COLLECTION = "tiles"

PIPELINE_VERSION = "geolens-pipeline-0.1.0"
EMBEDDER_NAME = "descriptor-v1"          # hand-built colour/texture/spectral descriptor
CHANGE_METHOD = "registered-normalised-diff-v1"

TILE_SIZE = 128                          # pixels
MIN_TILE_VALID = 0.70                    # tiles with less valid data than this are not indexed

# Fixed display stretch for Landsat 8 OLI L1 digital numbers (B4, B3, B2).
# One stretch for every scene keeps tiles from different scenes comparable.
LANDSAT8_STRETCH = {"R": (6000, 8700), "G": (6700, 8300), "B": (7400, 8650)}

for p in (DATA, RAW, PROCESSED, INCOMING, EVAL_DIR, MODELS):
    p.mkdir(parents=True, exist_ok=True)
