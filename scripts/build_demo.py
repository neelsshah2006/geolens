#!/usr/bin/env python3
"""One-shot build: ingest Landsat scenes, build demo sites, run change analysis, run evaluation.

    python scripts/fetch_data.py     # once, needs internet
    python scripts/build_demo.py     # offline
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from geolens import analysis, config, db, evaluate, ingest, scenarios  # noqa: E402
from geolens.embedding import DescriptorEmbedder  # noqa: E402
from geolens.index import VectorIndex  # noqa: E402


def main() -> None:
    t0 = time.time()
    if "--clean" in sys.argv:
        for p in (config.PROCESSED, config.DATA / "qdrant", config.EVAL_DIR):
            shutil.rmtree(p, ignore_errors=True)
        for f in config.DATA.glob("geolens.db*"):
            f.unlink()
        (config.MODELS / "descriptor_scaler.json").unlink(missing_ok=True)
        config.PROCESSED.mkdir(parents=True, exist_ok=True)
        config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    index, emb = VectorIndex(), DescriptorEmbedder()
    land = config.RAW / "landsat"
    sets = sorted({p.name.split("_")[2] for p in land.glob("LC08_*_B4.TIF")})
    if not sets:
        sys.exit("No Landsat bands found in data/raw/landsat. Run scripts/fetch_data.py first.")
    for pr in sets:
        stem = next(land.glob(f"LC08_*_{pr}_*_B4.TIF")).name[:-7]
        bands = {"R": land / f"{stem}_B4.TIF", "G": land / f"{stem}_B3.TIF", "B": land / f"{stem}_B2.TIF"}
        print(f"[ingest] {stem}")
        print("        ", ingest.ingest_landsat_bands(bands, index=index, embedder=emb, replace=True))
    print("[sites]  ", scenarios.build_all())
    with db.session() as s:
        ids = [x.site_id for x in s.query(db.Site).order_by(db.Site.site_id)]
    for sid in ids:
        r = analysis.analyze_site(sid)
        print(f"[change]  {sid:22s} events={len(r['events'])} first={r['first_supported']}")
    print("[eval]   ")
    m = evaluate.run(index, emb)
    print(json.dumps({k: m[k] for k in ("scenario_summary", "levir_pixel_mean", "retrieval")}, indent=2))
    index.close()
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
