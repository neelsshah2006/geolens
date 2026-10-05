# GeoLens

An offline workbench for searching satellite imagery by what it looks like and for finding, dating and
reviewing changes between acquisitions. Built for SIH 2026 problem statement SIH26227.

Everything runs on one machine. The map library, fonts, imagery, vector index and database are all local, and the
front end makes no request to any other host (checked by blocking all external traffic in a browser and running the
full demo flow).

## Run it

```bash
pip install -r requirements.txt
./run.sh                      # Windows: run.bat
# open http://localhost:8000
```

The zip ships with the demo already built (index, database, tiles, overlays, evidence images), so it runs as is.
To keep the download small it leaves out the raw Landsat bands and the cloud-optimised copies; neither is needed to
run the app. To rebuild from scratch (about 40 s on 2 CPU cores once the data is present):

```bash
python scripts/fetch_data.py  # downloads the Landsat bands (about 70 MB via PyPI); needs internet once
python scripts/build_demo.py --clean
python -m pytest backend/tests
```

## What you can do in it

| Screen | What it does |
| --- | --- |
| Search | Describe a surface in words, or supply a reference image, or start from a tile or a site. Ranked tiles appear as a contact sheet and on the map. Filters: date range, minimum image quality, scene. |
| Changes | Ranked change cases. Each shows an observation timeline (including passes that were left out and why), a before/after curtain, a plain-language explanation, a confidence breakdown, review actions, "Find similar locations", "Export case" and the full processing record. |
| Imagery | Upload a GeoTIFF. It is validated, tiled, quality-scored, embedded and inserted into the live index without rebuilding anything. Lists indexed scenes and recent runs. |
| Evaluation | The measured numbers, each next to the reason it should not be over-trusted. |

Keyboard: `j` / `k` move between change cases; arrow keys move the before/after curtain when it has focus.

## Data

| Data | Source | Used for |
| --- | --- | --- |
| Landsat 8 OLI scenes, path 224 rows 077 and 078, 2020-05-18, bands 4/3/2 at 30 m (Paraná reservoir and Ciudad del Este, about 6,150 km²) | USGS/NASA, public domain. Pulled through the `geowombat` package on PyPI because that was the only reachable host serving real scenes. | The searchable archive (337 tiles). |
| LEVIR-CD sample pairs (3 before/after pairs with hand-drawn change masks, 0.5 m) | Chen & Shi, 2020, from the `justchenhao/BIT_CD` GitHub repository | Real change detection and accuracy measurement. Check the dataset's terms before redistributing outside the competition. |
| MapLibre GL JS, IBM Plex Sans/Mono, Source Serif 4 | npm; BSD-3 and SIL OFL | Front end, vendored in `frontend/assets`. |

### What is real and what is synthetic

No public multi-date archive with known outcomes was reachable, so **time is partly synthetic**:

* Every pixel comes from a real image.
* The two real LEVIR epochs per chip are genuine. Their capture dates are not published per chip (the dataset spans
  2002 to 2018), so the dates shown (2008 and 2018) are **nominal**.
* Extra epochs (cloudy, hazy, shifted, drifted, dry-season, "urban patch added") are derived from real images to
  test the false-alarm logic. They carry a `synthetic` flag in the database, a dashed marker on the timeline and the
  word "synthetic" in the viewer.
* The LEVIR chips have no coordinates, so they are not on the map.
* The two Landsat scenes were acquired on the same day. The Landsat sites' later dates are synthetic.

## How it works

```
GeoTIFF -> validate CRS -> COG + web overlay -> 128 px tiles -> quality (cloud, shadow, haze)
        -> descriptor -> frozen scaler -> unit vector -> Qdrant (local) + SQLite metadata

Search   text -> controlled vocabulary -> attribute constraints + prototype vector -> top 120 by cosine
         -> re-rank by attribute match and image quality -> top K
Change   register (phase correlation, verified) -> mask cloud/shadow/haze -> robust radiometric match
         -> change score (30 m) or built-surface comparison (0.5 m) -> regions -> rule classifier
         -> persistence in later passes -> seven-factor confidence -> earliest supported date
```

Confidence is a weighted geometric mean of seven factors (strength of change, usable imagery, alignment, persistence,
season match, sensor match, type-rule fit), and is capped at 0.35 if quality, alignment, persistence or season
collapses. All factors are stored and shown. The reporting threshold is 0.55.

## Measured results

From `data/evaluation/metrics.json` (also shown in the Evaluation screen):

| Check | Result | Limit of the number |
| --- | --- | --- |
| Temporal scenarios (cloud then real change, misregistration then change, seasonal only, shift only, urban growth) | 8 of 8 correct | Thresholds were tuned until these passed and several epochs are synthetic. It is a regression suite, not field accuracy. |
| Pixel change detection on real LEVIR-CD pairs | F1 0.72, IoU 0.56, precision 0.66, recall 0.79 | Tuned on those same 3 pairs, so optimistic. About 9.4% of unchanged pixels were flagged. |
| Distorted copy of a tile finds its original (top 10) | 97 to 100% | Measures robustness, not semantic quality. |
| Neighbours share the tile's dominant surface class (top 10) | 0.78 vs 0.51 chance | Labels come from the same colour rules that feed the descriptor, so this is optimistic. |
| Search latency | 0.7 ms median over 337 tiles | Tiny index. |
| Adding one scene | about 7 s (40 ms per tile) on 2 CPU cores | No GPU used. |

There is **no human-judged retrieval benchmark**, so precision@K, nDCG and MRR as listed in the design document were
not measured. There is no held-out split either: the data available offline was too small to hold anything out.

## Against the design document

| Design document item | Status |
| --- | --- |
| Text to image search, image to image search | Done, with a caveat: text search uses a fixed vocabulary of surface terms (water, vegetation, bare soil, built-up, road, cloud, texture...), not open language. "Near" means "in the same tile". Time words are reported as unused. |
| Pretrained remote-sensing / multimodal encoder | **Not done.** Model hosts were unreachable, so the embedder is a hand-built descriptor (colour, spectral relations, texture, layout). It is one small module (`embedding.py`) behind `embed_tile`/`embed_image`; see below to replace it. |
| Two-stage retrieval with reranking | Done (vector top 120, then attribute and quality re-rank). |
| Registration, quality masks, normalisation | Done. Translation only; no rotation or warping. Cloud and haze masks are rule-based, not a trained model. |
| Change detection and classification | Done for construction, clearance, water expansion/reduction, road, other. Rule-based, using only RGB. Weak on subtle changes. |
| Earliest supported change with evidence | Done. |
| False-alarm suppression | Done for cloud, haze, translation, radiometric drift and a seasonal vegetation shift. Not done for snow, sun-angle or viewing-geometry differences (no metadata for them). |
| Similar-site discovery | Nearest-neighbour from a tile, an image, or a change case (Landsat sites only). Clustering was optional and is not built. |
| Analyst review, audit trail, export | Done. Export is JSON per case and GeoJSON for all supported changes. |
| Provenance | Done: scenes, dates, sensor, CRS, source files, pipeline version, embedder, parameters, registration, normalisation, factors, decisions. |
| Incremental ingestion | Done (tested through the API: valid file, no georeferencing, wrong type, duplicate, replace, delete). Single files only; new-file watching is not built. |
| Offline | Done: no external requests, map has no basemap. |
| Next.js, TypeScript, Tailwind | **Different:** plain JavaScript with no build step, so there is nothing to install or compile offline. |
| PostgreSQL + PostGIS | **Different:** SQLite. `GEOLENS_DB` accepts a PostgreSQL URL but that path is untested, and geometry is stored as GeoJSON text plus bounding-box columns rather than PostGIS types. |
| Celery + Redis | Not used; ingestion runs in a background thread. Fine for one user, not for a team. |
| Docker Compose | Files provided but **never run** (no Docker in the build environment). |
| PyTorch, GeoPandas | Not used. |

## Replacing the embedder with a pretrained model

`backend/geolens/embedding.py` defines what the rest of the code needs: `raw(rgb, valid)`, `fit`, `transform`,
`embed_tile(rgb, valid)`, `embed_image(rgb)` and `name`. Implement those with your encoder (return unit-length
vectors), set `EMBEDDER_NAME` in `config.py`, and run `python scripts/build_demo.py --clean`. Text-to-image search then
needs the encoder's text tower in `search.py` in place of the vocabulary prototype. Record the model name, version and
licence in `docs/` as the problem statement requires.

## API

`GET /api/overview` `GET /api/scenes` `DELETE /api/scenes/{id}` `POST /api/search` `POST /api/search/upload`
`GET /api/sites` `GET /api/sites/{id}` `POST /api/sites/{id}/analyze` `POST /api/sites/{id}/similar`
`GET /api/changes` `GET /api/changes/{id}` `POST /api/changes/{id}/review` `GET /api/changes/export.geojson`
`GET /api/export/{id}` `GET /api/provenance/{id}` `POST /api/ingest` `GET /api/jobs/{id}` `GET /api/runs`
`GET /api/evaluation`. Interactive docs at `/docs`.

## Layout

```
backend/geolens/   imaging, change, temporal, analysis, embedding, index, search, ingest, scenarios, evaluate, db, main
backend/tests/     checks for registration, change, quality, query parsing, confidence
frontend/          index.html, app.css, app.js, assets/ (map library and fonts)
scripts/           fetch_data.py, build_demo.py
data/              raw/, processed/, evaluation/, qdrant/, geolens.db
```

## Known limits

* The descriptor and rule classifier are hand-built; expect plausible but not trained-model quality.
* Change classification at 30 m using RGB alone cannot separate many land-cover types.
* Registration corrects translation only.
* Thresholds were tuned on three real pairs and eight scenarios. Measure on the full LEVIR-CD test set and on
  organiser data before trusting any figure.
