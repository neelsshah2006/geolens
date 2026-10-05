#!/usr/bin/env python3
"""Fetch the public datasets GeoLens ships with.

Sources (all public):
  * Landsat 8 OLI Level-1 scenes 224077 / 224078, acquired 2020-05-18
    (USGS / NASA, public domain). Distributed as sample rasters inside the
    `geowombat` PyPI source distribution, so they can be pulled through PyPI.
  * LEVIR-CD sample pairs (Chen & Shi, 2020) from the BIT_CD repository on
    GitHub. Real before/after image pairs with hand-drawn change masks.

Run once with internet access; afterwards everything works offline.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

LANDSAT_FILES = [
    "LC08_L1TP_224077_20200518_20200518_01_RT_B2.TIF",
    "LC08_L1TP_224077_20200518_20200518_01_RT_B3.TIF",
    "LC08_L1TP_224077_20200518_20200518_01_RT_B4.TIF",
    "LC08_L1TP_224078_20200518_20200518_01_RT_B2.TIF",
    "LC08_L1TP_224078_20200518_20200518_01_RT_B3.TIF",
    "LC08_L1TP_224078_20200518_20200518_01_RT_B4.TIF",
]

LEVIR_BASE = "https://raw.githubusercontent.com/justchenhao/BIT_CD/master/samples"
LEVIR_PAIRS = [
    "test_2_0000_0000.png",
    "test_2_0000_0512.png",
    "test_7_0256_0512.png",
]


def fetch_landsat() -> None:
    dest = RAW / "landsat"
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / f).exists() for f in LANDSAT_FILES):
        print("landsat: already present")
        return
    with tempfile.TemporaryDirectory() as tmp:
        print("landsat: downloading geowombat sdist from PyPI (~70 MB)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "download", "geowombat", "--no-deps",
             "--no-binary", ":all:", "-d", tmp, "-q"]
        )
        archive = next(Path(tmp).glob("geowombat-*.tar.gz"))
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                name = Path(member.name).name
                if name in LANDSAT_FILES and "/data/" in member.name:
                    print("  extract", name)
                    src = tf.extractfile(member)
                    assert src is not None
                    with open(dest / name, "wb") as out:
                        shutil.copyfileobj(src, out)


def fetch_levir() -> None:
    for sub in ("A", "B", "label"):
        (RAW / "levir" / sub).mkdir(parents=True, exist_ok=True)
    for name in LEVIR_PAIRS:
        for sub in ("A", "B", "label"):
            target = RAW / "levir" / sub / name
            if target.exists():
                continue
            url = f"{LEVIR_BASE}/{sub}/{name}"
            print("levir:", url)
            with urllib.request.urlopen(url, timeout=60) as r:
                target.write_bytes(r.read())


if __name__ == "__main__":
    fetch_landsat()
    fetch_levir()
    print("done")
