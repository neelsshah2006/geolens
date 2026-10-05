"""Fast checks on the parts that decide whether a change is believed. Run: pytest backend/tests

They use the real LEVIR-CD sample pairs in data/raw (run scripts/fetch_data.py first) and small
synthetic arrays; no server, database or index is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from geolens import change, imaging, scenarios, search  # noqa: E402

LEVIR = ROOT / "data" / "raw" / "levir"
needs_data = pytest.mark.skipif(not (LEVIR / "A").exists(), reason="run scripts/fetch_data.py first")


def _pair(name="test_2_0000_0000.png"):
    a = np.asarray(Image.open(LEVIR / "A" / name).convert("RGB"))
    b = np.asarray(Image.open(LEVIR / "B" / name).convert("RGB"))
    t = np.asarray(Image.open(LEVIR / "label" / name).convert("L")) > 127
    return a, b, t


@needs_data
def test_real_pair_detects_new_buildings():
    a, b, truth = _pair()
    res = change.analyze_pair(a, b, 0.5)
    tp = (res.mask & truth).sum()
    f1 = 2 * tp / (res.mask.sum() + truth.sum())
    assert f1 > 0.6
    assert any(r["type"] == "construction" for r in res.regions)


@needs_data
def test_registration_recovers_known_shift():
    a, *_ = _pair()
    moved = scenarios.shift_image(a, 4, -3)
    ones = np.ones(a.shape[:2], bool)
    reg = change.register(a, moved, ones, ones)["info"]
    assert abs(reg["shift_dy"] + 4) < 0.5 and abs(reg["shift_dx"] - 3) < 0.5
    assert reg["verified"]


@needs_data
def test_shifted_and_drifted_copy_is_not_change():
    a, *_ = _pair()
    other = scenarios.radiometric_drift(scenarios.shift_image(a, 3, 2))
    res = change.analyze_pair(a, other, 0.5)
    assert res.stats["changed_frac"] < 0.01


@needs_data
def test_cloud_cover_is_measured():
    a, *_ = _pair()
    q = imaging.assess_quality(scenarios.add_cloud(a, 0.5, seed=3))
    assert q["cloud_frac"] > 0.2
    assert imaging.assess_quality(a)["cloud_frac"] < 0.05


def test_text_queries_parse_negation_and_report_unknown_words():
    i = search.parse_query("forest without roads near the xylophone")
    labels = {(w["label"], w["negated"]) for w in i.wants}
    assert ("vegetation", False) in labels and ("linear detail", True) in labels
    assert "xylophone" in i.unknown


def test_attribute_score_orders_tiles():
    i = search.parse_query("open water")
    wet = {"water": 0.8, "texture": 0.02}
    dry = {"water": 0.0, "texture": 0.1}
    assert search.attr_score(wet, i)[0] > search.attr_score(dry, i)[0]


def test_confidence_is_capped_when_a_factor_collapses():
    good = dict(evidence=0.95, quality=1, registration=1, temporal=1, sensor=1, season=1, classification=0.9)
    assert change.combine_confidence(good) > 0.8
    bad = {**good, "season": 0.1}
    assert change.combine_confidence(bad) <= 0.35
