"""Query handling: text -> attribute constraints, example/upload -> embedding, then hybrid ranking.

Honest scope: with no pretrained image-text model available offline, text queries are *not*
open-vocabulary. A small controlled vocabulary maps words to interpretable tile attributes
(water, vegetation, built-up, bare/farmland, texture...). The ranking combines
  1. vector similarity to a prototype built from the tiles that best satisfy the attributes, and
  2. the attribute-satisfaction score itself,
so results are explainable (each hit says which attributes matched).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .embedding import DescriptorEmbedder
from .index import VectorIndex

# word -> (attribute, direction, weight, label)
VOCAB: list[tuple[list[str], str, str, float, str]] = [
    (["water", "lake", "reservoir", "river", "sea", "lagoon", "pond", "coast", "shore", "waterbody",
      "waterbodies", "flood", "flooded", "stream", "creek", "canal", "waterway", "pond", "bay", "estuary", "wetland"], "water", "high", 1.0, "water"),
    (["forest", "forests", "woodland", "vegetation", "vegetated", "green", "trees", "tree", "crop",
      "crops", "cropland", "agricultural", "agriculture", "farmland", "fields", "field"], "veg", "high", 1.0, "vegetation"),
    (["bare", "soil", "fallow", "ploughed", "plowed", "harvested", "barren", "dry"], "bare", "high", 1.0, "bare soil"),
    (["urban", "city", "town", "built", "built-up", "buildings", "building", "settlement", "settlements",
      "residential", "housing", "houses", "village", "villages", "construction", "developed", "suburb",
      "structure", "structures", "infrastructure"],
     "urban", "high", 1.0, "built-up"),
    (["road", "roads", "highway", "street", "streets", "runway"], "texture", "high", 0.35, "linear detail"),
    (["dark", "shadow", "shadowed"], "dark", "high", 0.8, "dark surfaces"),
    (["cloud", "clouds", "cloudy"], "cloud", "high", 1.0, "cloud"),
    (["smooth", "uniform", "homogeneous", "plain", "flat"], "texture", "low", 0.6, "smooth texture"),
    (["textured", "complex", "heterogeneous", "mixed", "fragmented", "patchwork"], "texture", "high", 0.6, "textured"),
    (["bright", "light"], "brightness", "high", 0.5, "bright"),
]
TEMPORAL = {"new", "newly", "recent", "recently", "changed", "changes", "change", "latest", "emerging"}
NEGATIONS = {"no", "without", "not", "free", "excluding", "exclude", "minus"}
SCALES = {"water": 0.5, "veg": 0.6, "bare": 0.4, "urban": 0.5, "dark": 0.4, "cloud": 0.3,
          "texture": 0.14, "brightness": 0.5}


@dataclass
class Intent:
    wants: list[dict] = field(default_factory=list)       # {attr, direction, weight, label, negated}
    unknown: list[str] = field(default_factory=list)
    temporal: list[str] = field(default_factory=list)     # time words that tile search cannot use
    raw: str = ""

    def to_json(self) -> dict:
        return {"query": self.raw, "attributes": self.wants, "ignored_words": self.unknown, "temporal_words": self.temporal}


def parse_query(text: str) -> Intent:
    tokens = re.findall(r"[a-zA-Z\-]+", text.lower())
    intent = Intent(raw=text)
    neg = False
    seen: set[tuple[str, bool]] = set()
    stop = {"a", "an", "the", "of", "with", "and", "or", "in", "on", "near", "by", "area", "areas", "region",
            "show", "find", "tiles", "tile", "patch", "patches", "images", "image", "scene", "that", "have",
            "has", "large", "small", "some", "any", "around", "next", "to", "mostly", "mainly", "similar",
            "body", "bodies", "dense", "big", "wide", "very", "highly", "heavily", "open", "pure", "only", "all",
            "where", "which", "is", "are", "for", "from", "look", "looks", "like", "type", "cover", "land"}
    for tok in tokens:
        if tok in TEMPORAL:
            intent.temporal.append(tok)
            continue
        if tok in NEGATIONS:
            neg = True
            continue
        hit = False
        forms = {tok, tok[:-1] if tok.endswith("s") else tok, tok[:-2] if tok.endswith("es") else tok}
        for words, attr, direction, weight, label in VOCAB:
            if forms & set(words):
                key = (label, neg)
                if key not in seen:
                    seen.add(key)
                    intent.wants.append({"attr": attr, "direction": direction, "weight": weight,
                                         "label": label, "negated": neg, "word": tok})
                hit = True
                break
        if not hit and tok not in stop:
            intent.unknown.append(tok)
        if hit or tok in stop:
            neg = False if hit else neg
    return intent


def attr_score(payload: dict, intent: Intent) -> tuple[float, list[str]]:
    """0..1 satisfaction of the text constraints, plus human-readable matched attributes."""
    if not intent.wants:
        return 0.0, []
    num = den = 0.0
    matched: list[str] = []
    for w in intent.wants:
        v = float(payload.get(w["attr"], 0.0))
        scale = SCALES.get(w["attr"], 0.5)
        hi = w["direction"] == "high"
        if w["negated"]:
            hi = not hi
        s = float(np.clip(v / scale, 0, 1)) if hi else float(np.clip(1 - v / scale, 0, 1))
        num += w["weight"] * s
        den += w["weight"]
        if s >= 0.6:
            tag = ("no " if w["negated"] else "") + w["label"]
            if w["attr"] in ("water", "veg", "bare", "urban", "dark", "cloud") and not w["negated"]:
                tag += f" {v:.0%}"
            matched.append(tag)
    return num / den, matched


def prototype(index: VectorIndex, intent: Intent, payload_pool: list[dict], vecs: np.ndarray,
              top: int = 25) -> np.ndarray | None:
    if not intent.wants or len(payload_pool) == 0:
        return None
    scores = np.array([attr_score(p, intent)[0] * (0.5 + 0.5 * p.get("quality", 1.0)) for p in payload_pool])
    idx = np.argsort(-scores)[:top]
    if scores[idx[0]] < 0.4:
        return None
    c = vecs[idx].mean(0)
    n = np.linalg.norm(c)
    return (c / n) if n > 0 else None


def hybrid_search(index: VectorIndex, embedder: DescriptorEmbedder, *, text: str | None = None,
                  vector: np.ndarray | None = None, flt: dict | None = None, limit: int = 40,
                  exclude_cloud: bool = True) -> dict[str, Any]:
    flt = dict(flt or {})
    wants_cloud = bool(text) and any(w["attr"] == "cloud" and not w["negated"] for w in parse_query(text).wants)
    if exclude_cloud and not wants_cloud and flt.get("min_quality") is None:
        flt["min_quality"] = 0.5
    intent = parse_query(text) if text else Intent()
    pool_cache = None
    qvec = vector
    mode = "example" if vector is not None else "text"
    notes: list[str] = []
    if text and intent.wants:
        vecs, ids, payloads = index.all_vectors()
        if len(ids):
            proto = prototype(index, intent, payloads, vecs)
            if proto is not None:
                qvec = proto if qvec is None else (0.5 * qvec + 0.5 * proto)
                qvec = qvec / np.linalg.norm(qvec)
            else:
                notes.append("no indexed tile satisfies these attributes well; showing closest matches")
    if text and intent.temporal:
        notes.append("time words (" + ", ".join(intent.temporal) + ") do not affect tile search; "
                     "open Changes to see dated change events")
    if text and intent.unknown:
        notes.append("words outside the controlled vocabulary were ignored: " + ", ".join(intent.unknown))
    if text and not intent.wants and vector is None:
        notes.append("no recognised terms; try water, forest, farmland, urban, bare soil, road, cloud ...")
        return {"mode": mode, "intent": intent.to_json(), "hits": [], "notes": notes}
    if qvec is None:
        return {"mode": mode, "intent": intent.to_json(), "hits": [], "notes": notes}

    cand = index.search(qvec, flt, limit=max(limit * 4, 120))
    hits = []
    for c in cand:
        sim = c["score"]
        a, matched = attr_score(c["payload"], intent) if intent.wants else (0.0, [])
        if intent.wants and vector is None:
            score = 0.35 * sim + 0.65 * a
        elif intent.wants:
            score = 0.6 * sim + 0.4 * a
        else:
            score = sim
        score *= (0.85 + 0.15 * c["payload"].get("quality", 1.0))
        hits.append({"point_id": c["id"], "score": float(score), "similarity": float(sim),
                     "attribute_score": float(a), "matched": matched, "payload": c["payload"]})
    hits.sort(key=lambda h: -h["score"])
    return {"mode": mode, "intent": intent.to_json(), "hits": hits[:limit], "notes": notes}


def diversify(hits: list[dict], per_scene_cap: int | None = None, min_gap_tiles: int = 0) -> list[dict]:
    """Optional: stop one scene or one contiguous patch from filling the whole list."""
    if per_scene_cap is None and min_gap_tiles == 0:
        return hits
    out, counts = [], {}
    for h in hits:
        sid = h["payload"]["scene_id"]
        if per_scene_cap and counts.get(sid, 0) >= per_scene_cap:
            continue
        counts[sid] = counts.get(sid, 0) + 1
        out.append(h)
    return out
