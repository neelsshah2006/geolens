"""Vector index over tile embeddings (Qdrant, embedded/local mode: no server, no network)."""
from __future__ import annotations

import datetime as dt
import threading
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from . import config

_lock = threading.RLock()


def date_ord(iso: str) -> int:
    return dt.date.fromisoformat(iso).toordinal()


class VectorIndex:
    def __init__(self, path: str | None = None, dim: int | None = None) -> None:
        self.path = path or config.QDRANT_PATH
        self.client = QdrantClient(path=self.path)
        self.dim = dim

    # ------------------------------------------------------------------ collection
    def exists(self) -> bool:
        return self.client.collection_exists(config.COLLECTION)

    def ensure(self, dim: int) -> None:
        with _lock:
            if not self.exists():
                self.client.create_collection(
                    config.COLLECTION,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
            self.dim = dim

    def reset(self) -> None:
        with _lock:
            if self.exists():
                self.client.delete_collection(config.COLLECTION)

    def count(self) -> int:
        if not self.exists():
            return 0
        return int(self.client.count(config.COLLECTION, exact=True).count)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ writes
    def upsert(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        with _lock:
            self.ensure(len(items[0]["vector"]))
            self.client.upsert(config.COLLECTION, points=[
                models.PointStruct(id=int(i["id"]), vector=[float(x) for x in i["vector"]],
                                   payload=i["payload"]) for i in items])

    def delete_scene(self, scene_id: str) -> None:
        if not self.exists():
            return
        with _lock:
            self.client.delete(config.COLLECTION, points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(
                    key="scene_id", match=models.MatchValue(value=scene_id))])))

    # ------------------------------------------------------------------ reads
    @staticmethod
    def build_filter(f: dict | None) -> models.Filter | None:
        if not f:
            return None
        must: list[Any] = []
        must_not: list[Any] = []
        if f.get("scene_ids"):
            must.append(models.FieldCondition(key="scene_id", match=models.MatchAny(any=f["scene_ids"])))
        if f.get("sensors"):
            must.append(models.FieldCondition(key="sensor", match=models.MatchAny(any=f["sensors"])))
        if f.get("min_quality") is not None:
            must.append(models.FieldCondition(key="quality", range=models.Range(gte=float(f["min_quality"]))))
        if f.get("date_from") or f.get("date_to"):
            must.append(models.FieldCondition(key="date_ord", range=models.Range(
                gte=date_ord(f["date_from"]) if f.get("date_from") else None,
                lte=date_ord(f["date_to"]) if f.get("date_to") else None)))
        if f.get("bbox"):
            x0, y0, x1, y1 = f["bbox"]
            must.append(models.FieldCondition(key="center", geo_bounding_box=models.GeoBoundingBox(
                top_left=models.GeoPoint(lon=x0, lat=y1), bottom_right=models.GeoPoint(lon=x1, lat=y0))))
        if f.get("polygon"):
            ring = [models.GeoPoint(lon=p[0], lat=p[1]) for p in f["polygon"]]
            must.append(models.FieldCondition(key="center", geo_polygon=models.GeoPolygon(
                exterior=models.GeoLineString(points=ring))))
        if f.get("exclude_ids"):
            must_not.append(models.HasIdCondition(has_id=[int(i) for i in f["exclude_ids"]]))
        if f.get("exclude_synthetic"):
            must_not.append(models.FieldCondition(key="synthetic", match=models.MatchValue(value=True)))
        return models.Filter(must=must or None, must_not=must_not or None)

    def search(self, vector: np.ndarray, flt: dict | None = None, limit: int = 100) -> list[dict]:
        if not self.exists():
            return []
        with _lock:
            res = self.client.query_points(
                config.COLLECTION, query=[float(x) for x in vector], limit=limit,
                query_filter=self.build_filter(flt), with_payload=True).points
        return [{"id": int(p.id), "score": float(p.score), "payload": p.payload} for p in res]

    def get_vector(self, point_id: int) -> np.ndarray | None:
        if not self.exists():
            return None
        with _lock:
            pts = self.client.retrieve(config.COLLECTION, ids=[int(point_id)], with_vectors=True)
        return np.asarray(pts[0].vector, np.float32) if pts else None

    def all_vectors(self) -> tuple[np.ndarray, list[int], list[dict]]:
        if not self.exists():
            return np.zeros((0, 0), np.float32), [], []
        ids: list[int] = []
        vecs: list[list[float]] = []
        payloads: list[dict] = []
        offset = None
        with _lock:
            while True:
                pts, offset = self.client.scroll(config.COLLECTION, limit=512, offset=offset,
                                                 with_payload=True, with_vectors=True)
                for p in pts:
                    ids.append(int(p.id))
                    vecs.append(p.vector)
                    payloads.append(p.payload)
                if offset is None:
                    break
        return np.asarray(vecs, np.float32), ids, payloads
