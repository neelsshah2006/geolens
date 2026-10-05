"""Descriptor embedder: turns an RGB tile into a fixed-length unit vector.

This is the default, fully self-contained embedder (no network, no pretrained weights). The
interface is intentionally tiny so a pretrained remote-sensing encoder can replace it: implement
`embed_tile` and `name`, and re-index.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config, imaging

SCALER_PATH = config.MODELS / "descriptor_scaler.json"


class DescriptorEmbedder:
    name = config.EMBEDDER_NAME

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self._load()

    # ---------------------------------------------------------------- scaler (frozen after build)
    def _load(self) -> None:
        if SCALER_PATH.exists():
            d = json.loads(SCALER_PATH.read_text())
            self.mean = np.asarray(d["mean"], np.float32)
            self.std = np.asarray(d["std"], np.float32)

    @property
    def fitted(self) -> bool:
        return self.mean is not None

    def fit(self, raw: np.ndarray) -> None:
        self.mean = raw.mean(0).astype(np.float32)
        self.std = np.maximum(raw.std(0), 1e-3).astype(np.float32)
        SCALER_PATH.write_text(json.dumps({
            "embedder": self.name, "n_fit": int(raw.shape[0]),
            "mean": self.mean.tolist(), "std": self.std.tolist()}))

    def transform(self, raw: np.ndarray) -> np.ndarray:
        assert self.fitted, "embedder scaler has not been fitted"
        z = np.clip((raw - self.mean) / self.std, -4, 4)
        n = np.linalg.norm(z, axis=-1, keepdims=True)
        return (z / np.maximum(n, 1e-6)).astype(np.float32)

    # ---------------------------------------------------------------- tile -> raw descriptor
    @staticmethod
    def raw(rgb: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        return imaging.descriptor(rgb, valid)

    def embed_tile(self, rgb: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        return self.transform(self.raw(rgb, valid)[None])[0]

    def embed_image(self, rgb: np.ndarray) -> np.ndarray:
        """Embed an arbitrary RGB image (e.g. an uploaded reference) by resizing to tile size."""
        from PIL import Image
        im = Image.fromarray(rgb).convert("RGB").resize((config.TILE_SIZE, config.TILE_SIZE),
                                                         Image.BILINEAR)
        return self.embed_tile(np.asarray(im))
