"""Offline BEV raster format, store, and runner shared by bev.data and bev.viz.

An offline job rasterizes each keyframe's BEV ground truth once and writes it to
disk keyed by sample_token; the training dataset and the visualizer then load
the same precomputed rasters. The rasterization itself (map + 3D-box -> grid) is
defined by a ``BEVRasterizer`` subclass.

On disk::

    <root>/meta.json           grid spec, layer names, dtype, version
    <root>/<sample_token>.npz  array "data" of shape (C, H, W)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch


FORMAT_VERSION = 1


@dataclass(frozen=True)
class BEVGridSpec:
    """Metric extent and resolution of the BEV grid (defaults: +-50 m at 0.5 m)."""

    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    resolution: float = 0.5

    @property
    def nx(self) -> int:
        return round((self.x_max - self.x_min) / self.resolution)

    @property
    def ny(self) -> int:
        return round((self.y_max - self.y_min) / self.resolution)

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BEVGridSpec":
        return cls(
            x_min=d["x_min"],
            x_max=d["x_max"],
            y_min=d["y_min"],
            y_max=d["y_max"],
            resolution=d["resolution"],
        )

    def in_bounds(self, pt: np.ndarray) -> bool:
        return self.x_min <= pt[0] <= self.x_max and self.y_min <= pt[1] <= self.y_max

    def idx_in_bounds(self, idx: tuple[int, int]) -> bool:
        return 0 <= idx[0] < self.nx and 0 <= idx[1] < self.ny

    def to_index(self, pt: np.ndarray) -> tuple[int, int]:
        return int((pt[0] - self.x_min) / self.resolution), int((pt[1] - self.y_min) / self.resolution)
    
    def ctr_from_index(self, idx: tuple[int, int]) -> np.ndarray:
        return np.array([(idx[0]+0.5) * self.resolution + self.x_min, (idx[1]+0.5) * self.resolution + self.y_min])


@dataclass(frozen=True)
class BEVRaster:
    """One sample's BEV raster: (C, H, W) over `layer_names`, numpy or torch."""

    data: np.ndarray | "torch.Tensor"  # (C, H, W)
    layer_names: tuple[str, ...]
    spec: BEVGridSpec
    sample_token: str

@dataclass
class BEVRasterBatch:
    data: torch.Tensor  # (N, C, H, W)
    layer_names: tuple[str, ...]
    spec: BEVGridSpec

    def to(self, device: str):
        self.data = self.data.to(device)
        return self


class BEVRasterStore:
    """Read/write precomputed BEV rasters under a directory, keyed by sample_token."""

    META = "meta.json"

    def __init__(
        self,
        root: Path | str,
        spec: BEVGridSpec,
        layer_names: Sequence[str],
        dtype: str = "float32",
    ) -> None:
        self.root = Path(root)
        self.spec = spec
        self.layer_names = tuple(layer_names)
        self.dtype = dtype

    @classmethod
    def open(cls, root: Path | str) -> "BEVRasterStore":
        """Open an existing store, reading its meta.json for spec + layers."""
        root = Path(root)
        meta_path = root / cls.META
        if not meta_path.is_file():
            raise FileNotFoundError(f"no BEV raster store at {root} (missing {cls.META})")
        meta = json.loads(meta_path.read_text())
        return cls(
            root,
            BEVGridSpec.from_dict(meta["spec"]),
            tuple(meta["layer_names"]),
            meta.get("dtype", "float32"),
        )

    def write_meta(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": FORMAT_VERSION,
            "spec": self.spec.to_dict(),
            "layer_names": list(self.layer_names),
            "dtype": self.dtype,
        }
        (self.root / self.META).write_text(json.dumps(meta, indent=2))

    def path_for(self, sample_token: str) -> Path:
        return self.root / f"{sample_token}.npz"

    def has(self, sample_token: str) -> bool:
        return self.path_for(sample_token).exists()

    def write(self, sample_token: str, data: np.ndarray) -> None:
        arr = np.asarray(data)
        expected = (len(self.layer_names), self.spec.ny, self.spec.nx)
        if arr.shape != expected:
            raise ValueError(
                f"raster for {sample_token} has shape {arr.shape}, expected {expected} "
                f"(C={len(self.layer_names)}, ny={self.spec.ny}, nx={self.spec.nx})"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path_for(sample_token), data=arr.astype(self.dtype, copy=False))

    def load(self, sample_token: str, *, to_torch: bool = False) -> BEVRaster:
        path = self.path_for(sample_token)
        if not path.is_file():
            raise FileNotFoundError(f"no raster for {sample_token!r} at {path}")
        with np.load(path) as npz:
            data = npz["data"]
        if to_torch:
            import torch

            data = torch.from_numpy(np.ascontiguousarray(data))
        return BEVRaster(
            data=data,
            layer_names=self.layer_names,
            spec=self.spec,
            sample_token=sample_token,
        )


class BEVRasterizer(ABC):
    """Turns one keyframe sample into a (C, H, W) BEV raster over `layer_names`.

    Subclasses implement `rasterize`: the BEV target generation, i.e. map
    rasterization and 3D-box projection into the grid.
    """

    def __init__(self, spec: BEVGridSpec, layer_names: Sequence[str]) -> None:
        self.spec = spec
        self.layer_names = tuple(layer_names)

    @abstractmethod
    def rasterize(self, sample: object) -> np.ndarray:
        """Return a (len(layer_names), spec.ny, spec.nx) array for one sample."""
        raise NotImplementedError


def build_rasters(
    rasterizer: BEVRasterizer,
    samples: Iterable[object],
    store: BEVRasterStore,
    *,
    overwrite: bool = False,
    token_of: Callable[[object], str] = lambda s: s.sample_token,
    progress: Callable[[int, str], None] | None = None,
) -> int:
    """Rasterize each sample and write it to the store; return the count written."""
    store.write_meta()
    written = 0
    for sample in samples:
        token = token_of(sample)
        if not overwrite and store.has(token):
            continue
        store.write(token, rasterizer.rasterize(sample))
        written += 1
        if progress is not None:
            progress(written, token)
    return written
