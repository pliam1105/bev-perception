"""Shared data types and parsing helpers for nuScenes across bev.data and bev.viz.

Calibration is carried as raw numpy (framework-neutral metadata); quaternions are
unit wxyz. Boxes use the nuScenes ``Box`` (global frame) wrapped in ``Annotation``
so both the training loader and the visualizer speak one representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

if TYPE_CHECKING:
    import torch


CAMERAS: tuple[str, ...] = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
LIDAR: str = "LIDAR_TOP"


@dataclass(frozen=True)
class SensorCalib:
    """Raw calibration for one sample_data record. Quaternions are unit wxyz.

    Arrays are numpy by default; the training loader gets torch tensors via
    ``build_calib(..., to_torch=True)``. Both paths use this one class.
    """

    intrinsic: np.ndarray | "torch.Tensor" | None  # (3, 3) for cameras, None for lidar
    sensor2ego_translation: np.ndarray | "torch.Tensor"  # (3,)
    sensor2ego_rotation: np.ndarray | "torch.Tensor"  # (4,) wxyz
    ego2global_translation: np.ndarray | "torch.Tensor"  # (3,)
    ego2global_rotation: np.ndarray | "torch.Tensor"  # (4,) wxyz
    timestamp: int  # microseconds, this sensor's own capture time


@dataclass(frozen=True)
class Annotation:
    """A sample_annotation: a nuScenes global-frame Box plus its metadata."""

    box: Box  # nuscenes.utils.data_classes.Box, in the global frame
    instance_token: str
    num_lidar_pts: int
    num_radar_pts: int
    visibility: str


def _unit_wxyz(rotation: Sequence[float]) -> np.ndarray:
    q = np.asarray(rotation, dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / norm if norm else q


def build_calib(nusc: NuScenes, sd: dict, *, to_torch: bool = False) -> SensorCalib:
    """Assemble raw calibration for a sample_data record (passed as the dict).

    ``to_torch`` returns every array field (intrinsic and poses) as a torch
    tensor; torch is imported lazily so this module stays importable without it.
    """
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    intrinsic = None
    if cs.get("camera_intrinsic"):
        arr = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
        if arr.shape == (3, 3):
            intrinsic = arr
    sensor2ego_translation = np.asarray(cs["translation"], dtype=np.float64)
    sensor2ego_rotation = _unit_wxyz(cs["rotation"])
    ego2global_translation = np.asarray(ep["translation"], dtype=np.float64)
    ego2global_rotation = _unit_wxyz(ep["rotation"])

    if to_torch:
        import torch

        def to_t(a: np.ndarray | None) -> "torch.Tensor | None":
            return None if a is None else torch.from_numpy(np.ascontiguousarray(a))

        intrinsic = to_t(intrinsic)
        sensor2ego_translation = to_t(sensor2ego_translation)
        sensor2ego_rotation = to_t(sensor2ego_rotation)
        ego2global_translation = to_t(ego2global_translation)
        ego2global_rotation = to_t(ego2global_rotation)

    return SensorCalib(
        intrinsic=intrinsic,
        sensor2ego_translation=sensor2ego_translation,
        sensor2ego_rotation=sensor2ego_rotation,
        ego2global_translation=ego2global_translation,
        ego2global_rotation=ego2global_rotation,
        timestamp=int(sd["timestamp"]),
    )


def build_annotation(nusc: NuScenes, ann_token: str) -> Annotation:
    rec = nusc.get("sample_annotation", ann_token)
    return Annotation(
        box=nusc.get_box(ann_token),
        instance_token=rec["instance_token"],
        num_lidar_pts=int(rec["num_lidar_pts"]),
        num_radar_pts=int(rec["num_radar_pts"]),
        visibility=str(rec["visibility_token"]),
    )
