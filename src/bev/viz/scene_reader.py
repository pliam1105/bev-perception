"""Per-scene nuScenes reader for visualization use.

Walks one scene's keyframes in chronological order and yields raw per-sample
data: image paths, lidar point arrays, per-sensor calibration (sensor->ego,
ego->global, camera intrinsic), and 3D annotation boxes in the global frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from nuscenes.nuscenes import NuScenes


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
    """Raw calibration for one sample_data record. Quaternions are wxyz."""

    sensor2ego_translation: np.ndarray  # (3,) float64
    sensor2ego_rotation: np.ndarray  # (4,) float64, wxyz
    ego2global_translation: np.ndarray  # (3,) float64
    ego2global_rotation: np.ndarray  # (4,) float64, wxyz
    intrinsic: np.ndarray | None  # (3, 3) float64 for cameras, None for lidar
    timestamp_us: int


@dataclass(frozen=True)
class CameraFrame:
    channel: str
    image_path: Path  # on-disk JPEG, decoded lazily by the caller
    width: int
    height: int
    calib: SensorCalib


@dataclass(frozen=True)
class LidarFrame:
    points: np.ndarray  # (N, 5) float32: x, y, z, intensity, ring
    calib: SensorCalib


@dataclass(frozen=True)
class Box3D:
    """One sample_annotation, kept in the global frame exactly as nuScenes stores it."""

    token: str
    category: str
    center: np.ndarray  # (3,) float64 in the global frame
    wlh: np.ndarray  # (3,) float64: width, length, height (nuScenes convention)
    rotation: np.ndarray  # (4,) float64 wxyz in the global frame
    num_lidar_pts: int
    num_radar_pts: int
    visibility: str


@dataclass(frozen=True)
class SceneSample:
    sample_token: str
    scene_name: str
    timestamp_us: int
    cameras: dict[str, CameraFrame]
    lidar: LidarFrame | None
    boxes: list[Box3D]


class NuScenesSceneReader:
    """Iterate one nuScenes scene's keyframes in chronological order.

    Loads the devkit index once, then walks a scene's linked-list of samples
    on demand.
    """

    def __init__(
        self,
        dataroot: Path | str,
        version: str = "v1.0-mini",
        *,
        cameras: Sequence[str] = CAMERAS,
        include_lidar: bool = True,
    ) -> None:
        self.dataroot = Path(dataroot)
        if not self.dataroot.exists():
            raise FileNotFoundError(f"dataroot not found: {self.dataroot}")
        self.version = version
        self.cameras = tuple(cameras)
        self.include_lidar = include_lidar
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=False)

    def list_scenes(self) -> list[tuple[str, str]]:
        """Return `(token, name)` for every scene in this split."""
        return [(s["token"], s["name"]) for s in self.nusc.scene]

    def read_scene(self, scene: str) -> Iterator[SceneSample]:
        """Yield SceneSample objects, walking `first_sample_token -> next -> ...`.

        `scene` accepts either a scene name (e.g. "scene-0061") or a scene token.
        """
        rec = self._resolve_scene(scene)
        sample_token = rec["first_sample_token"]
        while sample_token:
            sample = self.nusc.get("sample", sample_token)
            yield self._build_sample(sample, rec["name"])
            sample_token = sample["next"]

    def _resolve_scene(self, scene: str) -> dict:
        for s in self.nusc.scene:
            if s["name"] == scene or s["token"] == scene:
                return s
        raise KeyError(f"scene not found: {scene!r}")

    def _build_sample(self, sample: dict, scene_name: str) -> SceneSample:
        cams: dict[str, CameraFrame] = {}
        for channel in self.cameras:
            sd_token = sample["data"].get(channel)
            if sd_token is None:
                continue
            cams[channel] = self._camera_frame(channel, sd_token)

        lidar: LidarFrame | None = None
        if self.include_lidar and LIDAR in sample["data"]:
            lidar = self._lidar_frame(sample["data"][LIDAR])

        boxes = [self._box(ann_token) for ann_token in sample["anns"]]

        return SceneSample(
            sample_token=sample["token"],
            scene_name=scene_name,
            timestamp_us=sample["timestamp"],
            cameras=cams,
            lidar=lidar,
            boxes=boxes,
        )

    def _calib_for(self, sd_token: str) -> tuple[SensorCalib, dict]:
        sd = self.nusc.get("sample_data", sd_token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        intrinsic = None
        if cs.get("camera_intrinsic"):
            intrinsic = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
            if intrinsic.shape != (3, 3):
                intrinsic = None
        calib = SensorCalib(
            sensor2ego_translation=np.asarray(cs["translation"], dtype=np.float64),
            sensor2ego_rotation=np.asarray(cs["rotation"], dtype=np.float64),
            ego2global_translation=np.asarray(ep["translation"], dtype=np.float64),
            ego2global_rotation=np.asarray(ep["rotation"], dtype=np.float64),
            intrinsic=intrinsic,
            timestamp_us=sd["timestamp"],
        )
        return calib, sd

    def _camera_frame(self, channel: str, sd_token: str) -> CameraFrame:
        calib, sd = self._calib_for(sd_token)
        return CameraFrame(
            channel=channel,
            image_path=self.dataroot / sd["filename"],
            width=int(sd["width"]),
            height=int(sd["height"]),
            calib=calib,
        )

    def _lidar_frame(self, sd_token: str) -> LidarFrame:
        calib, sd = self._calib_for(sd_token)
        # nuScenes .pcd.bin: contiguous float32, five columns (x, y, z, intensity, ring).
        path = self.dataroot / sd["filename"]
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
        return LidarFrame(points=pts, calib=calib)

    def _box(self, ann_token: str) -> Box3D:
        a = self.nusc.get("sample_annotation", ann_token)
        return Box3D(
            token=ann_token,
            category=a["category_name"],
            center=np.asarray(a["translation"], dtype=np.float64),
            wlh=np.asarray(a["size"], dtype=np.float64),
            rotation=np.asarray(a["rotation"], dtype=np.float64),
            num_lidar_pts=int(a.get("num_lidar_pts", 0)),
            num_radar_pts=int(a.get("num_radar_pts", 0)),
            visibility=str(a.get("visibility_token", "")),
        )
