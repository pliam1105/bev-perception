"""Per-scene nuScenes reader for visualization use.

Two ways to walk a scene:

- ``read_scene`` yields keyframes (``sample``): all six cameras + lidar + boxes
  grouped at one instant, at the ~2 Hz annotated rate.
- ``read_sample_data`` yields the full ``sample_data`` stream (sweeps included)
  as individual per-sensor events in global timestamp order, at each sensor's
  own rate. Annotations exist only at keyframes, so boxes ride on the lidar
  keyframe events.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from nuscenes.nuscenes import NuScenes

from bev.types import (
    CAMERAS,
    LIDAR,
    Annotation,
    SensorCalib,
    build_annotation,
    build_calib,
)


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
class SceneSample:
    sample_token: str
    scene_name: str
    timestamp: int
    cameras: dict[str, CameraFrame]
    lidar: LidarFrame | None
    boxes: list[Annotation]


@dataclass(frozen=True)
class SensorEvent:
    """One sample_data record: a single sensor firing at its own timestamp.

    Exactly one of ``camera`` / ``lidar`` is set. ``boxes`` is non-empty only on
    lidar keyframe events, since annotations are defined per keyframe.
    """

    channel: str
    timestamp: int
    is_key_frame: bool
    camera: CameraFrame | None = None
    lidar: LidarFrame | None = None
    boxes: list[Annotation] = field(default_factory=list)


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

        boxes = [build_annotation(self.nusc, ann_token) for ann_token in sample["anns"]]

        return SceneSample(
            sample_token=sample["token"],
            scene_name=scene_name,
            timestamp=sample["timestamp"],
            cameras=cams,
            lidar=lidar,
            boxes=boxes,
        )

    def read_sample_data(
        self,
        scene: str,
        *,
        channels: Sequence[str] | None = None,
        keyframes_only: bool = False,
    ) -> Iterator[SensorEvent]:
        """Yield the full sample_data stream for a scene in timestamp order.

        Unlike `read_scene`, this includes intermediate sweeps and emits one
        sensor event at a time (cameras and lidar fire asynchronously at their
        own rates). `scene` accepts a name or token. `channels` defaults to the
        reader's cameras plus LIDAR_TOP when `include_lidar` is set.
        """
        rec = self._resolve_scene(scene)
        scene_samples = self._scene_sample_tokens(rec)
        if channels is None:
            channels = (*self.cameras, *([LIDAR] if self.include_lidar else ()))

        chains = [
            self._channel_sample_data(ch, rec, scene_samples, keyframes_only)
            for ch in channels
        ]
        for sd in heapq.merge(*chains, key=lambda r: r["timestamp"]):
            yield self._event(sd, scene_samples)

    def _scene_sample_tokens(self, rec: dict) -> set[str]:
        tokens: set[str] = set()
        token = rec["first_sample_token"]
        while token:
            tokens.add(token)
            token = self.nusc.get("sample", token)["next"]
        return tokens

    def _channel_sample_data(
        self, channel: str, rec: dict, scene_samples: set[str], keyframes_only: bool
    ) -> Iterator[dict]:
        """Walk one channel's sample_data chain, bounded to this scene, in time order."""
        first_sample = self.nusc.get("sample", rec["first_sample_token"])
        if channel not in first_sample["data"]:
            return
        sd = self.nusc.get("sample_data", first_sample["data"][channel])
        # The keyframe sits mid-chain; rewind to the scene's earliest sweep first.
        while sd["prev"]:
            prev = self.nusc.get("sample_data", sd["prev"])
            if prev["sample_token"] not in scene_samples:
                break
            sd = prev
        while True:
            if sd["sample_token"] in scene_samples and (not keyframes_only or sd["is_key_frame"]):
                yield sd
            if not sd["next"]:
                break
            nxt = self.nusc.get("sample_data", sd["next"])
            if nxt["sample_token"] not in scene_samples:
                break
            sd = nxt

    def _event(self, sd: dict, scene_samples: set[str]) -> SensorEvent:
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        sensor = self.nusc.get("sensor", cs["sensor_token"])
        channel, modality = sensor["channel"], sensor["modality"]
        calib = build_calib(self.nusc, sd)

        camera = lidar = None
        boxes: list[Annotation] = []
        if modality == "camera":
            camera = CameraFrame(
                channel=channel,
                image_path=self.dataroot / sd["filename"],
                width=int(sd["width"]),
                height=int(sd["height"]),
                calib=calib,
            )
        elif modality == "lidar":
            pts = np.fromfile(self.dataroot / sd["filename"], dtype=np.float32).reshape(-1, 5)
            lidar = LidarFrame(points=pts, calib=calib)
            if sd["is_key_frame"]:
                sample = self.nusc.get("sample", sd["sample_token"])
                boxes = [build_annotation(self.nusc, t) for t in sample["anns"]]

        return SensorEvent(
            channel=channel,
            timestamp=sd["timestamp"],
            is_key_frame=sd["is_key_frame"],
            camera=camera,
            lidar=lidar,
            boxes=boxes,
        )

    def _camera_frame(self, channel: str, sd_token: str) -> CameraFrame:
        sd = self.nusc.get("sample_data", sd_token)
        return CameraFrame(
            channel=channel,
            image_path=self.dataroot / sd["filename"],
            width=int(sd["width"]),
            height=int(sd["height"]),
            calib=build_calib(self.nusc, sd),
        )

    def _lidar_frame(self, sd_token: str) -> LidarFrame:
        sd = self.nusc.get("sample_data", sd_token)
        # nuScenes .pcd.bin: contiguous float32, five columns (x, y, z, intensity, ring).
        pts = np.fromfile(self.dataroot / sd["filename"], dtype=np.float32).reshape(-1, 5)
        return LidarFrame(points=pts, calib=build_calib(self.nusc, sd))
