"""nuScenes keyframe reader: surround images + lidar + raw calibration.

A torch ``Dataset`` that returns one keyframe (``sample``) at a time, carrying
per-sensor raw calibration and annotation boxes in the global frame. Calibration
is passed through untouched — intrinsics and the sensor->ego / ego->global poses
are handed on as-is for downstream geometry to consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pyquaternion
import torch
from PIL import Image
import torch.nn.functional as F

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from bev.raster import BEVGridSpec, BEVRaster, BEVRasterStore, BEVRasterBatch
from bev.types import (
    CAMERAS,
    LIDAR,
    Annotation,
    SensorCalib,
    build_annotation,
    build_calib,
)

from bev.transforms import Transform


@dataclass
class NuScenesConfig:
    dataroot: Path
    version: str = "v1.0-mini"
    split: str = "mini_train"
    cameras: Sequence[str] = CAMERAS
    load_lidar: bool = False
    load_annotations: bool = True
    bev_raster_root: Path | None = None  # precomputed BEV rasters (see scripts/rasterize_bev.py)
    verbose: bool = False


@dataclass(frozen=True)
class CameraData:
    channel: str
    image: torch.Tensor  # uint8, (3, H, W)
    path: Path
    calib: SensorCalib

@dataclass
class CameraDataBatch:
    channels: tuple[str, ...] # N
    images: torch.Tensor  # uint8, (B, N, 3, H, W)
    bev2pixel: torch.Tensor # float32, (B, N, 3, 4)

    def to(self, device: str):
        self.images = self.images.to(device)
        self.bev2pixel = self.bev2pixel.to(device)
        return self

@dataclass(frozen=True)
class LidarData:
    points: torch.Tensor  # float32, (N, 5): x, y, z, intensity, ring
    path: Path
    calib: SensorCalib


@dataclass(frozen=True)
class Sample:
    sample_token: str
    scene_name: str
    timestamp: int  # keyframe timestamp, microseconds
    cameras: dict[str, CameraData]
    lidar: LidarData | None
    boxes: list[Annotation] = field(default_factory=list)
    bev_raster: BEVRaster | None = None

@dataclass
class SampleBatch:
    camera_batch: CameraDataBatch
    bev_raster_batch: BEVRasterBatch

    def pin_memory(self: SampleBatch):
        self.camera_batch.images = self.camera_batch.images.pin_memory()
        self.camera_batch.bev2pixel = self.camera_batch.bev2pixel.pin_memory()
        self.bev_raster_batch.data = self.bev_raster_batch.data.pin_memory()

        return self

    def to(self, device: str):
        self.camera_batch.to(device)
        self.bev_raster_batch.to(device)

        return self


class NuScenesBEVDataset(torch.utils.data.Dataset):
    """One keyframe per item, ordered by (scene, time) within the split."""

    def __init__(self, config: NuScenesConfig) -> None:
        self.config = config
        dataroot = Path(config.dataroot)
        if not dataroot.exists():
            raise FileNotFoundError(f"dataroot not found: {dataroot}")
        self.dataroot = dataroot
        self.cameras = tuple(config.cameras)

        self.nusc = NuScenes(
            version=config.version, dataroot=str(dataroot), verbose=config.verbose
        )

        splits = create_splits_scenes()
        if config.split not in splits:
            raise KeyError(f"Unknown split {config.split!r} (have {sorted(splits)})")
        wanted = set(splits[config.split])

        scene_token_of: dict[str, str] = {}
        self.scene_names: dict[str, str] = {}
        rows: list[tuple[str, int, str]] = []  # (scene_token, timestamp, sample_token)
        for scene in self.nusc.scene:
            if scene["name"] not in wanted:
                continue
            token = scene["first_sample_token"]
            while token:
                sample = self.nusc.get("sample", token)
                rows.append((scene["token"], sample["timestamp"], token))
                scene_token_of[token] = scene["token"]
                self.scene_names[token] = scene["name"]
                token = sample["next"]

        rows.sort(key=lambda r: (r[0], r[1]))
        self.sample_tokens: list[str] = [r[2] for r in rows]

        self.raster_store = (
            BEVRasterStore.open(config.bev_raster_root) if config.bev_raster_root else None
        )

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, index: int) -> Sample:
        token = self.sample_tokens[index]
        sample = self.nusc.get("sample", token)

        cameras: dict[str, CameraData] = {}
        for channel in self.cameras:
            sd_token = sample["data"].get(channel)
            if sd_token is not None:
                cameras[channel] = self._camera(channel, sd_token)

        lidar: LidarData | None = None
        if self.config.load_lidar and LIDAR in sample["data"]:
            lidar = self._lidar(sample["data"][LIDAR])

        boxes: list[Annotation] = []
        if self.config.load_annotations:
            boxes = [self._annotation(t) for t in sample["anns"]]

        bev_raster = self.raster_store.load(token, to_torch=True) if self.raster_store else None

        return Sample(
            sample_token=token,
            scene_name=self.scene_names[token],
            timestamp=sample["timestamp"],
            cameras=cameras,
            lidar=lidar,
            boxes=boxes,
            bev_raster=bev_raster,
        )

    def _camera(self, channel: str, sd_token: str) -> CameraData:
        sd = self.nusc.get("sample_data", sd_token)
        path = self.dataroot / sd["filename"]
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB"))  # np.array copies -> writable buffer
        image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return CameraData(channel=channel, image=image, path=path, calib=build_calib(self.nusc, sd, to_torch=True))

    def _lidar(self, sd_token: str) -> LidarData:
        sd = self.nusc.get("sample_data", sd_token)
        path = self.dataroot / sd["filename"]
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
        points = torch.from_numpy(np.ascontiguousarray(pts))
        return LidarData(points=points, path=path, calib=build_calib(self.nusc, sd, to_torch=True))

    def _annotation(self, ann_token: str) -> Annotation:
        return build_annotation(self.nusc, ann_token)


def collate_fn(batch: Sequence[Sample]):
    """Collate a batch of samples.
    """
    H_NEW = 448
    W_NEW = 800
    imgs = torch.stack([torch.stack([(camera.image.float()/255.0 - torch.Tensor([0.485,0.456,0.406]).reshape(3,1,1))/torch.Tensor([0.229,0.224,0.225]).reshape(3,1,1) for camera in sample.cameras.values()]) for sample in batch])
    return SampleBatch(
        camera_batch=CameraDataBatch(
            channels = tuple(batch[0].cameras.keys()), # assume all samples have the same cameras
            images = F.interpolate(imgs.reshape(imgs.shape[0]*imgs.shape[1], *imgs.shape[2:]), size=(H_NEW, W_NEW), mode='bilinear', align_corners=False, antialias=True).reshape(imgs.shape[0], imgs.shape[1], 3, H_NEW, W_NEW),
            bev2pixel = torch.stack([torch.stack([torch.diag(torch.tensor([W_NEW/camera.image.shape[2], H_NEW/camera.image.shape[1], 1])) @  camera.calib.intrinsic.float() @ torch.tensor(Transform(camera.calib.sensor2ego_translation.numpy(), pyquaternion.Quaternion(camera.calib.sensor2ego_rotation.numpy()).rotation_matrix, batch[i].timestamp).inverse().toMatrix()[:3, :], dtype=torch.float32) for camera in batch[i].cameras.values()]) for i in range(len(batch))]),
        ),
        bev_raster_batch=BEVRasterBatch(
            data = torch.stack([sample.bev_raster.data for sample in batch]),
            layer_names = batch[0].bev_raster.layer_names,
            spec = batch[0].bev_raster.spec,
        ),
    )