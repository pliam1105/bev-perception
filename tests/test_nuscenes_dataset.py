"""Loader smoke tests against the synthetic tree from make_fake_nuscenes.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, collate_fn  # noqa: E402
from make_fake_nuscenes import build  # noqa: E402


@pytest.fixture(scope="session")
def dataroot(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("nuscenes") / "data")


def _dataset(dataroot: Path, **kwargs) -> NuScenesBEVDataset:
    return NuScenesBEVDataset(NuScenesConfig(dataroot=dataroot, **kwargs))


def test_missing_dataroot_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataroot not found"):
        _dataset(tmp_path / "nope")


def test_unknown_split_rejected(dataroot):
    with pytest.raises(KeyError, match="Unknown split"):
        _dataset(dataroot, split="not_a_split")


def test_splits_are_disjoint(dataroot):
    train = set(_dataset(dataroot, split="mini_train").sample_tokens)
    val = set(_dataset(dataroot, split="mini_val").sample_tokens)
    assert train and val
    assert not train & val


def test_samples_are_scene_then_time_ordered(dataroot):
    dataset = _dataset(dataroot, split="mini_train")
    keys = [
        (dataset.nusc.get("sample", t)["scene_token"],
         dataset.nusc.get("sample", t)["timestamp"])
        for t in dataset.sample_tokens
    ]
    assert keys == sorted(keys)


def test_sample_carries_every_camera_with_calibration(dataroot):
    sample = _dataset(dataroot, split="mini_train")[0]

    assert set(sample.cameras) == set(CAMERAS)
    for channel in CAMERAS:
        cam = sample.cameras[channel]
        assert cam.channel == channel
        assert cam.image.dtype == torch.uint8
        assert cam.image.ndim == 3 and cam.image.shape[0] == 3
        assert cam.path.is_file()

        calib = cam.calib
        assert calib.intrinsic.shape == (3, 3)
        assert calib.sensor2ego_translation.shape == (3,)
        assert calib.sensor2ego_rotation.shape == (4,)
        assert calib.ego2global_translation.shape == (3,)
        assert calib.ego2global_rotation.shape == (4,)
        # Quaternions come through normalised and in nuScenes wxyz order.
        assert np.isclose(np.linalg.norm(calib.sensor2ego_rotation), 1.0)

    assert sample.boxes


def test_per_sensor_ego_poses_are_not_collapsed(dataroot):
    """Cameras fire at different instants; each must keep its own ego pose."""
    sample = _dataset(dataroot, split="mini_train")[0]
    stamps = {c.calib.timestamp for c in sample.cameras.values()}
    poses = {tuple(c.calib.ego2global_translation) for c in sample.cameras.values()}
    assert len(stamps) == len(CAMERAS)
    assert len(poses) == len(CAMERAS)


def test_boxes_are_global_frame_and_unmoved(dataroot):
    """Box centres must match sample_annotation.translation byte for byte."""
    dataset = _dataset(dataroot, split="mini_train")
    sample = dataset[0]

    assert sample.boxes
    for ann in sample.boxes:
        record = dataset.nusc.get("sample_annotation", ann.box.token)
        assert np.allclose(ann.box.center, record["translation"])
        assert np.allclose(ann.box.wlh, record["size"])
        assert ann.box.name == record["category_name"]
        assert ann.instance_token == record["instance_token"]
        assert ann.num_lidar_pts == record["num_lidar_pts"]
        assert ann.visibility == record["visibility_token"]


def test_annotations_are_opt_out(dataroot):
    assert _dataset(dataroot, split="mini_train", load_annotations=False)[0].boxes == []


def test_lidar_is_opt_in_and_shaped(dataroot):
    assert _dataset(dataroot, split="mini_train")[0].lidar is None

    lidar = _dataset(dataroot, split="mini_train", load_lidar=True)[0].lidar
    assert lidar is not None
    assert lidar.points.dtype == torch.float32
    assert lidar.points.ndim == 2 and lidar.points.shape[1] == 5
    assert lidar.calib.intrinsic is None  # not a camera


def test_camera_subset_is_honoured(dataroot):
    sample = _dataset(dataroot, split="mini_train", cameras=["CAM_FRONT"])[0]
    assert set(sample.cameras) == {"CAM_FRONT"}


def test_collate_fn_is_not_implemented(dataroot):
    with pytest.raises(NotImplementedError, match="not implemented"):
        collate_fn([_dataset(dataroot, split="mini_train")[0]])
