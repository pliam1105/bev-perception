"""Build a tiny synthetic nuScenes tree the devkit can load.

    dataroot = build(tmp_path / "data")
    nusc = NuScenes("v1.0-mini", dataroot=str(dataroot))

Produces a valid v1.0-mini directory: the 13 JSON tables plus real (small)
image and lidar files, two scenes (one in mini_train, one in mini_val), each
with two keyframes, six cameras and one lidar. Every sample_data gets its own
ego_pose with a distinct timestamp and translation, so per-sensor poses stay
distinguishable.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


CAMERAS: tuple[str, ...] = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
LIDAR = "LIDAR_TOP"

# scene-0061 is in mini_train, scene-0103 is in mini_val (see create_splits_scenes).
SCENES = ("scene-0061", "scene-0103")
SAMPLES_PER_SCENE = 2
IMG_W, IMG_H = 32, 18
LIDAR_N = 64

CATEGORY_TOKEN = "cat_car"
VISIBILITY_TOKEN = "4"
LOG_TOKEN = "log_0"


def _write_json(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, indent=0))


def _write_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


def _write_lidar(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((LIDAR_N, 5)).astype(np.float32)
    path.write_bytes(struct.pack(f"<{pts.size}f", *pts.reshape(-1)))


def build(dataroot: Path | str) -> Path:
    dataroot = Path(dataroot)
    table_dir = dataroot / "v1.0-mini"
    table_dir.mkdir(parents=True, exist_ok=True)

    sensors: list[dict] = []
    calibrated_sensors: list[dict] = []
    sensor_order = (*CAMERAS, LIDAR)
    for idx, channel in enumerate(sensor_order):
        modality = "lidar" if channel == LIDAR else "camera"
        sensors.append({"token": f"sensor_{channel}", "channel": channel, "modality": modality})
        if modality == "camera":
            fx = 1000.0 + 10 * idx
            intrinsic = [[fx, 0.0, IMG_W / 2], [0.0, fx, IMG_H / 2], [0.0, 0.0, 1.0]]
        else:
            intrinsic = []
        calibrated_sensors.append(
            {
                "token": f"cs_{channel}",
                "sensor_token": f"sensor_{channel}",
                "translation": [float(idx), 0.0, 1.5],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": intrinsic,
            }
        )

    scenes: list[dict] = []
    samples: list[dict] = []
    sample_data: list[dict] = []
    ego_poses: list[dict] = []
    annotations: list[dict] = []
    instances: list[dict] = []

    for si, scene_name in enumerate(SCENES):
        scene_token = f"scene_{si}"
        sample_tokens = [f"sample_{si}_{k}" for k in range(SAMPLES_PER_SCENE)]

        for k, sample_token in enumerate(sample_tokens):
            base_ts = si * 1_000_000 + k * 500_000
            samples.append(
                {
                    "token": sample_token,
                    "timestamp": base_ts,
                    "scene_token": scene_token,
                    "prev": sample_tokens[k - 1] if k > 0 else "",
                    "next": sample_tokens[k + 1] if k + 1 < len(sample_tokens) else "",
                }
            )

            for s_idx, channel in enumerate(sensor_order):
                ep_token = f"ego_{si}_{k}_{channel}"
                ego_poses.append(
                    {
                        "token": ep_token,
                        "timestamp": base_ts + s_idx,
                        "translation": [
                            si * 1000.0 + k * 100.0 + s_idx,
                            float(si),
                            float(k),
                        ],
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                    }
                )

                sd_token = f"sd_{si}_{k}_{channel}"
                if channel == LIDAR:
                    rel = f"samples/{channel}/{sd_token}.pcd.bin"
                    fpath = dataroot / rel
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    _write_lidar(fpath, seed=hash((sd_token,)) & 0xFFFF)
                    w, h, fmt = 0, 0, "bin"
                else:
                    rel = f"samples/{channel}/{sd_token}.jpg"
                    fpath = dataroot / rel
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    _write_image(fpath, seed=hash((sd_token,)) & 0xFFFF)
                    w, h, fmt = IMG_W, IMG_H, "jpg"

                sample_data.append(
                    {
                        "token": sd_token,
                        "sample_token": sample_token,
                        "ego_pose_token": ep_token,
                        "calibrated_sensor_token": f"cs_{channel}",
                        "timestamp": base_ts + s_idx,
                        "fileformat": fmt,
                        "is_key_frame": True,
                        "height": h,
                        "width": w,
                        "filename": rel,
                        "prev": "",
                        "next": "",
                    }
                )

            for a in range(3):
                ann_token = f"ann_{si}_{k}_{a}"
                inst_token = f"inst_{si}_{k}_{a}"
                instances.append(
                    {
                        "token": inst_token,
                        "category_token": CATEGORY_TOKEN,
                        "nbr_annotations": 1,
                        "first_annotation_token": ann_token,
                        "last_annotation_token": ann_token,
                    }
                )
                annotations.append(
                    {
                        "token": ann_token,
                        "sample_token": sample_token,
                        "instance_token": inst_token,
                        "visibility_token": VISIBILITY_TOKEN,
                        "attribute_tokens": [],
                        "translation": [10.0 + a, 20.0 + a + si, 1.0],
                        "size": [1.9 + 0.1 * a, 4.6 + 0.1 * a, 1.7],
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                        "num_lidar_pts": [10, 3, 20][a],
                        "num_radar_pts": [2, 0, 5][a],
                        "prev": "",
                        "next": "",
                    }
                )

        scenes.append(
            {
                "token": scene_token,
                "name": scene_name,
                "description": "synthetic",
                "log_token": LOG_TOKEN,
                "nbr_samples": len(sample_tokens),
                "first_sample_token": sample_tokens[0],
                "last_sample_token": sample_tokens[-1],
            }
        )

    map_rel = "maps/fake.png"
    map_path = dataroot / map_rel
    map_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(map_path, format="PNG")

    logs = [
        {
            "token": LOG_TOKEN,
            "logfile": "fake",
            "vehicle": "fake",
            "date_captured": "2020-01-01",
            "location": "singapore-onenorth",
        }
    ]
    maps = [
        {
            "token": "map_0",
            "filename": "maps/fake.png",
            "category": "semantic_pras",
            "log_tokens": [LOG_TOKEN],
        }
    ]
    categories = [{"token": CATEGORY_TOKEN, "name": "vehicle.car", "description": "car"}]
    attributes: list[dict] = []
    visibility = [
        {"token": "1", "level": "v0-40", "description": "0-40%"},
        {"token": "2", "level": "v40-60", "description": "40-60%"},
        {"token": "3", "level": "v60-80", "description": "60-80%"},
        {"token": "4", "level": "v80-100", "description": "80-100%"},
    ]

    _write_json(table_dir / "category.json", categories)
    _write_json(table_dir / "attribute.json", attributes)
    _write_json(table_dir / "visibility.json", visibility)
    _write_json(table_dir / "instance.json", instances)
    _write_json(table_dir / "sensor.json", sensors)
    _write_json(table_dir / "calibrated_sensor.json", calibrated_sensors)
    _write_json(table_dir / "ego_pose.json", ego_poses)
    _write_json(table_dir / "log.json", logs)
    _write_json(table_dir / "scene.json", scenes)
    _write_json(table_dir / "sample.json", samples)
    _write_json(table_dir / "sample_data.json", sample_data)
    _write_json(table_dir / "sample_annotation.json", annotations)
    _write_json(table_dir / "map.json", maps)

    return dataroot
