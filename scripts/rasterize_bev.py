#!/usr/bin/env python3
"""Offline BEV rasterization job: precompute per-keyframe BEV ground truth.

    python scripts/rasterize_bev.py --dataroot data/nuscenes --split mini_train \
        --out data/bev_rasters/mini_train

Writes one raster per keyframe under --out, which the training dataset and the
scene reader then load (pass the same directory as `bev_raster_root`).

Iteration, the on-disk format, and resumable skipping are handled here; the
rasterization lives in a BEVRasterizer subclass.

The dataset is opened with an empty camera set and load_lidar off, so iteration
carries just the boxes and tokens the rasterizer reads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.map_mask import MapMask
import pyquaternion

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, Sample  # noqa: E402
from bev.raster import BEVGridSpec, BEVRasterStore, BEVRasterizer, build_rasters  # noqa: E402
from bev.transforms import Transform  # noqa: E402
from bev.types import Box  # noqa: E402


def _load_map(map_path: Path, resolution: float = 0.1) -> MapMask:
    """Load a nuScenes binary map-mask PNG as a MapMask.

    MapMask bundles the mask array with its resolution and the global->pixel
    transform (transform_matrix / to_pixel_coords), and handles the mask's size
    past PIL's decompression-bomb guard on its own.
    """
    if not map_path.is_file():
        raise FileNotFoundError(f"map mask not found: {map_path}")
    return MapMask(str(map_path), resolution=resolution)

class BBoxRasterizer(BEVRasterizer):
    """Rasterizer for 3D bounding boxes of the nuScenes dataset by mapping and projecting the boxes to the BEV grid."""

    def __init__(self, spec: BEVGridSpec, map_paths: Sequence[Path], nusc: NuScenes) -> None:
        super().__init__(spec, layer_names=["drivable_area", "vehicle"])
        self.nusc = nusc
        self.maps = self._load_maps_by_location(map_paths)

    def _load_maps_by_location(self, map_paths: Sequence[Path]) -> dict[str, MapMask]:
        # map.json can reference logs outside this split; key off logs that exist.
        valid_logs = {log["token"] for log in self.nusc.log}
        location_by_file: dict[str, str] = {}
        for record in self.nusc.map:
            for log_token in record["log_tokens"]:
                if log_token in valid_logs:
                    location_by_file[Path(record["filename"]).name] = self.nusc.get(
                        "log", log_token
                    )["location"]
                    break

        maps: dict[str, MapMask] = {}
        for path in map_paths:
            path = Path(path)
            location = location_by_file.get(path.name)
            if location is None:
                print(f"  warning: {path.name} has no location in map.json; skipping")
                continue
            maps[location] = _load_map(path)
        return maps

    def _location_of(self, sample_token: str) -> str:
        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample["scene_token"])
        return self.nusc.get("log", scene["log_token"])["location"]

    def rasterize(self, sample: Sample) -> np.ndarray:
        raster = np.zeros((len(self.layer_names), self.spec.ny, self.spec.nx))
        ego2global = Transform(sample.lidar.calib.ego2global_translation.numpy(), pyquaternion.Quaternion(sample.lidar.calib.ego2global_rotation.numpy()).rotation_matrix, sample.timestamp)
        if len(sample.boxes) > 0:
            self._rasterize_boxes(raster, np.stack([box.box.bottom_corners().transpose(1,0) for box in sample.boxes], axis=0), ego2global) # (N, 4, 3)
        map_mask = self.maps.get(self._location_of(sample.sample_token))
        if map_mask is not None:
            self._rasterize_map(raster, map_mask, ego2global)
        return raster

    def _ccw(self, pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray) -> bool | np.ndarray:
        return np.cross(pt2 - pt1, pt3 - pt1) > 0
    
    def _inside_bbox(self, pt: np.ndarray, box_corners: np.ndarray):
        if not self._ccw(box_corners[0], box_corners[1], box_corners[2]):
            box_corners = box_corners[::-1]
        return self._ccw(box_corners[0], box_corners[1], pt) and \
            self._ccw(box_corners[1], box_corners[2], pt) and \
            self._ccw(box_corners[2], box_corners[3], pt) and \
            self._ccw(box_corners[3], box_corners[0], pt)
        
    def _rasterize_boxes(self, raster: np.ndarray, boxes_corners: np.ndarray, ego2global: Transform) -> np.ndarray:
        ego_boxes_corners = (boxes_corners @ ego2global.inverse().rotation.T + ego2global.inverse().translation)[:,:,:2] # (N,4,2)
        min_pt, max_pt = np.min(ego_boxes_corners, axis=1), np.max(ego_boxes_corners, axis=1) # (N,2)
        min_idcs = np.floor((min_pt - np.array([self.spec.x_min,self.spec.y_min])) / self.spec.resolution).astype(int) # (N,2)
        max_idcs = np.floor((max_pt - np.array([self.spec.x_min,self.spec.y_min])) / self.spec.resolution).astype(int) # (N.2)
        all_idcs = []
        all_boxes = []
        for i in range(len(ego_boxes_corners)):
            ii, jj = np.meshgrid(np.arange(min_idcs[i,0], max_idcs[i,0]+1), np.arange(min_idcs[i,1], max_idcs[i,1]+1))
            ii = ii.ravel()
            jj = jj.ravel()
            idcs = np.stack([ii,jj], axis=1) # (M,2)
            all_idcs.append(idcs)
            all_boxes.append(np.stack([ego_boxes_corners[i] if self._ccw(ego_boxes_corners[i,0], ego_boxes_corners[i,1], ego_boxes_corners[i,2]) else ego_boxes_corners[i,::-1,:]]*len(idcs), axis=0)) # (M.4.2)
        all_idcs = np.concatenate(all_idcs, axis=0) # (all, 2)
        all_boxes = np.concatenate(all_boxes, axis=0) # (all, 4, 2)
        all_pts = (all_idcs + 0.5)*self.spec.resolution + np.array([self.spec.x_min, self.spec.y_min]) # (all, 2)
        in_box_mask = self._ccw(all_boxes[:,0], all_boxes[:,1], all_pts) & \
            self._ccw(all_boxes[:,1], all_boxes[:,2], all_pts) & \
            self._ccw(all_boxes[:,2], all_boxes[:,3], all_pts) & \
            self._ccw(all_boxes[:,3], all_boxes[:,0], all_pts) & \
            (0 <= all_idcs[:,0]) & (all_idcs[:,0] < self.spec.nx) & \
            (0 <= all_idcs[:,1]) & (all_idcs[:,1] < self.spec.ny)
        if np.any(in_box_mask):
            raster[1, all_idcs[in_box_mask][:,1], all_idcs[in_box_mask][:,0]] = 1

    
    def _rasterize_box(self, raster: np.ndarray, box_corners: np.ndarray, ego2global: Transform) -> np.ndarray:
        ego_box_corners = np.array([ego2global.inverse()(box_corners[:,i])[:2] for i in range(4)])
        min_pt, max_pt = np.min(ego_box_corners, axis=0), np.max(ego_box_corners, axis=0)
        min_idx = self.spec.to_index(min_pt)
        max_idx = self.spec.to_index(max_pt)
        for i in range(min_idx[0], max_idx[0]+1):
            for j in range(min_idx[1], max_idx[1]+1):
                if not self.spec.idx_in_bounds((i, j)):
                    continue
                if self._inside_bbox(self.spec.ctr_from_index((i, j)), ego_box_corners):
                    raster[1, j, i] = 1

    def _rasterize_map(self, raster: np.ndarray, map_mask: MapMask, ego2global: Transform) -> np.ndarray:
        ii, jj = np.meshgrid(np.arange(self.spec.nx), np.arange(self.spec.ny))
        xs = (ii+0.5) * self.spec.resolution + self.spec.x_min
        ys = (jj+0.5) * self.spec.resolution + self.spec.y_min
        global_pts = (np.stack([xs, ys, np.zeros(xs.shape)], axis=0).T @ ego2global.rotation.T + ego2global.translation).T[:2]
        on_mask = map_mask.is_on_mask(*global_pts.reshape(2,-1)).reshape(xs.shape)
        raster[0, jj[on_mask], ii[on_mask]] = 1
        

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--split", default="mini_train")
    p.add_argument("--out", type=Path, required=True, help="output store directory")
    p.add_argument("--layers", nargs="+", default=["drivable_area", "vehicle"])
    p.add_argument("--x-range", type=float, nargs=2, default=(-50.0, 50.0))
    p.add_argument("--y-range", type=float, nargs=2, default=(-50.0, 50.0))
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true", help="rewrite existing rasters")
    p.add_argument("--require-files", action="store_true", help="only rasterize keyframes whose sensor files exist (partial downloads)")
    p.add_argument(
        "--map-paths",
        type=Path,
        nargs="+",
        default=None,
        help="map-mask PNGs to rasterize; default: every PNG under <dataroot>/maps",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    spec = BEVGridSpec(
        x_min=args.x_range[0],
        x_max=args.x_range[1],
        y_min=args.y_range[0],
        y_max=args.y_range[1],
        resolution=args.resolution,
    )
    store = BEVRasterStore(args.out, spec, layer_names=args.layers)

    dataset = NuScenesBEVDataset(
        NuScenesConfig(
            dataroot=args.dataroot,
            version=args.version,
            split=args.split,
            cameras=(),  # rasterization reads only boxes
            load_lidar=True,
            load_annotations=True,
            require_files=args.require_files,
        )
    )
    print(
        f"split {args.split!r}: {len(dataset)} keyframes -> {args.out}\n"
        f"grid {spec.nx}x{spec.ny} @ {spec.resolution} m, layers={args.layers}"
    )

    map_paths = args.map_paths or sorted((args.dataroot / "maps").glob("*.png"))
    if not map_paths:
        raise SystemExit(f"no map masks found under {args.dataroot / 'maps'}; pass --map-paths")
    print(f"maps: {len(map_paths)} mask(s)")

    rasterizer = BBoxRasterizer(spec, map_paths, dataset.nusc)
    written = build_rasters(
        rasterizer,
        (dataset[i] for i in range(len(dataset))),
        store,
        overwrite=args.overwrite,
        progress=lambda n, tok: print(f"  [{n}] {tok}") if n % 50 == 0 else None,
    )
    print(f"wrote {written} rasters to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
