#!/usr/bin/env python3
"""Per-camera geometry validation against the world frame.

nuScenes ego frame is x-forward, y-left. So each camera's lifted features must
land in the BEV direction that camera physically points: CAM_FRONT -> +x (0deg),
CAM_BACK -> 180deg, left cams -> +y, right cams -> -y. This lifts each camera on
its own and reports the energy-centroid azimuth for both lifts, so a wrong K^-1,
sensor->ego, or axis swap shows up as a centroid pointing the wrong way.

    python scripts/check_lift_directions.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, collate_fn  # noqa: E402
from bev.models import BEVLift, CameraBEVForwardScatter, FeatureDepthPredictor, ResNetBackbone  # noqa: E402

# approximate azimuth each nuScenes camera points, ego frame (x fwd, y left), degrees
EXPECTED_AZ = {
    "CAM_FRONT": 0, "CAM_FRONT_LEFT": 55, "CAM_FRONT_RIGHT": -55,
    "CAM_BACK_LEFT": 110, "CAM_BACK_RIGHT": -110, "CAM_BACK": 180,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--split", default="val")
    p.add_argument("--raster", type=Path, default=Path("data/bev_rasters/trainval_val"))
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--D", type=int, default=42)
    return p.parse_args()


def slice_cam(x, n: int):
    return SimpleNamespace(
        channels=(x.channels[n],),
        images=x.images[:, n:n+1],
        intrinsics=x.intrinsics[:, n:n+1],
        bev2ego=x.bev2ego[:, n:n+1],
        bev2pixel=x.bev2pixel[:, n:n+1],
    )


def centroid_az(e: torch.Tensor, spec, frame: str) -> tuple[float, float, float]:
    """Energy-weighted centroid (x,y) in metres and its azimuth; frame is 'xy' or 'yx'."""
    e = e.clamp(min=0)
    a = spec.x_min + (torch.arange(e.shape[0]) + 0.5) * spec.resolution
    b = spec.y_min + (torch.arange(e.shape[1]) + 0.5) * spec.resolution
    d0 = (e.sum(1) * a).sum() / e.sum()
    d1 = (e.sum(0) * b).sum() / e.sum()
    cx, cy = (d0, d1) if frame == "xy" else (d1, d0)  # scatter stores (y, x)
    return float(cx), float(cy), math.degrees(math.atan2(cy, cx))


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NuScenesBEVDataset(NuScenesConfig(
        dataroot=args.dataroot, version=args.version, split=args.split,
        cameras=CAMERAS, load_lidar=False, load_annotations=False,
        bev_raster_root=args.raster, require_files=True,
    ))
    spec = ds.raster_store.spec
    x = collate_fn([ds[args.index]]).to(device).camera_batch

    backbone, depth = ResNetBackbone(), FeatureDepthPredictor(args.D)
    pull = BEVLift(backbone, depth, spec, -0.5, 3.0, 4, 0.0, 42.0, args.D).to(device).eval()
    scatter = CameraBEVForwardScatter(backbone, depth, spec, 0.0, 42.0, args.D).to(device).eval()

    print(f"{'camera':<16}{'expect':>8}{'pull az':>10}{'scatter az':>12}   pull(x,y)   scatter(x,y)")
    with torch.no_grad():
        for n, ch in enumerate(x.channels):
            xc = slice_cam(x, n)
            pe = pull(xc)[0].abs().sum(0).cpu()
            se = scatter(xc)[0].abs().sum(0).cpu()
            pcx, pcy, paz = centroid_az(pe, spec, "xy")
            scx, scy, saz = centroid_az(se, spec, "yx")
            exp = EXPECTED_AZ.get(ch, float("nan"))
            print(f"{ch:<16}{exp:>8.0f}{paz:>10.0f}{saz:>12.0f}   "
                  f"({pcx:+5.1f},{pcy:+5.1f})   ({scx:+5.1f},{scy:+5.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
