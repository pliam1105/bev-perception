#!/usr/bin/env python3
"""Verify nuScenes loading and eyeball one keyframe.

    python scripts/inspect_sample.py --dataroot data/nuscenes --index 0 --out sample.png

Prints what the loader actually produced (shapes, dtypes, raw calibration) and
optionally renders the six cameras in their physical layout. This is the
smoke test to run before trusting anything downstream -- if the surround grid
does not look like a coherent ring around the car, stop here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig  # noqa: E402

# Physical arrangement of the surround ring, for the render only.
GRID = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--lidar", action="store_true", help="also load LIDAR_TOP")
    parser.add_argument("--out", type=Path, help="write the surround grid here (PNG)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset = NuScenesBEVDataset(
        NuScenesConfig(
            dataroot=args.dataroot,
            version=args.version,
            split=args.split,
            load_lidar=args.lidar,
            verbose=True,
        )
    )
    print(f"\nsplit {args.split!r}: {len(dataset)} keyframes "
          f"across {len(set(dataset.scene_names.values()))} scenes")

    if not len(dataset):
        print("empty split -- nothing to inspect", file=sys.stderr)
        return 1

    sample = dataset[args.index % len(dataset)]
    print(f"\nsample {sample.sample_token}  scene={sample.scene_name}  "
          f"t={sample.timestamp}  boxes={len(sample.boxes)}")

    np.set_printoptions(precision=3, suppress=True)
    for channel in CAMERAS:
        cam = sample.cameras.get(channel)
        if cam is None:
            continue
        calib = cam.calib
        fx, fy = calib.intrinsic[0, 0], calib.intrinsic[1, 1]
        cx, cy = calib.intrinsic[0, 2], calib.intrinsic[1, 2]
        print(
            f"\n  {channel:<16} image={tuple(cam.image.shape)} {cam.image.dtype}\n"
            f"    intrinsic      fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}\n"
            f"    sensor->ego    t={calib.sensor2ego_translation} q={calib.sensor2ego_rotation}\n"
            f"    ego->global    t={calib.ego2global_translation}\n"
            f"    dt vs keyframe {(calib.timestamp - sample.timestamp) / 1e3:+.1f} ms"
        )

    if sample.lidar is not None:
        lidar = sample.lidar
        pts = lidar.points
        print(
            f"\n  LIDAR_TOP        points={tuple(pts.shape)} {pts.dtype}\n"
            f"    xyz range      min={pts[:, :3].min(0).values.numpy()} "
            f"max={pts[:, :3].max(0).values.numpy()}\n"
            f"    sensor->ego    t={lidar.calib.sensor2ego_translation}\n"
            f"    dt vs keyframe {(lidar.calib.timestamp - sample.timestamp) / 1e3:+.1f} ms"
        )

    if sample.boxes:
        counts: dict[str, int] = {}
        for ann in sample.boxes:
            counts[ann.box.name] = counts.get(ann.box.name, 0) + 1
        print("\n  boxes (global frame), by category:")
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>3}  {name}")
        sparse = sum(a.num_lidar_pts < 5 for a in sample.boxes)
        print(f"    {sparse}/{len(sample.boxes)} have <5 lidar points")
        ann = sample.boxes[0]
        print(f"\n  first box  {ann.box.name}\n"
              f"    center (global) {ann.box.center}\n"
              f"    wlh             {ann.box.wlh}\n"
              f"    yaw             {ann.box.orientation.yaw_pitch_roll[0]:+.3f} rad\n"
              f"    lidar/radar pts {ann.num_lidar_pts}/{ann.num_radar_pts}  "
              f"visibility={ann.visibility}")

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 6))
        for row, channels in enumerate(GRID):
            for col, channel in enumerate(channels):
                ax = axes[row][col]
                ax.set_axis_off()
                cam = sample.cameras.get(channel)
                if cam is None:
                    continue
                ax.imshow(cam.image.permute(1, 2, 0).numpy())
                ax.set_title(channel, fontsize=9)
        fig.suptitle(f"{sample.scene_name} — {sample.sample_token}", fontsize=10)
        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
