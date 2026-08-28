#!/usr/bin/env python3
"""Geometry sanity check: pull-based BEVLift vs forward-scatter, same real batch.

Runs both lifts on one real keyframe through the *same* backbone weights, so any
difference in where features land is purely the lift geometry (not features).
Renders each lift's BEV feature-energy footprint next to the GT so a wrong K^-1,
sensor->ego, or an axis swap is visible.

    python scripts/compare_lifts.py --raster data/bev_rasters/trainval_val

The pull-based lift builds BEV in (x,y); the scatter writes (y,x). So a *correct*
scatter should match the pull-based lift's transpose. Both frames are shown, and
the footprint IoU is reported for each, so the true relationship is measured, not
assumed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, collate_fn  # noqa: E402
from bev.models import BEVLift, CameraBEVForwardScatter, FeatureDepthPredictor, ResNetBackbone  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--split", default="val")
    p.add_argument("--raster", type=Path, default=Path("data/bev_rasters/trainval_val"))
    p.add_argument("--index", type=int, default=0, help="which keyframe in the split")
    p.add_argument("--out", type=Path, default=Path("viz/compare_lifts.png"))
    p.add_argument("--min-z", type=float, default=-0.5)
    p.add_argument("--max-z", type=float, default=3.0)
    p.add_argument("--num-z", type=int, default=4)
    p.add_argument("--min-d", type=float, default=0.0)
    p.add_argument("--max-d", type=float, default=42.0)
    p.add_argument("--D", type=int, default=42)
    return p.parse_args()


def energy(bev: torch.Tensor) -> torch.Tensor:
    """Per-cell feature energy of a (1,C,H,W) BEV tensor -> (H,W)."""
    return bev[0].abs().sum(0).cpu()


def footprint_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    fa, fb = a > 0, b > 0
    union = (fa | fb).sum().item()
    return (fa & fb).sum().item() / union if union else 0.0


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ds = NuScenesBEVDataset(NuScenesConfig(
        dataroot=args.dataroot, version=args.version, split=args.split,
        cameras=CAMERAS, load_lidar=False, load_annotations=False,
        bev_raster_root=args.raster, require_files=True,
    ))
    spec = ds.raster_store.spec
    batch = collate_fn([ds[args.index]]).to(device)
    x = batch.camera_batch

    # shared backbone + depth head, so only the lift geometry differs
    backbone = ResNetBackbone()
    depth = FeatureDepthPredictor(args.D)
    pull = BEVLift(backbone, depth, spec, args.min_z, args.max_z, args.num_z,
                   args.min_d, args.max_d, args.D).to(device).eval()
    scatter = CameraBEVForwardScatter(backbone, depth, spec,
                                      args.min_d, args.max_d, args.D).to(device).eval()

    with torch.no_grad():
        e_pull = energy(pull(x))         # (x, y) frame
        e_scat = energy(scatter(x))      # (y, x) frame

    e_pull_T = e_pull.t().contiguous()   # bring pull into (y, x) to match scatter/GT
    gt = batch.bev_raster_batch.data[0].cpu()  # (layers, ny, nx) = (y, x)

    iou_aligned = footprint_iou(e_pull_T, e_scat)   # expected high if geometry consistent
    iou_raw = footprint_iou(e_pull, e_scat)         # expected lower (transposed)
    print(f"footprint IoU  scatter vs pull.T (y,x): {iou_aligned:.3f}")
    print(f"footprint IoU  scatter vs pull   (x,y): {iou_raw:.3f}")
    print(f"scatter nonzero cells: {(e_scat>0).sum().item()}   pull: {(e_pull>0).sum().item()}")

    # front camera image, denormalized for display
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    front_idx = list(x.channels).index("CAM_FRONT") if "CAM_FRONT" in x.channels else 0
    img = (x.images[0, front_idx].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    extent = [spec.x_min, spec.x_max, spec.y_min, spec.y_max]

    ax[0, 0].imshow(img); ax[0, 0].set_title(f"{x.channels[front_idx]}"); ax[0, 0].axis("off")
    ax[0, 1].imshow(gt[0], origin="lower", extent=extent, cmap="Greys")
    ax[0, 1].set_title("GT drivable (y,x)")
    ax[0, 2].imshow(gt[1], origin="lower", extent=extent, cmap="Greys")
    ax[0, 2].set_title("GT vehicle (y,x)")

    ax[1, 0].imshow(e_pull_T, origin="lower", extent=extent)
    ax[1, 0].set_title("pull energy -> (y,x) [reference]")
    ax[1, 1].imshow(e_scat, origin="lower", extent=extent)
    ax[1, 1].set_title(f"scatter energy (y,x)\nIoU vs pull.T = {iou_aligned:.3f}")
    ax[1, 2].imshow(e_pull, origin="lower", extent=extent)
    ax[1, 2].set_title(f"pull energy raw (x,y)\nIoU scatter-vs-this = {iou_raw:.3f}")
    for a in ax[1]:
        a.set_xlabel("x [m]"); a.set_ylabel("y [m]")

    fig.suptitle(f"lift geometry check — keyframe {args.index}  ({ds.scene_names[ds.sample_tokens[args.index]]})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
