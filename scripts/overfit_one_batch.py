#!/usr/bin/env python3
"""Overfit a single batch as an end-to-end sanity check.

    python scripts/overfit_one_batch.py --steps 300 --batch 2

Pulls one batch, then runs the full model + loss for many steps on that *same*
batch. A correct pipeline drives the loss down and per-class BEV IoU up; a flat
loss means a wiring bug (target misalignment, detached grad, wrong LR) that this
catches in seconds instead of after a full-dataset run.

The achievable IoU is capped by lift coverage: BEV cells that no camera sees get
zero features, so the ceiling is below 1.0 by design — the point is that the
covered region is learned.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, collate_fn  # noqa: E402
from bev.models import CameraBEVSeg, SegLoss  # noqa: E402


def per_class_iou(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> list[float]:
    """IoU per channel at a probability threshold; union==0 counts as perfect."""
    pred = (torch.sigmoid(logits) > thresh)
    gt = target > 0.5
    ious = []
    for c in range(logits.shape[1]):
        inter = (pred[:, c] & gt[:, c]).sum().float()
        union = (pred[:, c] | gt[:, c]).sum().float()
        ious.append(1.0 if union == 0 else float(inter / union))
    return ious


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--split", default="mini_train")
    p.add_argument("--raster-root", type=Path, default=Path("data/bev_rasters/mini_train"))
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-z", type=float, default=-0.5)
    p.add_argument("--max-z", type=float, default=3.0)
    p.add_argument("--num-z", type=int, default=8)
    p.add_argument("--min-d", type=float, default=4.0)
    p.add_argument("--max-d", type=float, default=45.0)
    p.add_argument("--D", type=int, default=41)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = NuScenesBEVDataset(
        NuScenesConfig(
            dataroot=args.dataroot,
            version=args.version,
            split=args.split,
            cameras=CAMERAS,
            load_lidar=False,
            load_annotations=False,
            bev_raster_root=args.raster_root,
        )
    )
    batch = collate_fn([ds[i] for i in range(args.batch)])
    batch.to(device)  # in-place; batch.*.to() moves tensors but does not return self
    images = batch.camera_batch
    target = batch.bev_raster_batch.data
    labels = ds.raster_store.layer_names
    print(f"overfitting {args.batch} frames on {device}; layers={labels}")

    model = CameraBEVSeg(
        ds.raster_store.spec, labels, args.min_z, args.max_z, args.num_z,
        args.min_d, args.max_d, args.D,
    ).to(device)
    seg_loss = SegLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    init_loss = None
    for step in range(args.steps):
        opt.zero_grad()
        logits = model(images)
        loss = seg_loss(logits, target)
        loss.backward()
        opt.step()
        if init_loss is None:
            init_loss = loss.item()
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            with torch.no_grad():
                ious = per_class_iou(logits, target)
            iou_str = "  ".join(f"{n}={v:.3f}" for n, v in zip(labels, ious))
            print(f"  step {step:4d}  loss {loss.item():.4f}   IoU  {iou_str}")

    with torch.no_grad():
        final = per_class_iou(model(images), target)
    print("\nfinal per-class IoU:", {n: round(v, 3) for n, v in zip(labels, final)})
    # Coverage caps IoU/loss below 1/0, so judge on the trend: a real pipeline
    # cuts the loss substantially and lifts IoU well off the floor.
    learns = loss.item() < 0.7 * init_loss and max(final) > 0.3
    print(f"loss {init_loss:.3f} -> {loss.item():.3f}")
    print("VERDICT:", "pipeline learns ✓" if learns else "SUSPECT — loss/IoU barely moved (wiring bug?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
