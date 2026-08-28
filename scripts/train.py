#!/usr/bin/env python3
"""Camera->BEV segmentation training (runnable copy of the notebook loop).

Loads warmup weights, finetunes the whole model with discriminative LRs
(pretrained backbone gentle, new heads faster), grad clipping, and per-epoch
train+val prediction montages. Early-stops on val loss.

    python scripts/train.py \
        --version v1.0-trainval --train-split train --val-split val \
        --train-raster data/bev_rasters/trainval_train \
        --val-raster   data/bev_rasters/trainval_val \
        --warmup models/frozen-warmup/camera_bev_seg_best.pt \
        --out models/trainval --viz viz_trainval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, collate_fn  # noqa: E402
from bev.data.nuscenes_dataset import SampleBatch  # noqa: E402
from bev.models import CameraBEVSeg, SegLoss  # noqa: E402
from bev.viz.predictions import plot_train_val_predictions  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--train-raster", type=Path, required=True)
    p.add_argument("--val-raster", type=Path, required=True)
    p.add_argument("--warmup", type=Path, default=Path("models/frozen-warmup/camera_bev_seg_best.pt"))
    p.add_argument("--out", type=Path, default=Path("models/trainval"))
    p.add_argument("--viz", type=Path, default=Path("viz_trainval"))
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-new", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    # model config (must match the warmup checkpoint)
    p.add_argument("--min-z", type=float, default=-0.5)
    p.add_argument("--max-z", type=float, default=3.0)
    p.add_argument("--num-z", type=int, default=4)
    p.add_argument("--min-d", type=float, default=0.0)
    p.add_argument("--max-d", type=float, default=42.0)
    p.add_argument("--D", type=int, default=42)
    return p.parse_args()


def build_loader(dataroot, version, split, raster_root, batch, workers):
    ds = NuScenesBEVDataset(
        NuScenesConfig(
            dataroot=dataroot, version=version, split=split,
            cameras=CAMERAS, load_lidar=False, load_annotations=False,
            bev_raster_root=raster_root, require_files=True,
        )
    )
    loader = DataLoader(
        ds, batch_size=batch, shuffle=(split.endswith("train")),
        collate_fn=collate_fn, num_workers=workers, pin_memory=True,
    )
    return ds, loader


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    args.viz.mkdir(parents=True, exist_ok=True)

    train_set, train_loader = build_loader(
        args.dataroot, args.version, args.train_split, args.train_raster, args.batch, args.workers
    )
    val_set, val_loader = build_loader(
        args.dataroot, args.version, args.val_split, args.val_raster, args.batch, args.workers
    )
    labels = train_set.raster_store.layer_names
    print(f"device={device}  train={len(train_set)} keyframes  val={len(val_set)} keyframes  layers={labels}", flush=True)

    model = CameraBEVSeg(
        train_set.raster_store.spec, labels,
        args.min_z, args.max_z, args.num_z, args.min_d, args.max_d, args.D,
    ).to(device)
    if args.warmup and args.warmup.is_file():
        missing, unexpected = model.load_state_dict(
            torch.load(args.warmup, map_location=device, weights_only=True), strict=False
        )
        print(f"loaded warmup {args.warmup}  (missing={len(missing)} unexpected={len(unexpected)})", flush=True)

    # discriminative LRs: pretrained resnet gentle, everything new faster
    resnet_ids = {id(p) for p in model.bevlift.backbone.resnet.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) in resnet_ids and p.requires_grad]
    new_params = [p for p in model.parameters() if id(p) not in resnet_ids and p.requires_grad]
    opt = torch.optim.Adam(
        [{"params": backbone_params, "lr": args.lr_backbone},
         {"params": new_params, "lr": args.lr_new}],
        weight_decay=1e-4,
    )
    seg_loss = SegLoss().to(device)

    # fixed viz frames (train + val)
    viz_train = collate_fn([train_set[i] for i in range(min(5, len(train_set)))]); viz_train.to(device)
    viz_val = collate_fn([val_set[i] for i in range(min(5, len(val_set)))]); viz_val.to(device)

    best_val = float("inf")
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        tl = 0.0
        for sample in train_loader:
            sample: SampleBatch
            x = sample.camera_batch.to(device)
            target = sample.bev_raster_batch.data.to(device)
            opt.zero_grad()
            loss = seg_loss(model(x), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tl += loss.item()
        tl /= max(1, len(train_loader))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for sample in val_loader:
                x = sample.camera_batch.to(device)
                target = sample.bev_raster_batch.data.to(device)
                vl += seg_loss(model(x), target).item()
        vl /= max(1, len(val_loader))
        print(f"epoch {epoch:3d}  train {tl:.4f}  val {vl:.4f}  patience {patience}", flush=True)

        torch.cuda.empty_cache()
        plot_train_val_predictions(
            model, viz_train.camera_batch, viz_train.bev_raster_batch.data,
            viz_val.camera_batch, viz_val.bev_raster_batch.data, labels,
            n=5, save_path=str(args.viz / f"epoch_{epoch}.png"),
            title=f"epoch {epoch}  train {tl:.3f} | val {vl:.3f}",
        )
        torch.save(model.state_dict(), args.out / "last.pt")
        if vl < best_val:
            best_val = vl
            patience = 0
            torch.save(model.state_dict(), args.out / "best.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stopping at epoch {epoch} (best val {best_val:.4f})", flush=True)
                break

    print(f"done. best val {best_val:.4f}  -> {args.out/'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
