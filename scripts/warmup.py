#!/usr/bin/env python3
"""Frozen-backbone warmup for one BEV model, matching the notebook hyperparameters.

Trains only the FPN + seg head (ResNet stays frozen). Optimizer, loss, grad-clip,
early-stopping and per-epoch viz mirror the notebook's warmup cell. Runs 15 epochs
with patience 5; if it reaches the cap while val loss is still dropping, the cap
lifts to 50 (patience 5). Per-epoch checkpoints, prediction montages and a
losses.json land in the model's own folders.

    python scripts/warmup.py --model nodepth --train-batch 2
    python scripts/warmup.py --model scatter --train-batch 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, collate_fn  # noqa: E402
from bev.data.nuscenes_dataset import SampleBatch  # noqa: E402
from bev.models import CameraBEVSeg, CameraBEVSegProjection, CameraBEVSegScatter, SegLoss  # noqa: E402
from bev.viz.predictions import plot_train_val_predictions  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# model config (notebook cell 15)
MIN_HEIGHT, MAX_HEIGHT, NUM_HEIGHTS = -0.5, 3.0, 4
MIN_DEPTH, MAX_DEPTH, NUM_DEPTHS = 0.0, 42.0, 42
# loss config (notebook cell 16)
GAMMA, ALPHAS, EPS, LAMBDA = 2, (0.25, 0.5), 1e-5, 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["depth", "nodepth", "scatter"])
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--train-raster", type=Path, default=Path("data/bev_rasters/trainval_train"))
    p.add_argument("--val-raster", type=Path, default=Path("data/bev_rasters/trainval_val"))
    p.add_argument("--train-batch", type=int, default=1)
    p.add_argument("--val-batch", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=15)
    p.add_argument("--extend-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--viz-dir", type=Path, default=None)
    return p.parse_args()


def build_model(kind: str, spec, labels):
    if kind == "depth":
        return CameraBEVSeg(spec, labels, MIN_HEIGHT, MAX_HEIGHT, NUM_HEIGHTS,
                            MIN_DEPTH, MAX_DEPTH, NUM_DEPTHS)
    if kind == "nodepth":
        return CameraBEVSegProjection(spec, labels, MIN_HEIGHT, MAX_HEIGHT, NUM_HEIGHTS)
    return CameraBEVSegScatter(spec, labels, MIN_DEPTH, MAX_DEPTH, NUM_DEPTHS)


def loader(dataroot, version, split, raster, batch, workers):
    ds = NuScenesBEVDataset(NuScenesConfig(
        dataroot=dataroot, version=version, split=split, cameras=CAMERAS,
        load_lidar=False, load_annotations=False, bev_raster_root=raster, require_files=True,
    ))
    return ds, DataLoader(ds, batch_size=batch, shuffle=split.endswith("train"),
                          collate_fn=collate_fn, num_workers=workers, pin_memory=True)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or Path(f"models/warmup-{args.model}")
    viz_dir = args.viz_dir or Path(f"viz/warmup-{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, train_loader = loader(args.dataroot, args.version, args.train_split,
                                     args.train_raster, args.train_batch, args.workers)
    val_set, val_loader = loader(args.dataroot, args.version, args.val_split,
                                 args.val_raster, args.val_batch, args.workers)
    labels = train_set.raster_store.layer_names
    spec = train_set.raster_store.spec

    model = build_model(args.model, spec, labels).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={args.model}  device={device}  train={len(train_set)} val={len(val_set)}  "
          f"trainable params={n_train}  batch={args.train_batch}", flush=True)

    seg_loss = SegLoss(GAMMA, ALPHAS, EPS, LAMBDA).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-12)

    viz_train = collate_fn([train_set[i] for i in range(min(5, len(train_set)))]); viz_train.to(device)
    viz_val = collate_fn([val_set[i] for i in range(min(5, len(val_set)))]); viz_val.to(device)

    hist: dict[str, list[float]] = {"train": [], "val": []}
    best_val = float("inf")
    patience = 0
    cap = args.max_epochs
    epoch = 0
    while epoch < cap:
        model.train()
        tl = 0.0
        for sample in train_loader:
            sample: SampleBatch
            x = sample.camera_batch.to(device)
            target = sample.bev_raster_batch.data.to(device)
            opt.zero_grad()
            loss = seg_loss(model(x), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        hist["train"].append(tl)
        hist["val"].append(vl)
        (out_dir / "losses.json").write_text(json.dumps(hist, indent=2))
        print(f"epoch {epoch:3d}  train {tl:.4f}  val {vl:.4f}  patience {patience}  cap {cap}", flush=True)

        torch.cuda.empty_cache()
        plot_train_val_predictions(
            model, viz_train.camera_batch, viz_train.bev_raster_batch.data,
            viz_val.camera_batch, viz_val.bev_raster_batch.data, labels, n=5,
            save_path=str(viz_dir / f"epoch_{epoch}.png"),
            title=f"{args.model}  epoch {epoch}  train {tl:.3f} | val {vl:.3f}",
        )
        torch.save(model.state_dict(), out_dir / f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), out_dir / "last.pt")
        if vl < best_val:
            best_val, patience = vl, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stopping at epoch {epoch} (best val {best_val:.4f})", flush=True)
                break

        # reached the 15-epoch cap while still improving -> give it room to 50
        still_dropping = len(hist["val"]) >= 2 and hist["val"][-1] < hist["val"][-2]
        if epoch == cap - 1 and cap < args.extend_epochs and still_dropping:
            cap = args.extend_epochs
            print(f"still improving at cap; extending to {cap} epochs", flush=True)
        epoch += 1

    print(f"done {args.model}. best val {best_val:.4f} -> {out_dir/'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
