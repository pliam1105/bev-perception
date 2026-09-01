#!/usr/bin/env python3
"""Per-class BEV IoU for the three warmup checkpoints on nuScenes-mini.

Accumulates dataset-level (micro) intersection and union per layer over a split,
at a fixed threshold, and prints a variant x class table. Micro IoU
(sum of intersections / sum of unions across frames) is the standard BEV-seg
report, so a few large frames don't get averaged away by many empty ones.

    python scripts/eval_iou.py --split mini_val --store data/bev_rasters/mini_val

Model loading and geometry come from scripts/make_comparison_video.py so the
architectures and checkpoints match the video exactly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, collate_fn  # noqa: E402
from bev.data.nuscenes_dataset import SampleBatch  # noqa: E402
from bev.raster import BEVRasterStore  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from make_comparison_video import build_models, predict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--split", default="mini_val")
    ap.add_argument("--store", type=Path, default=Path("data/bev_rasters/mini_val"))
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = BEVRasterStore.open(args.store)
    spec, labels = store.spec, store.layer_names
    models = build_models(spec, labels, device)

    ds = NuScenesBEVDataset(NuScenesConfig(
        dataroot=args.dataroot, version=args.version, split=args.split, cameras=CAMERAS,
        load_lidar=False, load_annotations=False,
        bev_raster_root=args.store, require_files=True,
    ))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
                        num_workers=2, pin_memory=True)
    print(f"[{args.split}] {len(ds)} keyframes  thresh={args.thresh}", flush=True)

    C = len(labels)
    inter = {name: np.zeros(C, dtype=np.int64) for name in models}
    union = {name: np.zeros(C, dtype=np.int64) for name in models}

    for i, sample in enumerate(loader):
        sample: SampleBatch
        cam = sample.camera_batch.to(device)
        gt = sample.bev_raster_batch.data[0].cpu().numpy() > 0.5  # (C, ny, nx) bool
        for name, m in models.items():
            pred = predict(m, cam, args.thresh)  # (C, ny, nx) bool
            for c in range(C):
                inter[name][c] += int(np.logical_and(pred[c], gt[c]).sum())
                union[name][c] += int(np.logical_or(pred[c], gt[c]).sum())
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ds)}", flush=True)

    variant_label = {"nodepth": "v1 projection (no depth)",
                     "depth": "v2 learned depth",
                     "scatter": "v3 forward scatter"}
    width = max(len(v) for v in variant_label.values())
    header = f"{'variant':<{width}} | " + " | ".join(f"{n:>12}" for n in labels) + " |    mean"
    print("\n" + header)
    print("-" * len(header))
    for name in ("nodepth", "depth", "scatter"):
        ious = union[name].astype(np.float64)
        ious = np.where(union[name] > 0, inter[name] / np.maximum(union[name], 1), float("nan"))
        cells = " | ".join(f"{v:12.4f}" for v in ious)
        print(f"{variant_label[name]:<{width}} | {cells} | {np.nanmean(ious):7.4f}")
    print(f"\nmicro IoU on {args.split} ({len(ds)} frames), threshold {args.thresh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
