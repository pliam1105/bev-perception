#!/usr/bin/env python3
"""Surround-view video of variant 2's learned depth distribution.

For each keyframe, runs the depth model's backbone + FeatureDepthPredictor on all
six cameras, turns the per-pixel softmax distribution over D=42 depth bins into an
expected-depth map (sum_d d * p_d), and lays it beside the input image in the
canonical surround layout. Streams to mp4.

    python scripts/make_depth_video.py --out viz/depth-distribution.mp4 --fps 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, collate_fn  # noqa: E402
from bev.data.nuscenes_dataset import SampleBatch  # noqa: E402
from bev.models import CameraBEVSeg  # noqa: E402
from bev.raster import BEVRasterStore  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# geometry (scripts/warmup.py)
MIN_HEIGHT, MAX_HEIGHT, NUM_HEIGHTS = -0.5, 3.0, 4
MIN_DEPTH, MAX_DEPTH, NUM_DEPTHS = 0.0, 42.0, 42

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_STORE_SPLIT = {"mini_train": "mini_train", "mini_val": "mini_val"}


def load_depth_model(spec, labels, device):
    model = CameraBEVSeg(spec, labels, MIN_HEIGHT, MAX_HEIGHT, NUM_HEIGHTS,
                         MIN_DEPTH, MAX_DEPTH, NUM_DEPTHS)
    ckpt = REPO_ROOT / "models" / "warmup-depth" / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    print(f"loaded depth <- {ckpt}", flush=True)
    return model.to(device).eval()


@torch.no_grad()
def expected_depth(model, images, d_values):
    """images (N,3,H,W) normalized -> (N, H/4, W/4) expected depth in metres."""
    feat = model.bevlift.backbone(images)                 # (N,128,H/4,W/4)
    logits = model.bevlift.depth_predictor(feat)          # (N,D,H/4,W/4)
    p = torch.softmax(logits, dim=1)
    return (p * d_values.view(1, -1, 1, 1)).sum(1)        # (N,H/4,W/4)


def denorm(img):
    return (img.cpu() * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--stores", type=Path, nargs="+",
                    default=[Path("data/bev_rasters/mini_train"),
                             Path("data/bev_rasters/mini_val")])
    ap.add_argument("--out", type=Path, default=Path("viz/depth-distribution.mp4"))
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store0 = BEVRasterStore.open(args.stores[0])
    model = load_depth_model(store0.spec, store0.layer_names, device)
    d_values = torch.arange(NUM_DEPTHS, device=device).float() * (
        (MAX_DEPTH - MIN_DEPTH) / (NUM_DEPTHS - 1)) + MIN_DEPTH

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=2)
    cmap = cm.get_cmap(args.cmap)

    n = 0
    for store_path in args.stores:
        split = _STORE_SPLIT.get(store_path.name, store_path.name)
        ds = NuScenesBEVDataset(NuScenesConfig(
            dataroot=args.dataroot, version=args.version, split=split, cameras=CAMERAS,
            load_lidar=False, load_annotations=False,
            bev_raster_root=store_path, require_files=True))
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True)
        print(f"[{split}] {len(ds)} keyframes", flush=True)
        for i, sample in enumerate(loader):
            sample: SampleBatch
            images = sample.camera_batch.images[0].to(device)     # (6,3,448,800)
            depth = expected_depth(model, images, d_values).cpu().numpy()  # (6,112,200)
            token = ds.sample_tokens[i]
            frame = render(images, depth, ds.scene_names[token], split, i, cmap)
            writer.append_data(frame)
            n += 1
            if n % 25 == 0:
                print(f"  {n} frames", flush=True)
            if args.limit and n >= args.limit:
                break
        if args.limit and n >= args.limit:
            break

    writer.close()
    print(f"wrote {n} frames -> {args.out}", flush=True)
    return 0


def render(images, depth, scene, split, i, cmap):
    fig, axes = plt.subplots(2, 6, figsize=(16.5, 5.4), dpi=110)
    for cam_idx, ch in enumerate(CAMERAS):
        r, k = divmod(cam_idx, 3)
        ax_in, ax_d = axes[r][2 * k], axes[r][2 * k + 1]
        ax_in.imshow(denorm(images[cam_idx]), aspect="auto")
        ax_in.set_title(ch.replace("CAM_", ""), fontsize=8.5, fontweight="bold")
        im = ax_d.imshow(depth[cam_idx], cmap=cmap, vmin=MIN_DEPTH, vmax=MAX_DEPTH,
                         aspect="auto")
        ax_d.set_title("E[depth]", fontsize=8.5)
        for ax in (ax_in, ax_d):
            ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=0.012, right=0.925, top=0.88, bottom=0.03,
                        wspace=0.06, hspace=0.22)
    cax = fig.add_axes([0.935, 0.12, 0.008, 0.72])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("expected depth (m)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"variant 2 — learned depth distribution   |   nuScenes-mini "
                 f"{split}  {scene}  frame {i}", fontsize=12)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


if __name__ == "__main__":
    raise SystemExit(main())
