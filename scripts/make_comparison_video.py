#!/usr/bin/env python3
"""Render a top-down BEV video comparing the three warmup models to ground truth.

Walks the nuScenes-mini keyframes in (scene, time) order and, for each frame,
runs every model's best checkpoint and lays the predictions beside the GT raster
as top-down panels: [ ground truth | nodepth | depth | scatter ]. Per-panel IoU
per layer is printed in each title. Frames are streamed into an mp4.

    python scripts/make_comparison_video.py \
        --stores data/bev_rasters/mini_train data/bev_rasters/mini_val \
        --out viz/mini-comparison.mp4 --fps 4

Model geometry (height/depth bins) matches scripts/warmup.py; the BEV grid spec
and layer names are read from the raster store so they line up with the GT.
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
from matplotlib.patches import Patch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, collate_fn  # noqa: E402
from bev.data.nuscenes_dataset import CameraDataBatch, SampleBatch  # noqa: E402
from bev.models import CameraBEVSeg, CameraBEVSegProjection, CameraBEVSegScatter  # noqa: E402
from bev.viz.predictions import _LAYER_COLORS, _bev_rgb, _iou  # noqa: E402
from bev.raster import BEVRasterStore  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# model geometry (scripts/warmup.py)
MIN_HEIGHT, MAX_HEIGHT, NUM_HEIGHTS = -0.5, 3.0, 4
MIN_DEPTH, MAX_DEPTH, NUM_DEPTHS = 0.0, 42.0, 42

# split name -> raster store dir, so each store is walked with its own keyframes
_SPLIT_OF_STORE = {"mini_train": "mini_train", "mini_val": "mini_val"}


def build_models(spec, labels, device):
    """Instantiate the three architectures with matching geometry and load best.pt."""
    specs = {
        "nodepth": CameraBEVSegProjection(spec, labels, MIN_HEIGHT, MAX_HEIGHT, NUM_HEIGHTS),
        "depth": CameraBEVSeg(spec, labels, MIN_HEIGHT, MAX_HEIGHT, NUM_HEIGHTS,
                              MIN_DEPTH, MAX_DEPTH, NUM_DEPTHS),
        "scatter": CameraBEVSegScatter(spec, labels, MIN_DEPTH, MAX_DEPTH, NUM_DEPTHS),
    }
    models = {}
    for name, model in specs.items():
        ckpt = REPO_ROOT / "models" / f"warmup-{name}" / "best.pt"
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        models[name] = model.to(device).eval()
        print(f"loaded {name:8s} <- {ckpt}", flush=True)
    return models


@torch.no_grad()
def predict(model, camera_batch: CameraDataBatch, thresh: float) -> np.ndarray:
    """One frame -> (C, ny, nx) boolean prediction mask."""
    prob = torch.sigmoid(model(camera_batch))[0].cpu().numpy()
    return prob >= thresh


def render_frame(gt: np.ndarray, preds: dict[str, np.ndarray], layer_names,
                 title: str, dpi: int) -> np.ndarray:
    """Lay GT + each model's prediction as top-down panels; return an RGB frame."""
    cols = [("ground truth", gt)] + [(name, preds[name]) for name in preds]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 4.0), dpi=dpi)
    h, w = gt.shape[1], gt.shape[2]
    for ax, (name, arr) in zip(axes, cols):
        ax.imshow(_bev_rgb(arr, layer_names))
        ax.plot(w / 2, h / 2, "r+", ms=9)  # ego
        ax.set_xticks([]); ax.set_yticks([])
        if name == "ground truth":
            ax.set_title(name, fontsize=11, fontweight="bold")
        else:
            ious = [_iou(arr[c], gt[c] > 0.5) for c in range(len(layer_names))]
            iou_str = "  ".join(f"{n[:4]}={v:.2f}" for n, v in zip(layer_names, ious))
            ax.set_title(f"{name}\n{iou_str}", fontsize=10)
    handles = [Patch(facecolor=np.array(_LAYER_COLORS[n]) / 255, label=n) for n in layer_names]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08, wspace=0.05)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--stores", type=Path, nargs="+",
                    default=[Path("data/bev_rasters/mini_train"),
                             Path("data/bev_rasters/mini_val")])
    ap.add_argument("--out", type=Path, default=Path("viz/mini-comparison.mp4"))
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None, help="cap frames (debug)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_store = BEVRasterStore.open(args.stores[0])
    spec, labels = first_store.spec, first_store.layer_names
    models = build_models(spec, labels, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=None)

    n = 0
    for store_path in args.stores:
        split = _SPLIT_OF_STORE.get(store_path.name, store_path.name)
        ds = NuScenesBEVDataset(NuScenesConfig(
            dataroot=args.dataroot, version=args.version, split=split, cameras=CAMERAS,
            load_lidar=False, load_annotations=False,
            bev_raster_root=store_path, require_files=True,
        ))
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True)
        print(f"[{split}] {len(ds)} keyframes from {store_path}", flush=True)
        for i, sample in enumerate(loader):
            sample: SampleBatch
            cam = sample.camera_batch.to(device)
            gt = sample.bev_raster_batch.data[0].cpu().numpy()
            preds = {name: predict(m, cam, args.thresh) for name, m in models.items()}
            token = ds.sample_tokens[i]
            scene = ds.scene_names[token]
            title = f"nuScenes-mini  {split}  {scene}  frame {i}"
            writer.append_data(render_frame(gt, preds, labels, title, args.dpi))
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


if __name__ == "__main__":
    raise SystemExit(main())
