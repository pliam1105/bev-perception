"""Prepare real nuScenes-mini VAL assets for the architecture figures.

Picks one mini_val keyframe (the one with the most vehicle cells, for a legible
top-down map) and writes, under docs/assets/:
  cam_front.jpg     - front camera image
  bev_gt.png        - rasterized BEV ground truth (blue drivable / orange vehicle)
  bev_pred.png      - variant-2 model's BEV segmentation prediction (thresholded)
  depth_front_v2.png, depth_front_v3.png - each model's expected-depth map E[depth]
                       for the front camera (soft-argmax over its D=42 depth bins)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bev.data import CAMERAS, NuScenesBEVDataset, NuScenesConfig, collate_fn  # noqa: E402
from bev.raster import BEVRasterStore  # noqa: E402
from bev.viz.predictions import _bev_rgb  # noqa: E402
from bev.models import CameraBEVSeg, CameraBEVSegScatter, CameraBEVSegProjection  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)
VAL = REPO / "data/bev_rasters/mini_val"

MIN_H, MAX_H, NZ = -0.5, 3.0, 4
MIN_D, MAX_D, D = 0.0, 42.0, 42


def pick_token(store):
    best_tok, best = None, -1
    for f in sorted(VAL.glob("*.npz")):
        veh = float(store.load(f.stem).data[1].sum())
        if veh > best:
            best_tok, best = f.stem, veh
    return best_tok


def save_depth(model, images, path):
    with torch.no_grad():
        feat = model.bevlift.backbone(images)           # (N,128,H/4,W/4)
        logits = model.bevlift.depth_predictor(feat)    # (N,D,H/4,W/4)
        p = torch.softmax(logits, dim=1)
        dvals = torch.arange(D).float() * ((MAX_D - MIN_D) / (D - 1)) + MIN_D
        exp = (p * dvals.view(1, -1, 1, 1)).sum(1)[1].cpu().numpy()  # front camera
    rgba = cm.get_cmap("turbo")(Normalize(MIN_D, MAX_D)(exp))
    im = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
    im = im.resize((im.width * 2, im.height * 2), Image.BILINEAR)
    im.save(path)
    print(f"{path.name}  {im.size}")


def main() -> int:
    store = BEVRasterStore.open(VAL)
    tok = pick_token(store)
    print(f"val sample {tok}")

    ds = NuScenesBEVDataset(NuScenesConfig(
        dataroot=REPO / "data/nuscenes", version="v1.0-mini", split="mini_val",
        cameras=CAMERAS, load_lidar=False, load_annotations=False,
        bev_raster_root=VAL, require_files=True))
    idx = ds.sample_tokens.index(tok)
    batch = collate_fn([ds[idx]])
    images = batch.camera_batch.images[0]              # (6,3,448,800)

    # front camera image + GT raster
    sd = ds.nusc.get("sample_data", ds.nusc.get("sample", tok)["data"]["CAM_FRONT"])
    img = Image.open(REPO / "data/nuscenes" / sd["filename"]).convert("RGB")
    img.thumbnail((520, 520)); img.save(ASSETS / "cam_front.jpg", quality=88)
    Image.fromarray(_bev_rgb(np.asarray(store.load(tok).data), store.layer_names)).save(ASSETS / "bev_gt.png")

    spec, labels = store.spec, store.layer_names
    models = {
        "v1": (CameraBEVSegProjection(spec, labels, MIN_H, MAX_H, NZ), "warmup-nodepth"),
        "v2": (CameraBEVSeg(spec, labels, MIN_H, MAX_H, NZ, MIN_D, MAX_D, D), "warmup-depth"),
        "v3": (CameraBEVSegScatter(spec, labels, MIN_D, MAX_D, D), "warmup-scatter"),
    }
    for k, (m, name) in models.items():
        m.load_state_dict(torch.load(REPO / f"models/{name}/best.pt", map_location="cpu"))
        m.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(m(batch.camera_batch))[0] >= 0.5).float().numpy()
        Image.fromarray(_bev_rgb(pred, labels)).save(ASSETS / f"bev_pred_{k}.png")
        print(f"bev_pred_{k}.png")
    # keep a default prediction (v2) for the combined figure
    Image.open(ASSETS / "bev_pred_v2.png").save(ASSETS / "bev_pred.png")

    save_depth(models["v2"][0], images, ASSETS / "depth_front_v2.png")
    save_depth(models["v3"][0], images, ASSETS / "depth_front_v3.png")

    # a subset of the backbone's stride-4 output channels (front camera)
    with torch.no_grad():
        feat = models["v2"][0].bevlift.backbone(images)[1].numpy()   # (128,112,200)
    idx = np.argsort(feat.reshape(feat.shape[0], -1).var(1))[::-1][:6]
    for n, ci in enumerate(idx):
        a = feat[ci]
        a = (a - a.min()) / (np.ptp(a) + 1e-6)
        rgba = cm.get_cmap("viridis")(a)
        im = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
        im.resize((im.width * 2, im.height * 2), Image.BILINEAR).save(ASSETS / f"backbone_ch_{n}.png")
    print("backbone_ch_0..5.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
