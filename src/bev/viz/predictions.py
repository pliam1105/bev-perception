"""BEV prediction visualization: model output vs ground-truth raster.

Renders a handful of frames as top-down RGB panels (one color per layer),
predicted BEV beside ground truth, for eyeballing training progress each epoch.
Prediction and ground truth are rendered with the same orientation so they are
directly comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from bev.data.nuscenes_dataset import CameraDataBatch

# Per-layer RGB, matched to the live Foxglove colors.
_LAYER_COLORS: dict[str, tuple[int, int, int]] = {
    "drivable_area": (70, 160, 235),
    "vehicle": (240, 140, 40),
}
_FALLBACK = (200, 200, 200)
_BG = (20, 20, 26)


def _bev_rgb(mask_by_layer: np.ndarray, layer_names: Sequence[str]) -> np.ndarray:
    """(C, H, W) occupancy -> (H, W, 3) uint8; later layers drawn on top."""
    _, h, w = mask_by_layer.shape
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = _BG
    for c, name in enumerate(layer_names):
        img[mask_by_layer[c] > 0.5] = _LAYER_COLORS.get(name, _FALLBACK)
    return np.flipud(img)  # place higher grid indices at the top; same for pred + GT


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(inter / union)


@torch.no_grad()
def plot_bev_predictions(
    model: torch.nn.Module,
    camera_batch: CameraDataBatch,
    target: torch.Tensor,
    layer_names: Sequence[str],
    *,
    n: int = 5,
    thresh: float = 0.5,
    save_path: Path | str | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Render `n` frames as prediction-vs-ground-truth BEV panels.

    Inference runs one frame at a time so this stays within memory regardless of
    `n`. Model is left in the mode it started in. Returns the matplotlib figure.
    """
    import matplotlib

    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    was_training = model.training
    model.eval()

    n = min(n, target.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(6.2, 3.1 * n), squeeze=False)
    for i in range(n):
        single = CameraDataBatch(
            channels=camera_batch.channels,
            images=camera_batch.images[i : i + 1],
            bev2pixel=camera_batch.bev2pixel[i : i + 1],
        )
        prob = torch.sigmoid(model(single))[0].cpu().numpy()  # (C, nx, ny)
        pred = prob >= thresh
        gt = target[i].cpu().numpy()

        ious = [_iou(pred[c], gt[c] > 0.5) for c in range(len(layer_names))]
        iou_str = "  ".join(f"{n_}={v:.2f}" for n_, v in zip(layer_names, ious))

        for ax, arr, tag in ((axes[i][0], pred, "pred"), (axes[i][1], gt, "GT")):
            ax.imshow(_bev_rgb(arr, layer_names))
            ax.plot(arr.shape[2] / 2, arr.shape[1] / 2, "r+", ms=9)  # ego
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i][0].set_ylabel(f"frame {i}", fontsize=9)
        axes[i][0].set_title(f"pred   {iou_str}", fontsize=9)
        axes[i][1].set_title("ground truth", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110)
    if show:
        plt.show()

    if was_training:
        model.train()
    return fig


@torch.no_grad()
def plot_train_val_predictions(
    model: torch.nn.Module,
    train_camera: CameraDataBatch,
    train_target: torch.Tensor,
    val_camera: CameraDataBatch,
    val_target: torch.Tensor,
    layer_names: Sequence[str],
    *,
    n: int = 5,
    thresh: float = 0.5,
    save_path: Path | str | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Render `n` train and `n` val frames side by side: for each row,
    [train pred | train GT | val pred | val GT]. Inference runs one frame at a
    time to stay within memory. Returns the matplotlib figure.
    """
    import matplotlib

    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    was_training = model.training
    model.eval()

    groups = (("train", train_camera, train_target), ("val", val_camera, val_target))
    n = min(n, train_target.shape[0], val_target.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(12.4, 3.1 * n), squeeze=False)
    for i in range(n):
        for g, (tag, cam, tgt) in enumerate(groups):
            single = CameraDataBatch(
                channels=cam.channels,
                images=cam.images[i : i + 1],
                bev2pixel=cam.bev2pixel[i : i + 1],
            )
            prob = torch.sigmoid(model(single))[0].cpu().numpy()
            pred = prob >= thresh
            gt = tgt[i].cpu().numpy()
            ious = [_iou(pred[c], gt[c] > 0.5) for c in range(len(layer_names))]
            iou_str = " ".join(f"{n_[:4]}={v:.2f}" for n_, v in zip(layer_names, ious))

            ax_pred, ax_gt = axes[i][2 * g], axes[i][2 * g + 1]
            ax_pred.imshow(_bev_rgb(pred, layer_names))
            ax_gt.imshow(_bev_rgb(gt, layer_names))
            for ax in (ax_pred, ax_gt):
                ax.plot(pred.shape[2] / 2, pred.shape[1] / 2, "r+", ms=8)
                ax.set_xticks([]); ax.set_yticks([])
            ax_pred.set_title(f"{tag} pred  {iou_str}", fontsize=8)
            ax_gt.set_title(f"{tag} GT", fontsize=8)
        axes[i][0].set_ylabel(f"frame {i}", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110)
    if show:
        plt.show()

    if was_training:
        model.train()
    return fig
