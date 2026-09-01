"""Combined camera-to-BEV pipeline, stitching the backbone and lift sub-figures.

Real camera frame -> frozen ResNet-18 + FPN backbone -> swappable lift (v1/v2/v3)
-> BEV feature -> BEVSeg head -> BEV logits -> prediction; SegLoss (Focal + Dice)
against the rasterized BEV ground truth, evaluated by per-class IoU.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

from arch_common import (ENC, DEC, HEAD, LOGIT, LOSS, GT, IMG, arrow, feat_sheets,
                         label, new_ax, rbox, save)

ASSETS = Path(__file__).resolve().parent / "assets"


def img_tile(ax, path, cx, cy, w, h, edge, title=None, z=3):
    ax.imshow(mpimg.imread(path), extent=[cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2],
              aspect="auto", zorder=z)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="none",
                 edgecolor=edge, lw=1.6, zorder=z + 0.1))
    if title:
        label(ax, cx, cy + h / 2 + 0.16, title, fs=8.2)


def main():
    fig, ax = new_ax(15.5, 5.4)
    Y = 0.8

    x_in, x_bb, x_lift, x_bev, x_head, x_log, x_out = 0.6, 3.0, 5.4, 7.7, 9.6, 11.2, 13.2

    img_tile(ax, ASSETS / "cam_front.jpg", x_in, Y, 1.7, 1.05, IMG[1],
             title="input x6  3x448x800")

    bb = feat_sheets(ax, x_bb, Y, 0.95, 0.78, 8, *ENC)
    label(ax, x_bb, bb["top"] + 0.24, "ResNet-18 + FPN", fs=8.5, weight="bold")
    label(ax, x_bb, bb["bottom"] - 0.24, "frozen", fs=8, style="italic", color="#6a6a6a")
    label(ax, x_bb, bb["bottom"] - 0.52, "128x112x200", fs=7.8)

    lift = rbox(ax, x_lift, Y, 1.35, 0.95, "BEV Lift\n(swappable)", DEC[0], DEC[1], fs=9)
    # three interchangeable variants below
    chips = [("v1  no-depth", 0), ("v2  depth  (best)", 1), ("v3  scatter", 2)]
    cy0 = -1.5
    for txt, kk in chips:
        col = (DEC[0], DEC[1]) if kk != 1 else ("#cfe8cf", "#2f6f2f")
        rbox(ax, x_lift, cy0 - kk * 0.55, 2.0, 0.42, txt, col[0], col[1], fs=8)
    arrow(ax, (x_lift, lift["bottom"]), (x_lift, cy0 + 0.24), ls=(0, (3, 3)), color=DEC[1], lw=1.2)
    label(ax, x_lift + 1.35, cy0 - 0.55, "one of", fs=8, style="italic", color=DEC[1])

    bev = feat_sheets(ax, x_bev, Y, 0.92, 0.92, 6, *DEC)
    label(ax, x_bev, Y, "BEV\nfeat", fs=8, weight="bold")
    label(ax, x_bev, bev["bottom"] - 0.22, "128x200x200", fs=7.8)

    head = rbox(ax, x_head, Y, 1.15, 0.95, "BEVSeg\nhead", HEAD[0], HEAD[1], fs=8.5)
    log = feat_sheets(ax, x_log, Y, 0.8, 0.8, 2, *LOGIT)
    label(ax, x_log, Y, "logits", fs=7.5, weight="bold")
    label(ax, x_log, log["bottom"] - 0.22, "2x200x200", fs=7.8)

    img_tile(ax, ASSETS / "bev_pred_v2.png", x_out, Y, 1.1, 1.1, "#2f8f86",
             title="BEV prediction (v2)")

    # flow arrows
    arrow(ax, (x_in + 0.85, Y), (bb["left"] - 0.02, Y))
    arrow(ax, (bb["right"], Y), (lift["left"], Y)); label(ax, (bb["right"] + lift["left"]) / 2, Y + 0.2, "lift", fs=8, style="italic")
    arrow(ax, (lift["right"], Y), (bev["left"] - 0.02, Y))
    arrow(ax, (bev["right"], Y), (head["left"], Y)); label(ax, (bev["right"] + head["left"]) / 2, Y + 0.22, "conv", fs=8, style="italic")
    arrow(ax, (head["right"], Y), (log["left"] - 0.02, Y))
    arrow(ax, (log["right"], Y), (x_out - 0.6, Y)); label(ax, (log["right"] + x_out - 0.6) / 2, Y + 0.2, "sigmoid", fs=7.5, style="italic")

    # loss + GT + metric (lower lane)
    yl = -1.9
    gt = img_tile(ax, ASSETS / "bev_gt.png", x_head - 1.0, yl, 1.0, 1.0, GT[1], title=None)
    label(ax, x_head - 1.0, yl - 0.62, "BEV GT  2x200x200", fs=7.8)
    loss = rbox(ax, x_head + 0.9, yl, 1.5, 0.8, "SegLoss\nFocal + Dice", LOSS[0], LOSS[1], fs=8.5)
    arrow(ax, (x_head - 0.5, yl), (loss["left"], yl))
    arrow(ax, (x_log, log["bottom"] - 0.02), (loss["cx"] + 0.3, loss["top"]), ls=(0, (3, 3)), rad=-0.2)
    arrow(ax, (loss["right"], yl), (loss["right"] + 1.0, yl), color="#999")
    label(ax, loss["right"] + 1.4, yl, "IoU", fs=10, style="italic", color="#555")

    ax.set_xlim(-0.5, x_out + 1.4)
    ax.set_ylim(-2.9, 1.7)
    save(fig, ax, "architecture")


if __name__ == "__main__":
    main()
