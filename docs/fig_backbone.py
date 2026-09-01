"""Backbone: frozen ResNet-18 bottom-up path + FPN top-down path with lateral skips.

ResNetBackbone (src/bev/models.py): a frozen ImageNet ResNet-18 gives C1..C4; a
top-down pathway upsamples, adds a lateral skip and a 3x3 smooth to yield P5..P7,
the stride-4 / 128-channel map (P7) that the lift consumes. Feature maps are drawn
as stacked sheets (channels = stack depth); the input is a real camera frame.
"""
from __future__ import annotations

from pathlib import Path

from arch_common import (ENC, DEC, IMG, arrow, feat_sheets, img_tile, label,
                         new_ax, oplus, save)

ASSETS = Path(__file__).resolve().parent / "assets"
XS = {4: 2.2, 8: 5.0, 16: 7.8, 32: 10.6}
Y_ENC, Y_DEC = 3.7, 0.6
SCALE = {4: 1.0, 8: 0.80, 16: 0.63, 32: 0.50}
SHEETS = {64: 4, 128: 6, 256: 8, 512: 10}


def fmap(ax, stride, y, C, HW, name, color):
    s = SCALE[stride]
    g = feat_sheets(ax, XS[stride], y, 1.25 * s, 1.0 * s, SHEETS[C], *color)
    label(ax, g["front_cx"], g["front_cy"], name, fs=9.5, weight="bold")
    label(ax, XS[stride], g["bottom"] - 0.24, f"{C}x{HW}", fs=8.2)
    return g


def main():
    fig, ax = new_ax(12.6, 5.6)

    img_tile(ax, ASSETS / "cam_front.jpg", 0.25, Y_ENC, 1.5, 0.86, IMG[1],
             title="image  3x448x800")

    c1 = fmap(ax, 4,  Y_ENC, 64,  "112x200", "C1", ENC)
    c2 = fmap(ax, 8,  Y_ENC, 128, "56x100",  "C2", ENC)
    c3 = fmap(ax, 16, Y_ENC, 256, "28x50",   "C3", ENC)
    c4 = fmap(ax, 32, Y_ENC, 512, "14x25",   "C4", ENC)

    arrow(ax, (1.02, Y_ENC), (c1["left"] - 0.05, Y_ENC))       # image -> C1
    for a, b in ((c1, c2), (c2, c3), (c3, c4)):
        arrow(ax, (a["right"], Y_ENC), (b["left"], Y_ENC))
    label(ax, (c1["right"] + c2["left"]) / 2, Y_ENC + 0.30, "stride 2", fs=8, style="italic")
    label(ax, XS[4] - 1.6, Y_ENC + 1.2, "frozen ResNet-18  (bottom-up)", fs=9.5,
          style="italic", ha="left", color="#5a5a5a")

    p5 = fmap(ax, 16, Y_DEC, 256, "28x50",  "P5", DEC)
    p6 = fmap(ax, 8,  Y_DEC, 128, "56x100", "P6", DEC)
    p7 = fmap(ax, 4,  Y_DEC, 128, "112x200", "P7", DEC)
    label(ax, XS[32], Y_DEC, "= C4", fs=8.5, style="italic", color="#5a5a5a")

    arrow(ax, (c4["cx"], c4["bottom"] - 0.05), (XS[16] + 0.9, Y_DEC), rad=0.15)
    for a, b in ((p5, p6), (p6, p7)):
        arrow(ax, (a["left"], Y_DEC), (b["right"], Y_DEC))
        label(ax, (a["left"] + b["right"]) / 2, Y_DEC + 0.02, "up x2", fs=7.5, style="italic")

    for enc, dec, stride in ((c3, p5, 16), (c2, p6, 8), (c1, p7, 4)):
        ox, oy = XS[stride], (enc["bottom"] + dec["top"]) / 2
        oplus(ax, ox, oy)
        arrow(ax, (enc["cx"], enc["bottom"] - 0.05), (ox, oy + 0.16), color="#8a8a8a", ls=(0, (4, 3)))
        arrow(ax, (ox, oy - 0.16), (dec["cx"], dec["top"] + 0.05))
        label(ax, ox + 0.6, oy, "1x1", fs=7.5, style="italic", color="#8a8a8a")

    label(ax, XS[8], Y_DEC - 1.55, "FPN top-down + skips  (trained)", fs=9.5,
          style="italic", ha="center", color="#3a6a3a")

    # output: a subset of the real stride-4 output channels, on the way to the lift
    arrow(ax, (p7["left"], Y_DEC), (-0.42, Y_DEC), lw=1.8)
    ch_pos = [(-2.05, 0.98), (-1.4, 0.98), (-0.75, 0.98),
              (-2.05, 0.22), (-1.4, 0.22), (-0.75, 0.22)]
    for n, (cx, cy) in enumerate(ch_pos):
        img_tile(ax, ASSETS / f"backbone_ch_{n}.png", cx, cy, 0.6, 0.34, DEC[1])
    label(ax, -1.4, 1.35, "output: 6 of 128 channels", fs=9, weight="bold")
    label(ax, -1.4, -0.22, "128x112x200   ->  to lift", fs=8.5, style="italic")

    ax.set_xlim(-2.7, XS[32] + 1.7)
    ax.set_ylim(Y_DEC - 2.0, Y_ENC + 1.7)
    save(fig, ax, "backbone")


if __name__ == "__main__":
    main()
