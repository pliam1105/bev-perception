"""Lift-variant mechanism figures (v1 / v2 / v3), paper-style oblique 3D.

Left column (aligned stack): the real front-camera frame, the model's predicted
depth map (v2 / v3 only), and the BEV prediction beside the ground truth. Right:
an oblique BEV grid coloured by that variant's prediction, with the depth that the
lift assigns to each ground cell drawn as a wedge over the camera FOV --

  v1  fixed-height projection, no depth  -> the same depth wedge at all 4 heights.
  v2  learned depth at fixed heights     -> 4 height wedges, the predicted depth
                                            distribution highlighting a band.
  v3  forward scatter                    -> one accumulated 2D depth map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from arch_common import (DEPTH, INK, ground_grid, bev_colors, inset_image,
                         line3d, poly3d, proj, label, new_ax, save)

ASSETS = Path(__file__).resolve().parent / "assets"
R, YF, N = 2.0, 3.6, 20
O = (6.4, 0.5)
APEX = (0.0, -1.0, 0.72)
ZS = [0.0, 0.42, 0.84, 1.26]
FOVH = 0.62
TURBO = cm.get_cmap("turbo")

X0, W, WSQ = 0.015, 0.30, 0.145   # left-stack geometry (figure fraction)


def left_stack(fig, pred_png, depth_png=None):
    """Aligned column: input, (depth), then prediction | ground truth."""
    yrow = 0.165 if depth_png else 0.135
    inset_image(fig, [X0, yrow, WSQ, 0.26], ASSETS / pred_png, title="prediction", edge="#2f8f86")
    inset_image(fig, [X0 + WSQ + 0.012, yrow, WSQ, 0.26], ASSETS / "bev_gt.png",
                title="ground truth", edge="#2f8f86")
    if depth_png:
        inset_image(fig, [X0, 0.475, W, 0.145], ASSETS / depth_png,
                    title="predicted depth  E[d]", edge=DEPTH[1])
        inset_image(fig, [X0, 0.66, W, 0.24], ASSETS / "cam_front.jpg", title="input (front cam)")
    else:
        inset_image(fig, [X0, 0.47, W, 0.30], ASSETS / "cam_front.jpg", title="input (front cam)")


def wedge():
    dx, dy = 2 * R / N, YF / N
    cells, rmax = [], 1e-6
    for i in range(N):
        for j in range(N):
            X = -R + (i + 0.5) * dx
            Y = (j + 0.5) * dy
            dyr = Y - APEX[1]
            r = float(np.hypot(X, dyr))
            infov = dyr > 0 and abs(np.arctan2(X, dyr)) <= FOVH
            if infov:
                rmax = max(rmax, r)
            cells.append((i, j, X, Y, r, infov))
    return cells, rmax, dx, dy


def draw_plane(ax, cells, rmax, dx, dy, Z, alpha_fn):
    for i, j, X, Y, r, infov in cells:
        if not infov:
            continue
        x0, y0 = -R + i * dx, j * dy
        poly3d(ax, [(x0, y0, Z), (x0 + dx, y0, Z), (x0 + dx, y0 + dy, Z), (x0, y0 + dy, Z)],
               facecolor=TURBO(r / rmax), edgecolor="none", alpha=alpha_fn(r), z=4 + Z, o=O)


def scene(ax, pred_png):
    ground_grid(ax, R, YF, N, o=O, colors=bev_colors(ASSETS / pred_png, N),
                edge="#cfcfcf", lw=0.3, z=2)
    poly3d(ax, [(-0.15, APEX[1] - 0.15, APEX[2]), (0.15, APEX[1] - 0.15, APEX[2]),
                (0.0, APEX[1] + 0.11, APEX[2])], facecolor=INK, edgecolor=INK, z=8, o=O)
    for s in (-1, 1):
        far = 1.05 * YF
        line3d(ax, (0, APEX[1], 0.02), (s * far * np.sin(FOVH), APEX[1] + far * np.cos(FOVH), 0.02),
               color="#9a9a9a", lw=0.8, ls=(0, (5, 4)), z=3, o=O)


def height_label(ax):
    p = proj(R * 0.5, YF * 0.92, ZS[-1] + 0.2, O)
    label(ax, p[0], p[1], "4 height bins (z)", fs=8, style="italic", color="#5a5a5a")


def depth_bar(fig, rect=(0.955, 0.16, 0.012, 0.32)):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap=TURBO), cax=cax)
    cb.set_ticks([0.03, 0.97]); cb.set_ticklabels(["near", "far"])
    cb.ax.tick_params(labelsize=7.5); cb.set_label("projected depth", fontsize=8)


def frame():
    fig, ax = new_ax(11.4, 5.6)
    ax.set_xlim(0.0, 11.0); ax.set_ylim(-0.6, 3.4)
    return fig, ax


def v1():
    fig, ax = frame()
    scene(ax, "bev_pred_v1.png")
    # projection rays from the camera to each cell pillar, marking where they meet
    # the pillar at each of the 4 sampled heights (no depth -> a ray hits every cell).
    lat = [-0.85, 0.0, 0.85]
    fwd = [1.1, 2.1, 3.1]
    for X in lat:
        for Y in fwd:
            line3d(ax, (X, Y, ZS[0]), (X, Y, ZS[-1]), color="#8a8a8a", lw=1.3, z=5, o=O)
            for z in ZS:
                line3d(ax, APEX, (X, Y, z), color="#c4c4c4", lw=0.6, z=3, o=O)
                ax.scatter(*proj(X, Y, z, O), s=15, c="#333333", edgecolors="white",
                           linewidths=0.4, zorder=6)
    height_label(ax)
    left_stack(fig, "bev_pred_v1.png")
    save(fig, ax, "lift_v1")


def v2():
    fig, ax = frame()
    scene(ax, "bev_pred_v2.png")
    cells, rmax, dx, dy = wedge()
    r_surf = 0.55 * rmax
    band = lambda r: 0.12 + 0.78 * np.exp(-((r - r_surf) / (0.16 * rmax)) ** 2)
    for Z in ZS:
        draw_plane(ax, cells, rmax, dx, dy, Z, band)
    height_label(ax)
    left_stack(fig, "bev_pred_v2.png", "depth_front_v2.png")
    depth_bar(fig)
    save(fig, ax, "lift_v2")


def v3():
    fig, ax = frame()
    scene(ax, "bev_pred_v3.png")
    cells, rmax, dx, dy = wedge()
    draw_plane(ax, cells, rmax, dx, dy, 0.14, lambda r: 0.62)
    left_stack(fig, "bev_pred_v3.png", "depth_front_v3.png")
    depth_bar(fig)
    save(fig, ax, "lift_v3")


if __name__ == "__main__":
    v1(); v2(); v3()
