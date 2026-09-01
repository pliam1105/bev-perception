"""Shared drawing helpers for the architecture figures (matplotlib primitives)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Circle, Rectangle

# palette: (face, edge)
IMG = ("#cfe0f3", "#3f6ea5")
ENC = ("#e6e6e6", "#7d7d7d")   # frozen encoder
DEC = ("#d3ead3", "#4a8a4a")   # trained FPN / BEV features
DEPTH = ("#fbe6c2", "#cf9532")
HEAD = ("#e6dff2", "#7a5aa8")
LOGIT = ("#d7ccf0", "#6a4aa0")
GT = ("#cfe9e5", "#2f8f86")
LOSS = ("#f3d4d4", "#b05a5a")
RAY = "#e2a53a"
GRID = "#b9b9b9"
INK = "#2a2a2a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def shade(color, f):
    r, g, b = mc.to_rgb(color)
    if f >= 0:
        return (r * (1 - f) + f, g * (1 - f) + f, b * (1 - f) + f)  # lighten
    f = -f
    return (r * (1 - f), g * (1 - f), b * (1 - f))                  # darken


def cuboid(ax, cx, cy, fw, fh, depth, face, edge, skew=0.42, z=3):
    """Pseudo-3D box centred at (cx, cy); front face fw x fh, thickness `depth`."""
    x, y = cx - fw / 2, cy - fh / 2
    dx, dy = depth * skew, depth * skew
    front = [(x, y), (x + fw, y), (x + fw, y + fh), (x, y + fh)]
    top = [(x, y + fh), (x + fw, y + fh), (x + fw + dx, y + fh + dy), (x + dx, y + fh + dy)]
    right = [(x + fw, y), (x + fw, y + fh), (x + fw + dx, y + fh + dy), (x + fw + dx, y + dy)]
    ax.add_patch(Polygon(top, closed=True, facecolor=shade(face, -0.12), edgecolor=edge, lw=1.1, zorder=z))
    ax.add_patch(Polygon(right, closed=True, facecolor=shade(face, -0.22), edgecolor=edge, lw=1.1, zorder=z))
    ax.add_patch(Polygon(front, closed=True, facecolor=face, edgecolor=edge, lw=1.3, zorder=z + 0.1))
    return {"cx": cx, "cy": cy, "left": x, "right": x + fw + dx, "front_right": x + fw,
            "bottom": y, "top": y + fh, "top3d": y + fh + dy}


def feat_sheets(ax, cx, cy, w, h, k, face, edge, off=0.09, z=3):
    """A feature map drawn as k offset stacked sheets (channels as depth)."""
    for s in range(k):
        x = cx - w / 2 + (s - (k - 1) / 2) * off
        y = cy - h / 2 + (s - (k - 1) / 2) * off
        fc = face if s == k - 1 else shade(face, 0.18)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=edge,
                     lw=1.1, zorder=z + s * 0.01))
    fx = cx + ((k - 1) / 2) * off
    fy = cy + ((k - 1) / 2) * off
    return {"cx": cx, "cy": cy, "left": cx - w / 2 - ((k - 1) / 2) * off,
            "right": fx + w / 2, "bottom": cy - h / 2 - ((k - 1) / 2) * off,
            "top": fy + h / 2, "front_cx": fx, "front_cy": fy}


def rbox(ax, cx, cy, w, h, text, face, edge, fs=10, z=3, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 facecolor=face, edgecolor=edge, lw=1.4, zorder=z))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=z + 1,
                color=INK, weight=weight)
    return {"cx": cx, "cy": cy, "left": cx - w / 2, "right": cx + w / 2,
            "bottom": cy - h / 2, "top": cy + h / 2}


def arrow(ax, p0, p1, color=INK, ls="-", rad=0.0, lw=1.5, ms=13, z=1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                 color=color, linestyle=ls, zorder=z,
                 connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, text, fs=9, color=INK, ha="center", va="center", style="normal",
          weight="normal", z=5):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, style=style,
            weight=weight, zorder=z)


def oplus(ax, cx, cy, r=0.13, color=INK, z=4):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=color, lw=1.4, zorder=z))
    ax.plot([cx - r * 0.6, cx + r * 0.6], [cy, cy], color=color, lw=1.3, zorder=z + 1)
    ax.plot([cx, cx], [cy - r * 0.6, cy + r * 0.6], color=color, lw=1.3, zorder=z + 1)


# --- oblique 3D (axonometric) projection: X=lateral, Y=forward/range, Z=up ----
EX = (0.98, 0.00)
EY = (0.52, 0.33)
EZ = (0.00, 0.95)


def proj(x, y, z, o=(0.0, 0.0)):
    return (o[0] + x * EX[0] + y * EY[0] + z * EZ[0],
            o[1] + x * EX[1] + y * EY[1] + z * EZ[1])


def poly3d(ax, pts, facecolor, edgecolor, lw=1.0, alpha=1.0, z=2, o=(0, 0)):
    scr = [proj(*p, o) for p in pts]
    ax.add_patch(Polygon(scr, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                 lw=lw, alpha=alpha, zorder=z))


def line3d(ax, p0, p1, color=INK, lw=1.2, ls="-", alpha=1.0, z=2, o=(0, 0)):
    a, b = proj(*p0, o), proj(*p1, o)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)


def ground_grid(ax, R, Yf, n, o=(0, 0), colors=None, z=2, edge="#c4c4c4", lw=0.5):
    """Ground plane Z=0 over X in [-R,R], Y in [0,Yf], n x n cells.

    colors: optional (n, n) array of RGB (0-1) or None for empty cells.
    Cell (i, j): i indexes X (lateral), j indexes Y (forward).
    """
    dx, dy = 2 * R / n, Yf / n
    for i in range(n):
        for j in range(n):
            x0, y0 = -R + i * dx, j * dy
            corners = [(x0, y0, 0), (x0 + dx, y0, 0), (x0 + dx, y0 + dy, 0), (x0, y0 + dy, 0)]
            fc = "none"
            if colors is not None:
                c = colors[j, i]
                fc = tuple(c) if c is not None else "none"
            poly3d(ax, corners, facecolor=fc, edgecolor=edge, lw=lw, z=z, o=o)


def bev_colors(png_path, n):
    """Downsample a BEV raster PNG to (n, n) RGB(0-1), background -> None (empty)."""
    from PIL import Image
    im = Image.open(png_path).convert("RGB").resize((n, n), Image.NEAREST)
    arr = np.asarray(im).astype(float) / 255.0
    out = np.empty((n, n), dtype=object)
    bg = np.array([20, 20, 26]) / 255.0
    for j in range(n):
        for i in range(n):
            c = arr[j, i]
            out[j, i] = None if np.allclose(c, bg, atol=0.12) else tuple(c)
    return out


def inset_image(fig, rect, path, title=None, edge=IMG[1]):
    """Place a real image (imshow) at figure-fraction rect [x, y, w, h]."""
    import matplotlib.image as mpimg
    iax = fig.add_axes(rect)
    iax.imshow(mpimg.imread(path))
    iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_edgecolor(edge); s.set_linewidth(1.6)
    if title:
        iax.set_title(title, fontsize=8.5)
    return iax


def img_tile(ax, path, cx, cy, w, h, edge, title=None, z=3, fs=8.2):
    """Draw a real image at data coords [cx +/- w/2, cy +/- h/2] with a border."""
    import matplotlib.image as mpimg
    ax.imshow(mpimg.imread(path), extent=[cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2],
              aspect="auto", zorder=z)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="none",
                 edgecolor=edge, lw=1.4, zorder=z + 0.1))
    if title:
        ax.text(cx, cy + h / 2 + 0.12, title, ha="center", va="bottom", fontsize=fs, zorder=z + 1)


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, ax, stem):
    from pathlib import Path
    out = Path(__file__).resolve().parent / stem
    fig.tight_layout(pad=0.3)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.svg")
