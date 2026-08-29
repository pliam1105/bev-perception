#!/usr/bin/env python3
"""Stack each model's best-epoch prediction montage into one comparison PNG.

    python scripts/compare_best_viz.py \
        depth:viz/warmup-depth/epoch_5.png:0.757 \
        nodepth:viz/warmup-nodepth/epoch_9.png:0.788 \
        scatter:viz/warmup-scatter/epoch_11.png:0.777
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = Path("viz/warmup-best-comparison.png")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    rows = []
    for spec in argv:
        label, path, val = spec.split(":")
        rows.append((label, Image.open(path), float(val)))

    fig, axes = plt.subplots(len(rows), 1, figsize=(16, 5.2 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, (label, img, val) in zip(axes, rows):
        ax.imshow(img)
        ax.set_title(f"{label}   (best epoch, val {val:.3f})   —   train pred | train GT | val pred | val GT",
                     fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Frozen-backbone warmup — best-epoch predictions (mini)", fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
