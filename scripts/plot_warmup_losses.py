#!/usr/bin/env python3
"""Combined train/val loss curves for the warmup runs.

Reads each model's losses.json (written by warmup.py) or an explicit label:path
pair, and plots train (solid) and val (dashed) loss per epoch on shared axes.

    python scripts/plot_warmup_losses.py \
        depth=models/warmup-depth/losses.json \
        nodepth=models/warmup-nodepth/losses.json \
        scatter=models/warmup-scatter/losses.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("viz/warmup-loss-curves.png")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    runs = {}
    for spec in argv:
        label, _, path = spec.partition("=")
        runs[label] = json.loads(Path(path).read_text())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10.colors
    for i, (label, hist) in enumerate(runs.items()):
        c = colors[i % len(colors)]
        epochs = range(len(hist["train"]))
        ax.plot(epochs, hist["train"], "-", color=c, label=f"{label} train")
        ax.plot(range(len(hist["val"])), hist["val"], "--", color=c, label=f"{label} val")

    ax.set_xlabel("epoch")
    ax.set_ylabel("SegLoss (mean per batch)")
    ax.set_title("frozen-backbone warmup — train (solid) vs val (dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=len(runs), fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
