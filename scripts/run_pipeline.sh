#!/usr/bin/env bash
# Overnight pipeline: wait for the trainval download, rasterize BEV GT for the
# scenes we actually have, free the GPU, then finetune from the warmup weights.
# Each stage logs with timestamps; a stage failure aborts the rest.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
DL_LOG="logs/download.log"

say () { echo "[$(date +%H:%M:%S)] [pipeline] $*"; }

# --- 1. wait for the download to finish -------------------------------------
say "waiting for download to complete ($DL_LOG)"
while ! grep -q "ALL DOWNLOADS COMPLETE" "$DL_LOG" 2>/dev/null; do
  sleep 30
done
say "download complete. disk free: $(df -h . | awk 'NR==2{print $4}')"

if [ ! -d data/nuscenes/samples/CAM_FRONT ] || [ ! -d data/nuscenes/v1.0-trainval ]; then
  say "ERROR: expected data/nuscenes/{samples,v1.0-trainval} missing after download"; exit 1
fi

# --- 2. rasterize BEV ground truth (available scenes only) ------------------
say "rasterizing train split -> data/bev_rasters/trainval_train"
$PY scripts/rasterize_bev.py --version v1.0-trainval --split train \
    --out data/bev_rasters/trainval_train --require-files \
  || { say "ERROR: train rasterization failed"; exit 1; }

say "rasterizing val split -> data/bev_rasters/trainval_val"
$PY scripts/rasterize_bev.py --version v1.0-trainval --split val \
    --out data/bev_rasters/trainval_val --require-files \
  || { say "ERROR: val rasterization failed"; exit 1; }

n_train=$(ls data/bev_rasters/trainval_train/*.npz 2>/dev/null | wc -l)
n_val=$(ls data/bev_rasters/trainval_val/*.npz 2>/dev/null | wc -l)
say "rasters written: train=$n_train  val=$n_val"
if [ "$n_train" -lt 100 ] || [ "$n_val" -lt 20 ]; then
  say "ERROR: too few rasters (train=$n_train val=$n_val) — download/scene-filter issue"; exit 1
fi

# --- 3. free the GPU (idle Jupyter kernel is holding it) ---------------------
if pgrep -f ipykernel_launcher >/dev/null; then
  say "killing idle Jupyter kernel to free the GPU (notebook file is untouched; restart it when you're back)"
  pkill -f ipykernel_launcher || true
  sleep 5
fi
say "GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null)"

# --- 4. train ---------------------------------------------------------------
say "starting training (finetune from warmup, batch 1)"
PYTORCH_ALLOC_CONF=expandable_segments:True $PY scripts/train.py \
    --version v1.0-trainval --train-split train --val-split val \
    --train-raster data/bev_rasters/trainval_train \
    --val-raster   data/bev_rasters/trainval_val \
    --warmup models/frozen-warmup/camera_bev_seg_best.pt \
    --out models/trainval --viz viz_trainval \
    --batch 1 --workers 4 --epochs 15 --patience 5 \
  || { say "ERROR: training failed"; exit 1; }

say "PIPELINE COMPLETE. best checkpoint: models/trainval/best.pt  viz: viz_trainval/"
