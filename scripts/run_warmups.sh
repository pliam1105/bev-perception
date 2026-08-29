#!/usr/bin/env bash
# Sequential frozen-backbone warmups: non-depth (batch 2) then forward-scatter
# (batch 1). Each follows the notebook hyperparameters and writes into its own
# models/ and viz/ folder. Run after the GPU is free (the depth warmup done and
# its kernel released). The depth loss curve is pulled from the notebook output
# separately; this only runs the two remaining models.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
say () { echo "[$(date +%H:%M:%S)] [warmups] $*"; }

# mini dataset, to match the depth warmup (v1.0-mini, mini_train/mini_val)
DATA="--version v1.0-mini --train-split mini_train --val-split mini_val \
      --train-raster data/bev_rasters/mini_train --val-raster data/bev_rasters/mini_val"

say "non-depth (projection) warmup, batch 2"
PYTORCH_ALLOC_CONF=expandable_segments:True $PY scripts/warmup.py --model nodepth --train-batch 2 $DATA \
  || { say "ERROR: nodepth warmup failed"; exit 1; }

say "forward-scatter warmup, batch 1"
PYTORCH_ALLOC_CONF=expandable_segments:True $PY scripts/warmup.py --model scatter --train-batch 1 $DATA \
  || { say "ERROR: scatter warmup failed"; exit 1; }

say "WARMUPS COMPLETE. models/warmup-{nodepth,scatter}/  viz/warmup-{nodepth,scatter}/"
