#!/usr/bin/env bash
# Download + extract nuScenes trainval metadata and the first two sensor blobs
# from the public CloudFront mirror (no auth). Archives are deleted right after
# extraction to bound peak disk use. Resumable and retrying so it survives an
# overnight run.
#
#   scripts/download_trainval.sh
set -uo pipefail

cd "$(dirname "$0")/.."
DATAROOT="data/nuscenes"
BASE="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"
mkdir -p "$DATAROOT"

fetch_extract () {
  local f="$1"
  local dst="$DATAROOT/$f"
  echo "[$(date +%H:%M:%S)] downloading $f"
  # --retry handles transient errors; the until loop handles a hard drop.
  until curl -fL -C - --retry 8 --retry-delay 15 --retry-connrefused -o "$dst" "$BASE/$f"; do
    echo "[$(date +%H:%M:%S)] download of $f interrupted, retrying in 20s"
    sleep 20
  done
  echo "[$(date +%H:%M:%S)] extracting $f"
  tar -xzf "$dst" -C "$DATAROOT"
  echo "[$(date +%H:%M:%S)] removing archive $f"
  rm -f "$dst"
  echo "[$(date +%H:%M:%S)] done $f  (disk now: $(df -h "$DATAROOT" | awk 'NR==2{print $4}') free)"
}

fetch_extract v1.0-trainval_meta.tgz
fetch_extract v1.0-trainval01_blobs.tgz
fetch_extract v1.0-trainval02_blobs.tgz
echo "[$(date +%H:%M:%S)] ALL DOWNLOADS COMPLETE"
