# bev-perception

Exploring the use of learned methods for the fusion of multiple cameras, LiDAR,
and other sensors in the Bird-Eye View frame.

## Setup

```bash
python3 -m venv --system-site-packages .venv   # reuses the system torch build
.venv/bin/pip install -r requirements.txt
```

## Data

The mini split (~4.2 GB compressed, ~9 GB peak during extract) needs no login:

```bash
./scripts/download_nuscenes.sh data/nuscenes
```

Full `trainval` requires an account — download from
<https://www.nuscenes.org/nuscenes#download> and extract into the same dataroot.
`data/` is gitignored.

## Verify the loader

```bash
PYTHONPATH=src .venv/bin/python scripts/inspect_sample.py \
    --dataroot data/nuscenes --lidar --out out/sample.png
```

Prints image shapes, intrinsics and raw calibration per camera, and writes the
six cameras in their physical layout. Run this before trusting anything
downstream.

```bash
.venv/bin/python -m pytest tests/ -q
```

Tests run against a synthetic nuScenes tree (`tests/make_fake_nuscenes.py`), so
they work without the real download.

## Layout

```
src/bev/data/nuscenes_dataset.py   keyframe reader: images + lidar + raw calibration
scripts/download_nuscenes.sh       fetch + extract v1.0-mini
scripts/inspect_sample.py          smoke test / surround-view render
tests/                             synthetic fixture + loader tests
```
