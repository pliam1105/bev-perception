#!/usr/bin/env python3
"""Stream one nuScenes scene to Foxglove over a live WebSocket bridge.

    python scripts/visualize_scene.py --dataroot data/nuscenes --scene scene-0061

Then open the Foxglove app and connect to ws://localhost:8765. Add an Image
panel per camera topic and a 3D panel (it will pick up the transforms, the
lidar point cloud, and the annotation boxes).

Walks the scene's keyframes in order and publishes, per sample: the transform
tree (global -> ego/<sensor> -> <sensor>), each camera JPEG, the lidar cloud,
and the 3D boxes. Playback is paced by the keyframe timestamps and replays on a
loop until interrupted; pass --once for a single pass.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bev.viz.foxglove_bridge import FoxgloveBridge  # noqa: E402
from bev.viz.scene_reader import NuScenesSceneReader, SceneSample, SensorEvent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", default=None, help="scene name or token (default: first scene)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rate", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--once", action="store_true", help="play the scene once instead of replaying on loop")
    parser.add_argument(
        "--full",
        action="store_true",
        help="stream all sample_data (sweeps included) at full sensor rate, not just keyframes",
    )
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument(
        "--bev-raster-root",
        type=Path,
        default=None,
        help="precomputed BEV raster store; publishes each keyframe's GT to /bev/<layer>",
    )
    return parser.parse_args()


class MonotonicClock:
    """Remap raw sensor timestamps to a stream that never steps backward.

    Each loop is shifted to continue after the previous one, so the published
    clock stays monotonic across replays while the relative timing between
    sensors within a pass is preserved. A one-second head start keeps the first
    sample's sub-keyframe sensor offsets from mapping below zero.
    """

    def __init__(self, gap_us: int = 500_000) -> None:
        self._origin: int | None = None
        self._loop_base = 0
        self._last = 0
        self._gap = gap_us

    def map(self, raw_us: int) -> int:
        if self._origin is None:
            self._origin = raw_us - 1_000_000
        mapped = raw_us - self._origin + self._loop_base
        self._last = max(self._last, mapped)
        return mapped

    def next_loop(self) -> None:
        self._loop_base = self._last + self._gap


# High-contrast per-layer colors, reused by the 3D clouds and the top-down image.
BEV_COLORS: dict[str, tuple[int, int, int]] = {
    "drivable_area": (110, 200, 255),  # cyan
    "vehicle": (255, 120, 20),         # orange
}
_BEV_FALLBACK = (200, 200, 200)
_BEV_BG = (18, 18, 22)


def publish_bev_raster(bridge: FoxgloveBridge, bev_raster, frame: str, timestamp: int) -> None:
    """Publish each occupied BEV cell as a colored point at ground level, one
    cloud per layer. The color is packed per point so it shows regardless of the
    viewer's color setting.
    """
    spec = bev_raster.spec
    data = np.asarray(bev_raster.data)
    for c, name in enumerate(bev_raster.layer_names):
        occ = np.argwhere(data[c] > 0)  # (K, 2): rows are (j=y-index, i=x-index)
        if occ.size == 0:
            continue
        xs = (occ[:, 1] + 0.5) * spec.resolution + spec.x_min
        ys = (occ[:, 0] + 0.5) * spec.resolution + spec.y_min
        pts = np.stack([xs, ys, np.zeros(len(occ))], axis=1)
        bridge.publish_pointcloud(f"/bev/{name}", frame, pts, timestamp, rgb=BEV_COLORS.get(name, _BEV_FALLBACK))


def publish_bev_image(bridge: FoxgloveBridge, bev_raster, timestamp: int) -> None:
    """Publish a top-down RGB image of the raster (forward = up) to /bev/image."""
    import io
    from PIL import Image

    data = np.asarray(bev_raster.data)
    h, w = data.shape[1], data.shape[2]
    img = np.full((h, w, 3), _BEV_BG, dtype=np.uint8)
    for c, name in enumerate(bev_raster.layer_names):
        img[data[c] > 0] = BEV_COLORS.get(name, _BEV_FALLBACK)
    img = np.flipud(img)  # row 0 is +y (ego forward) at the top
    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(buf, format="PNG")
    bridge.publish_compressed_image("/bev/image", "bev", buf.getvalue(), timestamp, image_format="png")


def publish_sample(bridge: FoxgloveBridge, sample: SceneSample, clock: MonotonicClock) -> None:
    for channel, cam in sample.cameras.items():
        ego_frame = f"ego/{channel}"
        c = cam.calib
        t = clock.map(c.timestamp)
        bridge.publish_transform(
            "/tf", "global", ego_frame, c.ego2global_translation, c.ego2global_rotation, t
        )
        bridge.publish_transform(
            "/tf", ego_frame, channel, c.sensor2ego_translation, c.sensor2ego_rotation, t
        )
        bridge.publish_compressed_image(
            f"/camera/{channel}", channel, cam.image_path.read_bytes(), t
        )

    if sample.lidar is not None:
        lc = sample.lidar.calib
        t = clock.map(lc.timestamp)
        bridge.publish_transform(
            "/tf", "global", "ego/LIDAR_TOP", lc.ego2global_translation, lc.ego2global_rotation, t
        )
        bridge.publish_transform(
            "/tf", "ego/LIDAR_TOP", "LIDAR_TOP", lc.sensor2ego_translation, lc.sensor2ego_rotation, t
        )
        bridge.publish_pointcloud("/lidar", "LIDAR_TOP", sample.lidar.points, t)
        if sample.bev_raster is not None:
            publish_bev_raster(bridge, sample.bev_raster, "ego/LIDAR_TOP", t)

    if sample.bev_raster is not None:
        publish_bev_image(bridge, sample.bev_raster, clock.map(sample.timestamp))

    if sample.boxes:
        bridge.publish_boxes("/boxes", "global", sample.boxes, clock.map(sample.timestamp))


def publish_event(bridge: FoxgloveBridge, event: SensorEvent, clock: MonotonicClock) -> None:
    t = clock.map(event.timestamp)

    if event.camera is not None:
        ch = event.channel
        ego_frame = f"ego/{ch}"
        c = event.camera.calib
        bridge.publish_transform(
            "/tf", "global", ego_frame, c.ego2global_translation, c.ego2global_rotation, t
        )
        bridge.publish_transform(
            "/tf", ego_frame, ch, c.sensor2ego_translation, c.sensor2ego_rotation, t
        )
        bridge.publish_compressed_image(
            f"/camera/{ch}", ch, event.camera.image_path.read_bytes(), t
        )

    if event.lidar is not None:
        lc = event.lidar.calib
        bridge.publish_transform(
            "/tf", "global", "ego/LIDAR_TOP", lc.ego2global_translation, lc.ego2global_rotation, t
        )
        bridge.publish_transform(
            "/tf", "ego/LIDAR_TOP", "LIDAR_TOP", lc.sensor2ego_translation, lc.sensor2ego_rotation, t
        )
        bridge.publish_pointcloud("/lidar", "LIDAR_TOP", event.lidar.points, t)

    if event.boxes:
        bridge.publish_boxes("/boxes", "global", event.boxes, t)


def play_once_full(
    bridge: FoxgloveBridge, reader: NuScenesSceneReader, scene: str, rate: float, clock: MonotonicClock
) -> None:
    prev_ts: int | None = None
    count = 0
    for event in reader.read_sample_data(scene):
        if prev_ts is not None and rate > 0:
            dt = (event.timestamp - prev_ts) / 1e6 / rate
            if dt > 0:
                time.sleep(dt)
        prev_ts = event.timestamp
        publish_event(bridge, event, clock)
        count += 1
        if count % 100 == 0:
            print(f"  {count} events  (last: {event.channel} @ t={event.timestamp})")


def play_once(
    bridge: FoxgloveBridge, reader: NuScenesSceneReader, scene: str, rate: float, clock: MonotonicClock
) -> None:
    prev_ts: int | None = None
    for sample in reader.read_scene(scene):
        if prev_ts is not None and rate > 0:
            dt = (sample.timestamp - prev_ts) / 1e6 / rate
            if dt > 0:
                time.sleep(dt)
        prev_ts = sample.timestamp
        publish_sample(bridge, sample, clock)
        print(
            f"  t={sample.timestamp}  cams={len(sample.cameras)}  "
            f"lidar={'-' if sample.lidar is None else sample.lidar.points.shape[0]}  "
            f"boxes={len(sample.boxes)}"
        )


def main() -> int:
    args = parse_args()

    reader = NuScenesSceneReader(
        args.dataroot,
        version=args.version,
        include_lidar=not args.no_lidar,
        bev_raster_root=args.bev_raster_root,
    )
    scenes = reader.list_scenes()
    if not scenes:
        print("no scenes found", file=sys.stderr)
        return 1
    scene = args.scene or scenes[0][1]
    print(f"scene {scene!r}  ({len(scenes)} scenes available in {args.version})")

    with FoxgloveBridge(port=args.port) as bridge:
        print(f"Foxglove bridge live on ws://localhost:{args.port}  — connect the app, Ctrl-C to stop")
        clock = MonotonicClock()
        play = play_once_full if args.full else play_once
        try:
            while True:
                play(bridge, reader, scene, args.rate, clock)
                if args.once:
                    break
                clock.next_loop()
                print("  (loop) restarting scene")
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
