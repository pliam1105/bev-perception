#!/usr/bin/env python3
import importlib.util, time, traceback
from pathlib import Path
import numpy as np
spec_ = importlib.util.spec_from_file_location("rb", "scripts/rasterize_bev.py")
rb = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(rb)
from bev.raster import BEVGridSpec
from bev.data import NuScenesBEVDataset, NuScenesConfig
from bev.transforms import Transform
import pyquaternion
ds = NuScenesBEVDataset(NuScenesConfig(dataroot="data/nuscenes", cameras=(), load_lidar=True))
r = rb.BBoxRasterizer(BEVGridSpec(), sorted(Path("data/nuscenes/maps").glob("*.png")), ds.nusc)
s = ds[0]
try:
    m = r.maps.get(r._location_of(s.sample_token))
    e2g = Transform(s.lidar.calib.ego2global_translation.numpy(), pyquaternion.Quaternion(s.lidar.calib.ego2global_rotation.numpy()).rotation_matrix, s.timestamp)
    rr = np.zeros((2, r.spec.ny, r.spec.nx))
    for _ in range(3): r._rasterize_map(rr, m, e2g)  # warmup
    rr = np.zeros((2, r.spec.ny, r.spec.nx)); t0=time.time(); r._rasterize_map(rr, m, e2g); dt=(time.time()-t0)*1000
    px=int(rr[0].sum())
    print(f"map pass: {dt:.2f} ms  (loop ~3200 ms) | drivable px: {px} vs loop 7962 -> {'MATCH' if px==7962 else 'DIFFERS'}")
    t0=time.time(); full=r.rasterize(s); print(f"full sample: {(time.time()-t0)*1000:.1f} ms  | drivable {int(full[0].sum())} vehicle {int(full[1].sum())}")
except Exception:
    traceback.print_exc()