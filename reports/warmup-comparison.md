# Frozen-backbone warmup comparison — three BEV lifts

**Date:** 2026-08-29
**Dataset:** nuScenes **mini** (`mini_train` 323 / `mini_val` 81 keyframes), all 6 cameras, camera-only.
**Setup:** frozen ImageNet ResNet-18 backbone (FPN + seg head trainable only); no unfreezing.

## Models compared

| model | lift | class |
|---|---|---|
| **depth** | pull-based, LSS-style depth-weighted sampling | `CameraBEVSeg` |
| **nodepth** | projection-only (no depth), mean over cameras, sum over height | `CameraBEVSegProjection` |
| **scatter** | forward-scatter (LSS splat), depth-weighted `index_put_` | `CameraBEVSegScatter` |

All three lift into a `(y, x)` BEV grid (200×200 @ 0.5 m, ±50 m) consistent with the GT raster.

## Hyperparameters (identical across runs, matching the notebook warmup)

- Optimizer: `Adam(lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-12)`
- Loss: `SegLoss` = FocalLoss(γ=2, α=(0.25, 0.5)) + DiceLoss(eps=1e-5), λ=1
- Grad clip: 1.0
- Early stopping: patience 5 on val loss; cap 15 epochs, auto-extends to 50 if still improving at the cap
- Batch: nodepth 2, depth 1, scatter 1
- Loss reported as **mean per batch** (depth's notebook sums were converted: train ÷323, val ÷81)

## Results

| model | trainable params | best val (mean/batch) | best epoch | final train | epochs run |
|---|---|---|---|---|---|
| **depth** | 1,292,364 | **0.7574** | 5 | 0.463 | 11 (early-stop) |
| **scatter** | 1,292,364 | **0.7772** | 11 | 0.487 | 17 (early-stop) |
| **nodepth** | 1,281,378 | **0.7882** | 9 | 0.569 | 15 (early-stop) |

Lower val loss is better. Ranking: **depth < scatter < nodepth**.

### Per-epoch val loss

| epoch | depth | nodepth | scatter |
|---|---|---|---|
| 0 | 0.832 | 0.818 | 0.832 |
| 1 | 0.801 | 0.823 | 0.832 |
| 2 | 0.803 | 0.809 | 0.799 |
| 3 | 0.782 | 0.858 | 0.803 |
| 4 | 0.790 | 0.829 | 0.860 |
| 5 | **0.757** | 0.801 | 0.838 |
| 6 | 0.827 | 0.802 | 0.804 |
| 7 | 0.804 | 0.810 | 0.778 |
| 8 | 0.829 | 0.797 | 0.781 |
| 9 | 0.835 | **0.788** | 0.781 |
| 10 | 0.818 | 0.849 | 0.806 |
| 11 | — | 0.874 | **0.777** |
| 12 | — | 0.804 | 0.780 |
| 13 | — | 0.840 | 0.834 |
| 14 | — | 0.810 | 0.804 |
| 15 | — | — | 0.851 |
| 16 | — | — | 0.797 |

## Findings

- **Both depth-explicit lifts beat the projection baseline** (depth 0.757, scatter 0.777 vs nodepth 0.788): modeling depth earns a real, if modest, val improvement over pure projection.
- **Pull-based depth edges out forward-scatter** (0.757 vs 0.777) and reaches its best in fewer epochs. The two are close, and this is mini.
- **All three overfit** (train ≪ val by the end — e.g. depth 0.46 train vs 0.76 val), expected with 323 scenes and a frozen backbone. These are relative warmup signals, not absolute quality; the real separation should appear on trainval with the backbone unfrozen.

## Notes / fixes made during this run set

- **BEV transpose bug (pull-based lift):** its output was `(x, y)` while the GT raster is `(y, x)`. Verified geometry was correct (each camera's features land in the right world direction) but the axis convention disagreed with GT — a conv head cannot undo a transpose, which capped drivable and killed vehicles. Fixed by building `bev_norm_coords` y-first. `BEVLiftProjection` (recovered from commit `5bbe6de`) got the same fix.
- **Forward-scatter backward OOM:** the D=42 loop re-gathered `backbone_out` via an identity index every iteration, retaining 42 copies. Replacing it with `backbone_out * depth_probs_out[:,:,d].unsqueeze(2)` cut backward peak from 5.64 GB (OOM) to 1.60 GB.

## Artifacts

- Best weights: `models/warmup-{depth,nodepth,scatter}/best.pt`
- Per-epoch losses: `models/warmup-{depth,nodepth,scatter}/losses.json`
- Per-epoch prediction montages: `viz/warmup-{depth,nodepth,scatter}/epoch_*.png`
- Loss curves (all three): `viz/warmup-loss-curves.png`
- Best-epoch prediction comparison: `viz/warmup-best-comparison.png`
