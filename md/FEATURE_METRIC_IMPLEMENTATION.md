# Feature-Metric Odometry — Implementation Summary

## What Was Built

Five files implement the feature-metric GN pipeline described in `FEATURE_METRIC_PLAN.md`, replacing DEIO's `CorrBlock + GRU Update + BA` with a direct feature-alignment solver and replacing TartanAir data loading with data from `~/repos/deep_event_odometry`.

---

## Files Changed / Created

### New Files

| File | Role |
|---|---|
| `devo/data_readers/deep_event_odom.py` | Dataset adapter |
| `devo/feature_metric_gn.py` | Gauss-Newton solver |
| `devo/fmnet.py` | FMNet network class |
| `train_fm.py` | Training script |

### Modified Files

| File | Change |
|---|---|
| `devo/data_readers/factory.py` | Added `'deep_event_odom'` entry |

---

## File-by-File Details

### 1. `devo/data_readers/deep_event_odom.py`

Wraps `MultiKeyframeDataset` from `~/repos/deep_event_odometry` and converts its outputs to tensors compatible with the GN training pipeline.

**Conversions applied:**

| Source format | Target format | How |
|---|---|---|
| Log-norm depth `[0,1]` | Inverse depth `1/d` | `expm1(d_log * log1p(d_max * k)) / k` |
| Normalised intrinsics `[fx/W, fy/H, cx/W, cy/H]` | Pixel-unit intrinsics `[fx, fy, cx, cy]` | Multiply by `[W, H, W, H]` |
| 4×4 SE3 matrix `(N, 4, 4)` | 7-vector `[tx, ty, tz, qx, qy, qz, qw]` | `scipy.Rotation.from_matrix().as_quat()` |
| Full-res disparity `(N, H, W)` | Feature-map-res disparity `(N, H/4, W/4)` | Subsample `[:, 1::4, 1::4]` |

**Output dict per sample:**
```
images:          (N, K_sub, H, W)   — event voxel grids
disps:           (N, H/4, W/4)      — inverse depth at feature resolution
poses:           (N, 7)             — SE3 7-vectors
intrinsics:      (4,)               — [fx, fy, cx, cy] at feature resolution
intrinsics_full: (4,)               — [fx, fy, cx, cy] at full resolution
```

---

### 2. `devo/feature_metric_gn.py`

Implements the direct feature-metric Gauss-Newton solver.  One iteration does:

```
1. project patches into target frame         [reuses projective_ops.transform()]
2. sample target features f_target           [bilinear grid_sample]
3. sample source features f_source           [bilinear grid_sample]
4. residual:  e_k = f_target - f_source      (B, E, C)
5. feature gradient ∂fmap/∂pixel             [central differences]
6. chain-rule Jacobian:
      J_j = (∂fmap/∂pixel) @ (∂pixel/∂ξ_j)  (B, E, C, 6)
      J_i = (∂fmap/∂pixel) @ (∂pixel/∂ξ_i)  (B, E, C, 6)
7. weighted normal equations:
      H += w_k * J^T J    (6×6 block)
      b += w_k * J^T e    (6,)
8. Schur complement solve for δξ and δz      [reuses ba.CholeskySolver]
9. retract: pose_retr(), disp_retr()          [reuses ba.py]
```

**Reused from existing code:**

| Function | Source |
|---|---|
| `CholeskySolver` | `devo/ba.py:12` |
| `safe_scatter_add_mat/vec` | `devo/ba.py:40–46` |
| `block_matmul`, `block_solve` | `devo/ba.py:58–76` |
| `pose_retr`, `disp_retr` | `devo/ba.py:54–56` |
| `transform(..., jacobian=True)` | `devo/projective_ops.py:53` |

**Coarse-to-fine:** called at pyramid levels `[2, 1]` (½-res then full-res) for each GN round.

---

### 3. `devo/fmnet.py`

The complete network class, keeping DEIO encoders and patch selection but replacing the GRU-based update loop.

**Kept from DEIO:**
- `BasicEncoder4Evs` — fnet (instance norm) and inet (no norm)
- `Scorer` + `PatchSelector` — learned patch importance
- `coords_grid_with_index`, `set_depth`, `flatmeshgrid` utilities
- Incremental frame addition logic (mirrors `eVONet`)

**Removed:**
- `CorrBlock` (7×7 correlation search)
- `Update` (GRU flow predictor)
- `BA` call (flow targets → pose)

**Added:**
- `weight_head`: `Linear(384→128) → ReLU → Linear(128→1) → Sigmoid` — predicts per-patch confidence from context features
- `feature_metric_gn()` call at two pyramid levels per step

**Forward signature:**
```python
net(images, poses, disps, intrinsics,
    STEPS=12, patches_per_image=80,
    structure_only=False, gn_iters=4, lm=1e-4, ep=10.0)
# returns: traj — list of (poses_pred, poses_gt, resids) per step
```

---

### 4. `train_fm.py`

Training script modelled after `train.py`.

**Losses:**

| Loss | Formula | Weight |
|---|---|---|
| `L_pose` | Geodesic error on all frame pairs (with Kabsch scale alignment) | `--pose_weight` (default 10.0) |
| `L_feat` | `mean(resids²)` — feature-consistency (encourages non-zero spatial gradients) | `--feat_weight` (default 0.1) |

**Key flags:**
```
--data_path       HDF5 dataset root (required)
--name            experiment name
--n_keyframes     keyframe window size (default: 4)
--interval_frames frames between keyframes (default: 5)
--bins            event voxel bins (default: 5)
--iters           GN refinement rounds per forward pass (default: 12)
--gn_iters        GN iterations per round, split across 2 pyramid levels (default: 4)
--pose_weight     weight for geodesic pose loss (default: 10.0)
--feat_weight     weight for feature-consistency loss (default: 0.1)
```

**Example launch:**
```bash
python train_fm.py \
    --data_path /path/to/hdf5_data \
    --name fmnet_run1 \
    --n_keyframes 4 \
    --interval_frames 5 \
    --bins 5 \
    --steps 100000 \
    --lr 8e-5
```

---

### 5. `devo/data_readers/factory.py` (modified)

Added one import and one entry:
```python
from .deep_event_odom import DeepEventOdomDataset

dataset_map = {
    ...
    'deep_event_odom': (DeepEventOdomDataset,),
}
```

---

## Architecture Comparison

| Component | DEIO (original) | FMNet (new) |
|---|---|---|
| Feature extraction | `BasicEncoder4Evs` (fnet, inet) | Same |
| Patch selection | `Scorer` + `PatchSelector` | Same |
| Matching | `CorrBlock` (7×7 search window) | Direct projection + feature residual |
| Update | `GRU Update` (predicts δflow) | `weight_head` MLP (predicts confidence only) |
| Pose/depth solve | `BA` (flow targets → Gauss-Newton) | `feature_metric_gn` (feature metric → Gauss-Newton) |
| Data | TartanAir (synthetic, RGB) | `deep_event_odometry` (real, HDF5 events) |

---

## Data Flow

```
HDF5 sequence
    → EventOdometryDataset → MultiKeyframeDataset
    → DeepEventOdomDataset (convert depth / intrinsics / poses)
    → DataLoader batch (images, disps, poses, intrinsics)
    ↓
FMNet.forward()
    fnet → fmaps (B, N, 128, H/4, W/4)
    inet → imaps (B, N, 384, H/4, W/4)
    Scorer → patch coords (N × M patches)
    weight_head(imap at patch centres) → w_k  (B, M)
    for step in range(STEPS):
        for level in [2, 1]:  ← coarse-to-fine
            feature_metric_gn_step(fmaps_l, patches, poses, ..., w_k)
                project → residuals → Jacobians → Schur solve → retract
    → traj [(poses_pred, poses_gt, resids) × STEPS]
    ↓
train_fm.py loss
    L_pose = geodesic(poses_pred, poses_gt)   × 10.0
    L_feat = mean(resids²)                    × 0.1
```

---

## Verification Checklist

1. **Dataset smoke test:**
   ```python
   from devo.data_readers.deep_event_odom import DeepEventOdomDataset
   ds = DeepEventOdomDataset('/path/to/data')
   b = ds[0]
   assert b['images'].shape[1] == 5        # K_sub bins
   assert b['disps'].min() >= 0
   assert b['poses'].shape[-1] == 7
   ```

2. **GN solver shape test:** run `feature_metric_gn_step` with random inputs and verify output pose shape matches input.

3. **Feature gradient check:** after training, confirm `∂fmap/∂pixel` is non-zero at >90% of patch locations.

4. **Training convergence:** `L_pose` should decrease monotonically after the structure-only warmup (first ~1k steps).

5. **Benchmark:** evaluate on DAVIS240C / UZH-FPV against DEIO checkpoint.
