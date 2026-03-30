# DEIO System Components

## Overview

The system has two phases with a clean separation:

- **Training**: only the neural network (`eVONet` or `FMNet`) is trained on short fixed-length clips
- **Inference**: the frozen network is embedded inside a full VIO SLAM system (`DBA`) that handles long sequences, IMU, loop closure, and global BA

---

## Component Map

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TRAINING — two separate tracks                    │
│                                                                      │
│  Track 1: DEIO (indirect/learned)       Track 2: FMNet (direct)     │
│  ─────────────────────────────          ──────────────────────────   │
│  train.py → eVONet (enet.py)            train_fm.py → FMNet          │
│      │                                      │                        │
│      ├─ Patchifier (patch extract)          ├─ fnet (feature extractor)
│      ├─ CorrBlock (correlation vol)         ├─ inet (context extractor)
│      ├─ Update GRU (learned flow)           ├─ weight_head MLP        │
│      └─ ba.py (diff. visual BA)             └─ feature_metric_gn.py  │
│           ↓                                      ↓                   │
│      optional: cm_refine()              Gauss-Newton on feature      │
│      (CM loss as extra training signal) residuals, no GRU, no corr  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         INFERENCE                                    │
│                                                                      │
│  dba.py (DBA)  — full VIO SLAM                                       │
│       │                                                              │
│       ├── loads frozen eVONet (.pth)                                 │
│       │      ↓                                                       │
│       │   per frame: CorrBlock + GRU → flow δ, weights Σ            │
│       │      ↓                                                       │
│       │   ba.py → visual Hessian (H, v)                              │
│       │                                                              │
│       ├── PatchGraph  (active + inactive edges, loop closure)        │
│       │                                                              │
│       └── MultiSensorState + GTSAM                                  │
│              ↓                                                       │
│           CustomHessianFactor(H, v) + CombinedImuFactor             │
│              ↓                                                       │
│           LM optimize → poses, velocities, biases                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Reference

### `devo/enet.py` — `eVONet` (indirect, learned)

The only thing trained in the standard pipeline. A patch-based recurrent network:

- **Patchifier**: extracts event patches from voxel grids, selects informative ones via learned Scorer
- **CorrBlock**: builds multi-scale correlation volume between source patch features and target feature map
- **Update (GRU)**: recurrent operator that predicts flow corrections `δ` and confidence weights `Σ` per edge, using correlation features + context features + neighbor message passing
- **ba.py** (called internally): differentiable visual BA that solves reprojection minimization over the active window, outputs updated poses and depths

Trained on short fixed-length clips (~15 frames). Saved as `.pth`, frozen at inference.

---

### `devo/ba.py` — Differentiable Visual BA (CUDA)

- Called inside `eVONet.forward()` at every BA iteration
- Pure visual, no IMU, no GTSAM
- Solves the Schur-complement system for poses and inverse depths
- At inference, also produces the visual Hessian `(H, v)` that gets passed into GTSAM as a `CustomHessianFactor`

---

### `devo/devo2.py` — `DEVO` (visual-only SLAM wrapper)

- Loads frozen `eVONet`, runs it frame-by-frame on a sliding window
- Manages `PatchGraph`: tracks active vs inactive edges, handles keyframe removal, generates loop closure candidates via `edges_loop()`
- Uses `fastba` (CUDA) for windowed BA
- No IMU

---

### `devo/dba.py` — `DBA` (full VIO SLAM, used for inference)

Everything in `DEVO` plus IMU fusion via GTSAM:

- Loads and runs frozen `eVONet` per frame
- Maintains `PatchGraph` (same as DEVO) for visual tracking
- Adds `MultiSensorState`: poses, velocities, biases as GTSAM variables
- Builds factor graph each update with:
  - `CombinedImuFactor` between consecutive frames
  - `CustomHessianFactor` wrapping the visual BA Hessian
- Runs Levenberg-Marquardt to optimize poses + velocities + biases jointly
- Handles Visual-IMU alignment (gyro bias, gravity direction, scale) at initialization
- Marginalizes old states out of the factor graph

The coupling between neural network output and GTSAM:
```
eVONet → flow δ, weights Σ
       ↓
ba.py → visual Hessian (H, v)
       ↓
CustomHessianFactor(H, v)  →  GTSAM factor graph
       ↓
joint optimization with CombinedImuFactor
       ↓
poses, velocities, biases
```

---

## Direct Method Components

### `devo/feature_metric_gn.py` — Feature-Metric Gauss-Newton Solver

A direct-method alternative to CorrBlock + GRU + BA. For each patch-to-frame edge:

1. Project source patch into target frame using current pose + depth
2. Compute feature residual: `e = f_target - f_source`
3. Compute Jacobian: `J = (∂f/∂pixel) @ (∂pixel/∂pose)`
4. Accumulate normal equations: `H += wJᵀJ`, `b += wJᵀe`
5. Solve: `δξ = -(H + λI)⁻¹ b`
6. Retract poses and depths

Reuses `CholeskySolver`, `pose_retr`, `disp_retr` from `ba.py`. Depths **are** updated here (unlike CM refinement).

---

### `devo/fmnet.py` — `FMNet` (direct, alternative network)

A complete replacement for `eVONet`. Same patch extraction and scoring, but replaces the CorrBlock + GRU with `feature_metric_gn`. Trained via `train_fm.py`. Not used in standard DEIO inference.

- **fnet**: feature extractor with instance norm (for matching invariance)
- **inet**: context extractor (no norm, to preserve absolute values)
- **weight_head MLP**: `inet` features → per-patch confidence in `[0, 1]`
- **feature_metric_gn**: coarse-to-fine GN solver (level 2 then level 1)

---

### `devo/cm_refinement.py` — Contrast Maximization Refinement

An optional add-on to `eVONet` (not a replacement). Applied after the GRU BA loop. Uses raw event voxels directly — no learned parameters of its own.

**Two separate outputs with different purposes:**

| Output | Gradient | Purpose |
|--------|----------|---------|
| `cm_loss` | Attached to `Gs` → flows through BA → GRU → network weights | Training signal: teaches network to produce poses that maximize event sharpness |
| `Gs_refined` | Detached copy, gradient ascent only | Better poses for trajectory output, no network update |

**Three loss variants** (selected via `cm_loss_type`):
- `ncc` (default): per-patch NCC on raw voxels — invariant to event density scale
- `var_iwe`: global IWE spatial variance via forward splatting
- `aligned_var`: sparse IWE + dense target voxel combined variance

**Key design points:**
- Only refines **poses** (6-DOF), not depths — depths are detached before being passed in
- Activated on a curriculum: `cm_warmup_steps` delays until network converges on pose loss first, then `cm_ramp_steps` linearly ramps the weight to avoid destabilizing early training
- Gradient chain: `cm_loss → Gs → ba.py → GRU Update → eVONet weights`

---

## Summary Table

| Component | Type | Role | Used in |
|-----------|------|------|---------|
| `enet.py / eVONet` | Indirect (learned corr + GRU) | Main network | Training + Inference |
| `ba.py` | Differentiable visual BA (CUDA) | Inside eVONet forward | Training + Inference |
| `feature_metric_gn.py` | Direct (feature-metric GN) | Solver for FMNet | FMNet training only |
| `fmnet.py / FMNet` | Direct (learned features + GN) | Alternative to eVONet | `train_fm.py` only |
| `cm_refinement.py` | Direct (contrast maximization) | Optional add-on refiner | Training (optional) + Inference |
| `devo2.py / DEVO` | Visual SLAM wrapper | Wraps eVONet, no IMU | Inference only |
| `dba.py / DBA` | VIO SLAM wrapper | Wraps eVONet + GTSAM IMU | Inference only |

---

## Training vs Inference Gap

Training operates on short fixed clips; inference runs on continuous long sequences via the full SLAM system. The network never sees the SLAM machinery during training:

- `dba.py` / `devo2.py` are never instantiated during training
- Loop closure, inactive edge buffers (`ii_inac/jj_inac/kk_inac`), GTSAM, and IMU preintegration are inference-only
- The `_inac` buffers grow without bound at inference (never pruned), causing per-frame cost to increase linearly with sequence length — the primary performance bottleneck on long sequences
