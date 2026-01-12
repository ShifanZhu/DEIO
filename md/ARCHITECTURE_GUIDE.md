# DEIO Architecture Guide

This document provides a comprehensive guide to understanding the DEIO (Deep Event Inertial Odometry) codebase, now with extensive inline comments.

## Overview
 
DEIO is a learning-based event-inertial odometry system that combines:
- **Event-based recurrent optical flow** (eVONet) - Learning component
- **IMU pre-integration** (GTSAM) - Traditional optimization
- **Graph-based optimization** - Tight coupling of visual and inertial data

**Paper Reference**: "DEIO: Deep Event Inertial Odometry" (ICCV 2025)

---

## File Structure

### Core Network: `devo/enet.py` (FULLY COMMENTED)

This file implements the **eVONet** architecture - the learning component of DEIO.

#### Key Classes:

1. **`Update`** (Lines 54-174)
   - **Purpose**: Recurrent update operator for optical flow prediction
   - **Paper Reference**: Section III-A, Equation (2)
   - **What it does**:
     - Predicts optical flow corrections `δ_inj` (2D displacement)
     - Predicts confidence weights `Σ_inj` (uncertainty estimates)
     - Uses GRU-style recurrent updates for temporal consistency
     - Aggregates information from neighboring edges in the patch graph
   - **Key Methods**:
     - `forward()`: Combines correlation, context, and hidden state to predict flow

2. **`Patchifier`** (Lines 177-380)
   - **Purpose**: Extract and select informative event patches
   - **Paper Reference**: Section III-A, Equation (1)
   - **What it does**:
     - Extracts dense features using two CNNs (`fnet` for matching, `inet` for context)
     - Selects N patches per frame (default: 80, paper uses 96)
     - Three selection strategies:
       - **SCORER** (default): Learned importance network
       - **GRADIENT**: Event gradient-based (hand-crafted)
       - **RANDOM**: Baseline
     - Extracts patch-based features at selected locations
   - **Key Methods**:
     - `forward()`: Full pipeline from voxels to patch features

3. **`CorrBlock`** (Lines 383-436)
   - **Purpose**: Multi-scale correlation volume for patch matching
   - **Paper Reference**: Section III-A (correlation layer)
   - **What it does**:
     - Builds feature pyramid at multiple scales (levels [1, 4])
     - Computes correlation in 7×7 search windows
     - Measures visual similarity between patches across frames
   - **Key Methods**:
     - `__call__()`: Compute correlation features for given edges

4. **`eVONet`** (Lines 439-746)
   - **Purpose**: Main network combining all components
   - **Paper Reference**: Section III-A, Figure 1
   - **What it does**:
     - Orchestrates the complete pipeline:
       1. Normalize event voxels
       2. Extract patches and features
       3. Build patch co-visibility graph
       4. Iteratively refine poses and depths
       5. Return supervision signals for training
   - **Key Methods**:
     - `forward()`: Complete training/inference pipeline

---

## Data Flow Through eVONet

```
Event Voxels (B, N_frames, 5, H, W)
    ↓
[1] Normalization (std/rescale)
    ↓
[2] Feature Extraction (Patchifier)
    ├─→ fnet: Matching features (128-dim)
    ├─→ inet: Context features (384-dim)
    └─→ scorer: Patch selection scores
    ↓
Selected Patches (N_frames × 80 = N_patches)
    ├─→ gmap: Patch matching features (B, N_patches, 128, 3, 3)
    ├─→ imap: Patch context features (B, N_patches, 384, 1, 1)
    └─→ patches: Coordinates + depths (B, N_patches, 3, 3, 3)
    ↓
[3] Build Co-visibility Graph
    ├─→ Edges (i→j): Patch from frame i to frame j
    ├─→ Initialize 8 frames, then incremental
    └─→ Hidden state: net (B, num_edges, 384)
    ↓
[4] Iterative Refinement (STEPS=12 iterations)
    ├─→ a) Project patches: coords = π[T_j^(-1) · T_i · π^(-1)(P)]
    ├─→ b) Correlation: corr = CorrBlock(coords)
    ├─→ c) Update: (δ, Σ) = Update(net, imap, corr)
    ├─→ d) Bundle Adjustment: Optimize poses & depths
    └─→ e) Collect supervision signals
    ↓
[5] Return Trajectory
    └─→ List of (coords, coords_gt, poses, poses_gt, scores)
        for computing L_pose, L_flow, L_score
```

---

## Key Concepts Explained

### 1. Patch Representation (Equation 1)

```python
# Each patch P_in contains:
P_in = [x_i, y_i, 1, d_i]  # Shape: (1×p²) for each coordinate

# In code:
patches[:, n, 0, :, :]  # x-coordinates (3×3 grid)
patches[:, n, 1, :, :]  # y-coordinates (3×3 grid)
patches[:, n, 2, :, :]  # depths (constant per patch)
```

### 2. Reprojection Error (Equation 2)

```python
# Paper equation:
# e = Σ ||π[T_j^(-1)·T_i·π^(-1)(P̂_in)] - [P̂_in + δ_inj]||²_Σ_inj

# In code (enet.py:690-697):
coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)  # Projection
target = coords[..., p//2, p//2, :] + delta  # Add flow correction
Gs, patches = BA(Gs, patches, intrinsics, target, weight, ...)  # Minimize error
```

### 3. Co-visibility Graph

- **Nodes**: Keyframes (camera poses)
- **Patches**: Visual features tracked across frames
- **Edges**: Connections (patch i → frame j)
- **Graph indices**:
  - `ii`: Source frame for each edge
  - `jj`: Target frame for each edge
  - `kk`: Patch index for each edge

Example: With 8 frames and 80 patches/frame:
- Total patches: 640
- Each patch connects to all 8 frames
- Total edges: 640 × 8 = 5120

### 4. Training Losses (Section III-B)

```python
# Total loss = 10·L_pose + 0.1·L_flow + 0.05·L_score

# L_pose: Pose error (SE3 distance)
L_pose = Σ ||log_SE3(T_GT^(-1) · T_pred)||

# L_flow: Optical flow error (2D pixel distance)
L_flow = Σ ||coords_pred - coords_gt||

# L_score: Patch selection supervision
L_score = BCE(scores, tracking_quality)
```

---

## Integration with IMU (dba.py)

The eVONet output (Hessian information) is integrated with IMU in `devo/dba.py`:

1. **Extract Hessian**: From differentiable BA (Equation 10)
   ```python
   H_gg = [B - EC^(-1)E^T]^(-1)  # Pose-pose Hessian
   V_gg = v - EC^(-1)u           # Residual vector
   ```

2. **Event Residual Factor** (Equation 14):
   ```python
   r_event = 0.5 * ξ^T · H_gg · ξ - ξ^T · V_gg
   ```

3. **IMU Residual Factor** (Equation 15):
   ```python
   r_imu = [position, velocity, rotation, bias] errors
   ```

4. **GTSAM Optimization** (Equation 13):
   ```python
   J(χ) = ||r_event||² + ||r_imu||² + ||r_marginalization||²
   ```

---

## Code Navigation Tips

### Finding Specific Components:

1. **Patch Selection**:
   - `enet.py:276-337` - Three selection methods
   - `selector.py` - Scorer network implementation

2. **Feature Extraction**:
   - `extractor.py` - BasicEncoder4Evs
   - `res_net_extractor.py` - ResNetFPN alternative

3. **Correlation Computation**:
   - `enet.py:383-436` - CorrBlock class
   - `dpvo/altcorr/` - CUDA correlation kernels

4. **Bundle Adjustment**:
   - `ba.py` - Differentiable BA (CPU)
   - `dpvo/fastba/` - Fast BA (GPU/CUDA)

5. **Graph Optimization**:
   - `dba.py` - DEIO with IMU integration
   - `multi_sensor.py` - IMU pre-integration

### Understanding Shapes:

All shapes are documented in comments. Key pattern:
```python
# Example from enet.py:581-587
# fmap: (B, N_frames, 128, H/4, W/4) - Dense matching features
# gmap: (B, N_patches, 128, 3, 3) - Patch matching features
# imap: (B, N_patches, 384, 1, 1) - Patch context features
```

---

## Common Configurations

### Training (train.py):

```bash
python train.py \
    --datapath=/path/to/tartan \
    --evs  # Use event voxels
    --n_frames=32 \
    --batch_size=1 \
    --gpus=2
```

### Evaluation (script/eval_deio/davis240c.py):

```bash
python script/eval_deio/davis240c.py \
    --inputdir=/data/davis240c \
    --config=config/davis240c.yaml \
    --enable_event \
    --network=DEVO.pth
```

---

## Performance Notes

From Paper Table I (DAVIS240c average):
- **DEVO** (event-only): 0.21% MPE
- **DEIO** (event+IMU): **0.06% MPE** (71% improvement!)

Key innovation: Tight integration of learned features with graph optimization.

---

## Further Reading

1. **Paper Sections**:
   - Section III-A: Network architecture
   - Section III-B: Training supervision
   - Section III-C: Hessian derivation
   - Section III-D: IMU integration
   - Section III-E: System overview

2. **Related Code**:
   - `devo/devo.py` - Event-only version (no IMU)
   - `dpvo/dpvo.py` - RGB baseline
   - `train.py` - Training loop with losses

3. **Datasets**:
   - DAVIS240c: Most classical benchmark
   - UZH-FPV: Drone racing
   - VECtor: Large-scale outdoor
   - See Paper Section IV for full list

---

## Questions?

For implementation details, see the extensive inline comments in:
- **`devo/enet.py`** - Complete network architecture
- **`devo/dba.py`** - IMU integration (next file to comment)
- **`train.py`** - Training pipeline

Each function now has detailed docstrings explaining:
- What it does
- How it relates to the paper
- Input/output shapes
- Key implementation details
