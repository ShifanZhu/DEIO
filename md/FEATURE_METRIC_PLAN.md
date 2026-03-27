# Feature-Metric Odometry Plan

## Motivation

The current DEIO pipeline uses a **search-and-match** approach:
- CorrBlock searches a 7×7 window to find best patch match
- GRU predicts 2D flow correction (delta)
- BA converts flow targets → 6-DOF pose + depth

The proposed approach replaces this with **feature-metric direct alignment**:
- Project source patches into target frame using current pose + depth
- Compute feature residuals between projected and target features
- Solve for pose update analytically via Gauss-Newton on the feature metric

---

## Original Concerns and Resolutions

| Concern | Resolution |
|---|---|
| Weak gradient far from solution | Feature pyramid (coarse-to-fine) expands convergence basin |
| Global geometric consistency | All patch errors accumulated under a single SE3 update |
| Depth unknown | Depth image available — projection is deterministic |
| **6-DOF pose prediction is hard** | **Do not predict SE3 with network — solve analytically via GN** |

The key insight: **the network should never predict SE3 directly**. Pose comes from geometry; the network only learns features and weights.

---

## Proposed Architecture

### Current Pipeline
```
event voxel
    → fnet/inet → fmap, gmap, imap
    → CorrBlock (7×7 search window)    ← search-based matching
    → GRU Update → delta_flow, weight
    → target = coords + delta
    → BA(target, weight)               ← converts flow to pose
    → poses, depths
```

### Proposed Pipeline
```
event voxel
    → fnet/inet → fmap, gmap, imap
    → project patches with current pose + depth
    → e_k = fmap_target[projected] - gmap_source[k]    ← feature residual
    → J_k = (∂fmap/∂pixel) × (∂pixel/∂ξ)              ← chain rule Jacobian
    → GN: δξ = -(Σ w_k J_k^T J_k)⁻¹ (Σ w_k J_k^T e_k)
    → pose += δξ
    → repeat coarse-to-fine
    → poses, depths
```

---

## Component Roles

### Kept from DEIO
| Component | Role |
|---|---|
| `BasicEncoder4Evs` (fnet) | Learn spatially smooth, discriminative features |
| `BasicEncoder4Evs` (inet) | Learn context features for weight prediction |
| `Scorer` | Select informative patches |
| Feature pyramid (2 levels) | Coarse-to-fine for large displacement robustness |
| IMU fusion (GTSAM) | Unchanged — still handles visual-inertial fusion |

### Replaced
| Removed | Replaced by |
|---|---|
| `CorrBlock` (7×7 correlation search) | Direct projection + feature residual |
| `GRU Update` (predict delta_flow) | Analytical Gauss-Newton on feature metric |
| `BA` (flow targets → pose) | GN directly outputs pose update |

### Modified / New
| Component | Role |
|---|---|
| Weight predictor (from `inet`) | GRU or MLP predicts per-patch confidence `w_k` — the one part that cannot be computed analytically |
| `∂fmap/∂pixel` | Spatial gradient of feature map (computed from `fmap`, like image gradients in DSO) |
| `∂pixel/∂ξ` | Standard projection Jacobian — fully analytical given depth |

---

## Gauss-Newton Step (Detail)

For each patch `k` projecting from frame `i` to frame `j`:

```
# 1. Project source patch into target frame
x_proj = π( T_j⁻¹ · T_i · π⁻¹(p_k, z_k) )

# 2. Sample target features at projected location (bilinear)
f_target = fmap[j](x_proj)

# 3. Feature residual
e_k = f_target - gmap[k]         # shape: (128,)

# 4. Jacobians
∂pixel/∂ξ  = projection Jacobian  (analytical, shape: 2×6)
∂fmap/∂pixel = feature map gradient (from fmap spatial gradients, shape: 128×2)
J_k = ∂fmap/∂pixel × ∂pixel/∂ξ   # shape: 128×6

# 5. Accumulate normal equations
H += w_k * J_k^T J_k              # 6×6
b += w_k * J_k^T e_k              # 6×1

# 6. Solve (with LM damping for stability)
δξ = -(H + λI)⁻¹ b               # 6-DOF pose update
pose = pose · exp(δξ)             # SE3 retraction
```

Repeat for all patches, then iterate (coarse-to-fine).

---

## Training Losses

| Loss | Purpose |
|---|---|
| `L_feature` | Feature consistency: minimize `||fmap_target[projected] - gmap_source||²` — encourages spatially smooth, matchable features |
| `L_pose` | Pose error vs ground truth (same as current DEIO) |
| `L_weight` | Train weight predictor to assign low confidence to unreliable patches |

The feature consistency loss explicitly encourages `fnet` to produce features with **strong, non-zero spatial gradients** — critical for GN convergence.

---

## Relationship to Existing Systems

| System | Similarity |
|---|---|
| **DSO** | GN on direct photometric error — same optimization structure, but raw pixels instead of learned features |
| **SuperPoint + BA** | Learned features, but uses 2D matching then pose — not direct alignment |
| **DROID-SLAM / DEIO** | 2D flow → BA — current approach |
| **This proposal** | Learned features + GN directly on feature metric — DSO with learned features |

---

## Key Advantages over Current DEIO

1. **No intermediate flow representation** — pose comes directly from feature alignment
2. **Geometrically principled** — network never predicts SE3; geometry is exact
3. **Simpler architecture** — removes CorrBlock, GRU, and BA as separate components
4. **Better use of depth** — depth makes projection deterministic and removes scale ambiguity
5. **Feature learning is guided** — the GN loss directly encourages features useful for alignment

---

## Key Risks

1. **Feature gradient quality** — `fnet` must produce spatially smooth features with non-zero gradients. Requires careful loss design.
2. **Convergence basin** — even with feature pyramid, fast/large motions may exceed the basin. Hybrid approach (coarse search + GN refinement) may be needed.
3. **Rotation conditioning** — `∂pixel/∂ξ` can be ill-conditioned for rotation at certain depths. LM damping required.
4. **Depth noise** — real depth maps have noise/holes. Need robust weighting for invalid depth regions.

---

## Suggested Implementation Steps

1. Keep `fnet`/`inet`/`Scorer` unchanged
2. Add spatial gradient computation on `fmap` (∂fmap/∂pixel)
3. Implement analytical projection Jacobian `∂pixel/∂ξ` (already partially in `projective_ops.py`)
4. Replace `CorrBlock` + `Update` + `BA` with feature-metric GN solver
5. Add MLP weight predictor fed from `inet` features
6. Add `L_feature` consistency loss to training
7. Validate on DAVIS240C / UZH-FPV benchmarks vs current DEIO
