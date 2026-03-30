# Plan: Hybrid Feature-Based + Contrast Maximization Pose Estimation

## Context

The current DEIO architecture is purely feature-based: it extracts 80 patches per frame, computes descriptor correlations across frames, and refines poses via learned BA. The core weakness is that event data is motion-dependent — the same scene generates completely different voxel grids at different speeds, making feature descriptors unreliable across frames.

The fix: add a direct Contrast Maximization (CM) refinement stage **after** the feature-based stage. The feature-based stage solves CM's main weakness (non-convex, sensitive to initialization) by providing a good initial pose. CM solves the feature stage's main weakness (motion-dependent descriptors) by working directly on event physics. A curriculum gate ensures CM only activates once the feature-based stage produces stable enough poses to be a valid initialization.

---

## Frame and Time Window

TartanAir runs at **10 FPS → each voxel covers ~100ms** of events. The 5 C_bins subdivide that 100ms into 5 equal 20ms slices. Frames `i` and `j` are indices into the N_frames sequence dimension (e.g., frame 3 and frame 5), **not** into C_bins. C_bins is the temporal structure *within* one frame's time window.

---

## Architecture Overview

```
Patchifier (modified):
  Sample 3×patches_per_image_cm candidates (e.g. 3×200=600)
  Score all with scorer network → sort by score
  Keep top patches_per_image_cm (200) — all scorer-validated quality patches
  Stage 1 uses top patches_per_image (80) ← unchanged in behavior
  Stage 2 uses top patches_per_image_cm (200) ← superset, 2.5× coverage

Phase 1 — Feature-based (existing, unchanged):
  kk restricted to top-80 patches per frame
  for iter in range(STEPS=12):
    coords = transform(Gs, patches, intrinsics, ii, jj, kk)
    corr   = CorrBlock(kk, jj, coords)
    delta, weight = Update(net, imap, corr, ii, jj, kk)
    Gs, patches = BA(Gs, patches, target, weight, ...)
  → Gs: reasonable pose estimate (within CM convergence basin)

Phase 2 — Direct CM Refinement (new, activated after cm_warmup_steps):
  kk_cm includes all 200 patches per frame
  depths for patches 81–200: initialized from median of Stage 1 depths (same frame)
  for step in range(cm_steps=3):
    IWE = forward_warp_voxels(images_raw, Gs, patches_cm, intrinsics, ii_cm, jj_cm)
    L_cm = -Var(IWE)                   # maximize sharpness
    Gs   = Gs.retr(-lr_cm * ∂L_cm/∂ξ) # gradient ascent on SE3 manifold
  → Gs: CM-refined poses

Return: (traj, cm_loss) when use_cm=True, traj otherwise
```

---

## Critical Files

| File | Role | Change |
|------|------|--------|
| `devo/cm_refinement.py` | New CM module | **Create** |
| `devo/enet.py` | Main network | **Modify**: add CM phase to `eVONet.forward()` |
| `train.py` | Training loop | **Modify**: add CM args, curriculum gate, CM loss |
| `config/train_base.conf` | Config | **Modify**: add CM defaults |

### Reusable Existing Code
- `devo/feature_metric_gn.py::_sample_features()` — bilinear feature sampling via `F.grid_sample` (reuse directly)
- `devo/projective_ops.py::transform()` — patch projection (reuse for CM warping)
- `devo/ba.py::pose_retr()` — SE3 retraction for gradient step (reuse)
- `dpvo/lietorch SE3::retr()` — manifold retraction used in CM gradient step
- `devo/enet.py::Patchifier` — extend to return top-N patches (minor change to existing sampling)

---

## Step-by-Step Implementation

### Step 1: Create `devo/cm_refinement.py`

Three loss methods are implemented under a common `cm_refine()` interface, selected by a `cm_loss_type` argument (`"var_iwe"` / `"ncc"` / `"aligned_var"`). All share the same patch setup, projection, and gradient-ascent loop — only the loss function changes.

---

#### Shared: Patch Projection

All three methods start by projecting the 200 CM patches from source frame `ii` to target frame `jj`:

```python
# pops.transform reused directly — P=1 point patches
coords = pops.transform(Gs, patches_cm, intrinsics, ii, jj, kk)
# coords: (B, E, 1, 1, 2) — projected (x', y') per edge
coords_center = coords[..., 0, 0, :]  # (B, E, 2)
```

---

#### Method A — Global IWE Variance (`cm_loss_type="var_iwe"`)

**What it is**: Forward-warp source voxel values into frame `j`'s coordinate system, build a sparse Image of Warped Events, maximize its spatial variance.

**What the IWE looks like**: A `(H, W)` canvas (e.g., 120×160) with ~200 soft bilinear splats. 99% of pixels are zero. The 200 non-zero dots are at the projected patch locations.

**When correct pose**: 200 high-score patches (all on edges) warp to edge locations in frame `j` → dots cluster along edge lines → higher spatial variance.
**When wrong pose**: dots scatter to non-edge locations → flatter distribution → lower variance.

**Limitation**: global variance is dominated by the 99% zeros; signal is weak for sparse IWEs.

```python
def loss_var_iwe(images_raw, Gs, patches_cm, intrinsics, ii, jj, kk):
    # Project patches
    coords = pops.transform(Gs, patches_cm, intrinsics, ii, jj, kk)
    coords_center = coords[..., 0, 0, :]  # (B, E, 2)

    # Sample source voxel values at patch locations
    # _sample_features from feature_metric_gn.py, reused directly
    v_src = _sample_features(
        images_raw[:, ii].view(B*E, C, H, W),
        patches_cm[:, kk, :2, 0, 0])          # (B, E, C_bins)
    v_src_sum = v_src.abs().sum(dim=-1)        # (B, E) — scalar weight per patch

    # Soft forward splat into frame j canvas
    x, y = coords_center[..., 0], coords_center[..., 1]  # (B, E)
    x0, x1 = x.floor().long().clamp(0, W-1), (x.floor()+1).long().clamp(0, W-1)
    y0, y1 = y.floor().long().clamp(0, H-1), (y.floor()+1).long().clamp(0, H-1)
    dx, dy = x - x0.float(), y - y0.float()   # differentiable fractional parts

    IWE = torch.zeros(B, H*W, device=images_raw.device)
    IWE.scatter_add_(1, (y0*W + x0).view(B,-1), v_src_sum * (1-dx)*(1-dy))
    IWE.scatter_add_(1, (y0*W + x1).view(B,-1), v_src_sum * dx*(1-dy))
    IWE.scatter_add_(1, (y1*W + x0).view(B,-1), v_src_sum * (1-dx)*dy)
    IWE.scatter_add_(1, (y1*W + x1).view(B,-1), v_src_sum * dx*dy)
    IWE = IWE.view(B, H, W)

    # Maximize spatial variance of IWE
    return -IWE.var(dim=(-2,-1)).mean()
```

---

#### Method B — Per-Patch NCC on Raw Voxels (`cm_loss_type="ncc"`)

**What it is**: For each of the 200 patches, backward-sample the raw voxel at both the source location (frame `i`) and the projected location (frame `j`). Compute Normalized Cross-Correlation (NCC) between the two `(C_bins, P, P)` voxel patches. Loss = `mean(1 - NCC)`.

**Why NCC**: NCC is invariant to the overall event density scale — it measures the relative temporal pattern across C_bins, not the absolute count. So fast motion (many events, all in early bins) and slow motion (few events, spread across bins) at the same edge will still produce a high NCC score when correctly aligned.

**Key reuse**: `_sample_features()` from `feature_metric_gn.py` does exactly this backward sampling. This method is structurally identical to `feature_metric_gn_step` but with `images_raw` (C_bins=5) as the "feature map" instead of learned `fmaps` (C=128).

```python
def loss_ncc(images_raw, Gs, patches_cm, intrinsics, ii, jj, kk, P=3):
    # Project patches: get target coords
    coords = pops.transform(Gs, patches_cm, intrinsics, ii, jj, kk)
    coords_center = coords[..., 0, 0, :]  # (B, E, 2)

    # Sample source voxel at patch location (frame i)
    # reuse _sample_features from feature_metric_gn.py
    v_src = _sample_features(
        images_raw.view(B*N, C, H, W)[ii],  # (B, E, C_bins) with P=1
        patches_cm[:, kk, :2, 0, 0])

    # Sample target voxel at projected location (frame j)
    v_tgt = _sample_features(
        images_raw.view(B*N, C, H, W)[jj],
        coords_center)                       # (B, E, C_bins)

    # NCC across C_bins dimension (invariant to density scale)
    v_src_n = v_src - v_src.mean(dim=-1, keepdim=True)
    v_tgt_n = v_tgt - v_tgt.mean(dim=-1, keepdim=True)
    ncc = (v_src_n * v_tgt_n).sum(dim=-1) / (
        v_src_n.norm(dim=-1) * v_tgt_n.norm(dim=-1) + 1e-6)  # (B, E)

    return (1 - ncc).mean()
```

---

#### Method C — IWE + Target Voxel Aligned Variance (`cm_loss_type="aligned_var"`)

**What it is**: Build the sparse IWE (same forward warp as Method A), then **add the dense target voxel** `voxel_j` to it. Maximize variance of the combined image.

**Intuition**: The dense `voxel_j` provides the full edge structure. The sparse IWE provides 200 "votes" for where frame i's edges should land. When T_ij is correct, the 200 warped dots reinforce the same edges already present in `voxel_j` → combined image has sharper, more prominent edges → higher variance. When wrong, the dots land off-edge and add noise to `voxel_j` → lower variance.

This addresses Method A's weakness (sparse IWE dominated by zeros) by using the dense voxel_j as background structure.

```python
def loss_aligned_var(images_raw, Gs, patches_cm, intrinsics, ii, jj, kk):
    # Build sparse IWE (same as Method A)
    IWE = build_iwe(images_raw, Gs, patches_cm, intrinsics, ii, jj, kk)
    # IWE: (B, H, W), sparse — 200 dots

    # Dense target voxel (sum across C_bins)
    voxel_j = images_raw[:, jj_unique].abs().sum(dim=2)  # (B, N_frames, H, W)
    # Align IWE to same frame indices as voxel_j
    combined = IWE + voxel_j  # sparse dots reinforce dense edges

    # Maximize variance of combined image
    return -combined.var(dim=(-2,-1)).mean()
```

---

#### Shared: `cm_refine()` Loop

All three methods plug into the same gradient-ascent loop:

```python
def cm_refine(Gs, images_raw, patches_cm, intrinsics, ii, jj, kk,
              cm_steps=3, lr_cm=1e-3, cm_loss_type="ncc"):
    loss_fn = {"var_iwe": loss_var_iwe,
               "ncc":     loss_ncc,
               "aligned_var": loss_aligned_var}[cm_loss_type]

    for step in range(cm_steps - 1):
        delta_xi = torch.zeros(B, N, 6, requires_grad=True, device=Gs.device)
        Gs_p = Gs.retr(delta_xi)
        L = loss_fn(images_raw, Gs_p, patches_cm, intrinsics, ii, jj, kk)
        grad = torch.autograd.grad(L, delta_xi)[0]   # (B, N, 6)
        Gs = Gs.retr(-lr_cm * grad).detach()

    # Final step: keep graph for training loss backprop
    delta_xi = torch.zeros(B, N, 6, requires_grad=True, device=Gs.device)
    Gs_p = Gs.retr(delta_xi)
    cm_loss = loss_fn(images_raw, Gs_p, patches_cm, intrinsics, ii, jj, kk)
    return Gs_p, cm_loss
```

Key: only the last step keeps the computation graph. Intermediate steps detach so gradients don't accumulate through the loop.

---

#### Method Comparison

| | Method A: `var_iwe` | Method B: `ncc` | Method C: `aligned_var` |
|--|---------------------|-----------------|------------------------|
| IWE type | 200-dot sparse canvas | N/A — per-patch | 200 dots + dense background |
| Handles sparse patches | Poor — zeros dominate variance | Good — per-patch | Good — voxel_j fills gaps |
| Motion-invariant | No — raw counts | Yes — NCC normalizes | Partial |
| Extra data needed | None | None | `voxel_j` at each frame pair |
| Reuses existing code | `torch_scatter` only | `_sample_features` fully reused | Both |
| Gradient quality | Weak (sparse signal) | Strong (per-patch) | Medium |
| Implementation complexity | Medium (forward splat) | Low (backward sample) | Medium |

---

### Step 2: Modify `devo/enet.py`

**In `eVONet.__init__()`**: add CM module and hyperparameters
```python
from .cm_refinement import ContrastMaximization
self.cm = ContrastMaximization(cm_steps=3, lr_cm=1e-3)
```

**In `eVONet.forward()`**: add `use_cm=False`, `cm_steps=3`, `patches_per_image_cm=200` parameters

After the existing `while len(traj) < STEPS:` loop (after line ~835), add:
```python
if use_cm:
    # Build CM patch set: top-200 patches (superset of Stage 1's top-80)
    # patches already contains top-patches_per_image_cm sorted by score
    # Initialize depths for patches 81-200 from median of Stage 1 depths (same frame)
    patches_cm = patches.clone()
    for n in range(images.shape[1]):
        s1_depths = patches[:, ix < patches_per_image * (n+1), 2, 0, 0]
        patches_cm[:, (ix == n) & (torch.arange(...) >= patches_per_image), 2] = s1_depths.median()
    # Build CM edges: all 200 patches × adjacent frames
    kk_cm, jj_cm = flatmeshgrid(torch.arange(patches_cm.shape[1]), torch.arange(n_frames))
    ii_cm = ix[kk_cm]
    k_cm = ii_cm != jj_cm
    ii_cm, jj_cm, kk_cm = ii_cm[k_cm], jj_cm[k_cm], kk_cm[k_cm]
    # Run CM refinement
    Gs, cm_loss = self.cm(Gs, images_raw, patches_cm,
                          intrinsics, ii_cm, jj_cm, kk_cm, cm_steps)
```

**Return signature change**:
```python
if use_cm:
    return traj, cm_loss
return traj
```

Note: `images_raw` = the voxel grids before normalization. Save a reference before the existing normalization step at line ~643. CM should operate on unnormalized voxels (raw event counts, not rescaled).

---

### Step 3: Modify `train.py`

**New arguments** (after line ~404):
```python
parser.add_argument('--cm_warmup_steps', type=int, default=10000,
    help='training steps before activating CM refinement')
parser.add_argument('--cm_weight', type=float, default=0.1,
    help='weight of CM loss')
parser.add_argument('--cm_steps', type=int, default=3,
    help='number of CM gradient ascent steps')
parser.add_argument('--lr_cm', type=float, default=1e-3,
    help='learning rate for CM gradient ascent')
parser.add_argument('--cm_loss_type', type=str, default='ncc',
    choices=['var_iwe', 'ncc', 'aligned_var'],
    help='CM loss: var_iwe=IWE variance, ncc=per-patch NCC, aligned_var=IWE+voxelJ variance')
parser.add_argument('--patches_per_image_cm', type=int, default=200,
    help='number of patches for CM stage (superset of patches_per_image)')
```

**Curriculum gate** (after line ~176 where `so` is computed):
```python
use_cm = (total_steps // args.gpu_num) >= args.cm_warmup_steps
```

**Network call** (line ~180): pass `use_cm` and `cm_steps`:
```python
traj = net(images, poses, disps, intrinsics, M=1024, STEPS=args.iters,
           structure_only=so, patches_per_image=args.patches_per_image,
           use_cm=use_cm, cm_steps=args.cm_steps)
```

**Extract CM loss** (after the `with torch.amp.autocast` block):
```python
cm_loss = torch.as_tensor(0.0)
if use_cm and isinstance(traj, tuple):
    traj, cm_loss = traj
```

**Add to total loss** (after line ~253):
```python
loss += args.cm_weight * cm_loss
```

**Log CM loss** in metrics dict (after line ~284):
```python
"loss/cm_train": cm_loss.item(),
```

---

### Step 4: Modify `config/train_base.conf`

Add:
```
cm_warmup_steps = 10000
cm_weight = 0.1
cm_steps = 3
lr_cm = 1e-3
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Events source | Voxel grids (not raw HDF5) | No data pipeline changes needed |
| Voxels for CM | Pre-normalization (raw counts) | CM needs absolute density for variance signal |
| Warp direction | Forward warp (splat) | True CM: events move from i→j |
| Stage 2 patches | Top-200 scorer patches (superset of Stage 1's top-80) | Scorer-validated quality; no grid sampling or NN depth interp needed |
| Depth for patches 81–200 | Median of Stage 1 depths (same frame) | Same init used in eVONet for new frames (line ~768) |
| Gradient method | Autograd through SE3.retr() | Simpler than analytic Jacobians; LieTorch is differentiable |
| CM loss | Self-supervised (`-Var(IWE)`) | No GT needed; CM is self-supervised by construction |
| Backward compat | Return `(traj, cm_loss)` when `use_cm=True` | Existing eval/inference scripts unaffected |
| CM patch size | P=1 (point patches) | CM needs position accuracy, not neighborhood descriptors |

---

## Verification

1. **Unit test CM module**:
   - Synthesize a voxel grid with a known edge; confirm `forward_warp_voxels` produces a sharp IWE when `Gs` is the identity (no motion) and a blurry IWE when `Gs` is perturbed
   - Verify `cm_refine()` reduces the perturbation (IWE sharpens over iterations)

2. **Training smoke test**:
   ```bash
   python train.py -c config/train_base.conf --steps 100 --cm_warmup_steps 50 \
     --cm_weight 0.1 --cm_steps 2 --name test_cm
   ```
   - Confirm loss/cm_train appears in logs after step 50
   - Confirm no NaN in loss

3. **Comparison**:
   - Train two models: baseline (no CM) vs CM model to 240k steps
   - Evaluate on TartanAir val split using `script/infer/eval_tartan_evs.py`
   - Compare ATE/RPE metrics
