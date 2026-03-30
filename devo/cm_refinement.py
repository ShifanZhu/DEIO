"""Contrast Maximization (CM) refinement module for event-based pose estimation.

Stage 2 of the hybrid feature-based + direct method pipeline:
  - Stage 1 (feature-based) provides a good initial pose estimate
  - Stage 2 (CM, this module) refines it by maximizing the sharpness of the
    Image of Warped Events (IWE)

Three loss methods are provided, selectable via cm_loss_type:
  - "var_iwe":     Global IWE variance (forward splat 200 patches, maximize spread)
  - "ncc":         Per-patch NCC on raw voxels (reuses _sample_features, default)
  - "aligned_var": IWE + dense target voxel combined variance

The cm_refine() function computes:
  - cm_loss: CM quality at the current (feature-based) Gs — differentiable w.r.t.
             Gs, so gradient flows back through BA → Update → model parameters
  - Gs_refined: poses improved by cm_steps gradient ascent iterations (detached,
                for downstream pose use)
"""

import torch
import torch.nn.functional as F
from torch_scatter import scatter_sum

from dpvo.lietorch import SE3
from . import projective_ops as pops
from .feature_metric_gn import _sample_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _downsample_images(images_raw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Bilinear downsample raw voxel grids to feature-map resolution.

    Args:
        images_raw: (B, N, C, H, W) — full-resolution pre-norm voxels
        h, w: target spatial size (feature-map resolution = H/4, W/4)

    Returns:
        (B, N, C, h, w)
    """
    B, N, C, H, W = images_raw.shape
    flat = images_raw.view(B * N, C, H, W)
    small = F.interpolate(flat, size=(h, w), mode='bilinear', align_corners=False)
    return small.view(B, N, C, h, w)


def _sample_frames(
    images_small: torch.Tensor,
    frame_indices: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Sample raw voxel values from per-edge frame assignments.

    Mirrors the loop used in feature_metric_gn.py to handle batches correctly.

    Args:
        images_small: (B, N, C, h, w)
        frame_indices: (E,)  — which frame each edge samples from
        coords:        (B, E, 2) — pixel coords in feature-map space

    Returns:
        (B, E, C)
    """
    B, N, C, h, w = images_small.shape
    E = frame_indices.shape[0]

    sort_idx = torch.argsort(frame_indices)
    f_sorted = frame_indices[sort_idx]         # (E,)
    coords_sorted = coords[:, sort_idx]        # (B, E, 2)

    parts = []
    for frame_idx in range(N):
        mask = (f_sorted == frame_idx)
        if not mask.any():
            continue
        # images_small[:, frame_idx]: (B, C, h, w)
        # coords_sorted[:, mask]:     (B, E_f, 2)
        parts.append(_sample_features(images_small[:, frame_idx], coords_sorted[:, mask]))

    if not parts:
        return torch.zeros(B, E, C, device=images_small.device, dtype=images_small.dtype)

    sampled = torch.cat(parts, dim=1)                   # (B, E, C) in sorted order
    inv_sort = torch.argsort(sort_idx)
    return sampled[:, inv_sort]                         # (B, E, C) in original order


# ---------------------------------------------------------------------------
# Loss methods
# ---------------------------------------------------------------------------

def loss_var_iwe(
    images_small: torch.Tensor,
    Gs,
    patches_cm: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    kk: torch.Tensor,
) -> torch.Tensor:
    """Method A — Global IWE spatial variance.

    Forward-warp the scalar event mass at each of the ~200 patch locations
    from frame ii into frame jj's coordinate system via bilinear splatting.
    Maximise the spatial variance of the resulting Image of Warped Events.

    Limitation: with only ~200 patches on an h×w canvas, the IWE is
    very sparse and variance is dominated by zero pixels. Methods B/C
    may give stronger gradients.

    Args:
        images_small: (B, N, C, h, w)  downsampled raw voxels
        Gs:           SE3 poses (B, N, 7)
        patches_cm:   (B, M, 3, P, P)  patch coords + depths
        intrinsics:   (B, N, 4)        feature-map-resolution intrinsics
        ii, jj, kk:   (E,)             edge indices

    Returns:
        scalar loss (negative IWE variance, to be minimised)
    """
    B, N, C, h, w = images_small.shape
    P = patches_cm.shape[3]
    E = ii.shape[0]

    # Project patch centers from frame ii to frame jj
    coords = pops.transform(Gs, patches_cm, intrinsics, ii, jj, kk)
    coords_center = coords[:, :, P // 2, P // 2, :]   # (B, E, 2)

    # Sample source event mass at patch location (frame ii)
    src_coords = patches_cm[:, kk, :2, P // 2, P // 2]  # (B, E, 2)
    v_src = _sample_frames(images_small, ii, src_coords)  # (B, E, C)
    v_src_sum = v_src.abs().sum(dim=-1)                   # (B, E)

    # Bilinear forward splat into frame jj canvas
    x = coords_center[..., 0]   # (B, E)
    y = coords_center[..., 1]   # (B, E)

    x0 = x.detach().floor().long().clamp(0, w - 1)
    x1 = (x.detach().floor() + 1).long().clamp(0, w - 1)
    y0 = y.detach().floor().long().clamp(0, h - 1)
    y1 = (y.detach().floor() + 1).long().clamp(0, h - 1)

    dx = x - x0.float()   # differentiable fractional parts
    dy = y - y0.float()

    flat_00 = (y0 * w + x0)   # (B, E)
    flat_01 = (y0 * w + x1)
    flat_10 = (y1 * w + x0)
    flat_11 = (y1 * w + x1)

    val_00 = v_src_sum * (1 - dx) * (1 - dy)
    val_01 = v_src_sum * dx * (1 - dy)
    val_10 = v_src_sum * (1 - dx) * dy
    val_11 = v_src_sum * dx * dy

    # Accumulate into IWE — scatter_sum is differentiable through src (val_*)
    # scatter along dim=1, output size = h*w; batch dimension (dim=0) is implicit
    IWE = (
        scatter_sum(val_00, flat_00, dim=1, dim_size=h * w) +
        scatter_sum(val_01, flat_01, dim=1, dim_size=h * w) +
        scatter_sum(val_10, flat_10, dim=1, dim_size=h * w) +
        scatter_sum(val_11, flat_11, dim=1, dim_size=h * w)
    )  # (B, h*w)
    IWE = IWE.view(B, h, w)

    return -IWE.var(dim=(-2, -1)).mean()


def loss_ncc(
    images_small: torch.Tensor,
    Gs,
    patches_cm: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    kk: torch.Tensor,
) -> torch.Tensor:
    """Method B — Per-patch Normalised Cross-Correlation (default).

    For each edge (patch k in frame ii → frame jj):
      1. Sample the raw voxel at the source patch location in frame ii  → v_src (C_bins,)
      2. Project the patch and sample the raw voxel at that location in frame jj → v_tgt (C_bins,)
      3. Compute NCC across the C_bins dimension

    NCC is invariant to overall event density scale, so it handles the
    motion-dependence of raw event counts gracefully.

    Reuses _sample_features() from feature_metric_gn.py directly.

    Args:
        images_small: (B, N, C, h, w)
        Gs:           SE3 poses (B, N, 7)
        patches_cm:   (B, M, 3, P, P)
        intrinsics:   (B, N, 4)
        ii, jj, kk:   (E,)

    Returns:
        scalar loss (mean 1 - NCC, to be minimised)
    """
    B, N, C, h, w = images_small.shape
    P = patches_cm.shape[3]

    coords = pops.transform(Gs, patches_cm, intrinsics, ii, jj, kk)
    coords_center = coords[:, :, P // 2, P // 2, :]   # (B, E, 2)

    # Source: sample at source patch center in frame ii
    src_coords = patches_cm[:, kk, :2, P // 2, P // 2]  # (B, E, 2)
    v_src = _sample_frames(images_small, ii, src_coords)  # (B, E, C)

    # Target: sample at projected location in frame jj
    v_tgt = _sample_frames(images_small, jj, coords_center)  # (B, E, C)

    # NCC across C_bins dimension
    v_src_n = v_src - v_src.mean(dim=-1, keepdim=True)
    v_tgt_n = v_tgt - v_tgt.mean(dim=-1, keepdim=True)
    ncc = (v_src_n * v_tgt_n).sum(dim=-1) / (
        v_src_n.norm(dim=-1) * v_tgt_n.norm(dim=-1) + 1e-6
    )  # (B, E)

    return (1 - ncc).mean()


def loss_aligned_var(
    images_small: torch.Tensor,
    Gs,
    patches_cm: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    kk: torch.Tensor,
) -> torch.Tensor:
    """Method C — IWE + target voxel aligned variance.

    Builds the sparse IWE (same forward warp as Method A) and adds the dense
    target voxel image to it. The dense voxel provides full edge structure as
    background; the sparse IWE provides 200 "votes" for where frame ii's edges
    should land. When the pose is correct, the votes reinforce the same edges
    already present in voxel_j → combined image has higher variance.

    Args:
        images_small: (B, N, C, h, w)
        Gs:           SE3 poses (B, N, 7)
        patches_cm:   (B, M, 3, P, P)
        intrinsics:   (B, N, 4)
        ii, jj, kk:   (E,)

    Returns:
        scalar loss (negative combined variance, to be minimised)
    """
    B, N, C, h, w = images_small.shape
    P = patches_cm.shape[3]

    # Build sparse IWE (same code as Method A)
    coords = pops.transform(Gs, patches_cm, intrinsics, ii, jj, kk)
    coords_center = coords[:, :, P // 2, P // 2, :]   # (B, E, 2)

    src_coords = patches_cm[:, kk, :2, P // 2, P // 2]
    v_src = _sample_frames(images_small, ii, src_coords)
    v_src_sum = v_src.abs().sum(dim=-1)

    x = coords_center[..., 0]
    y = coords_center[..., 1]
    x0 = x.detach().floor().long().clamp(0, w - 1)
    x1 = (x.detach().floor() + 1).long().clamp(0, w - 1)
    y0 = y.detach().floor().long().clamp(0, h - 1)
    y1 = (y.detach().floor() + 1).long().clamp(0, h - 1)
    dx = x - x0.float()
    dy = y - y0.float()

    IWE = (
        scatter_sum(v_src_sum * (1 - dx) * (1 - dy), y0 * w + x0, dim=1, dim_size=h * w) +
        scatter_sum(v_src_sum * dx * (1 - dy),        y0 * w + x1, dim=1, dim_size=h * w) +
        scatter_sum(v_src_sum * (1 - dx) * dy,        y1 * w + x0, dim=1, dim_size=h * w) +
        scatter_sum(v_src_sum * dx * dy,               y1 * w + x1, dim=1, dim_size=h * w)
    ).view(B, h, w)  # (B, h, w)

    # Dense target voxel: average over all unique jj frames, sum over C_bins
    jj_unique = torch.unique(jj)
    voxel_j = images_small[:, jj_unique].abs().sum(dim=2).mean(dim=1)  # (B, h, w)

    combined = IWE + voxel_j
    return -combined.var(dim=(-2, -1)).mean()


# ---------------------------------------------------------------------------
# Main refinement function
# ---------------------------------------------------------------------------

_LOSS_FNS = {
    "var_iwe":     loss_var_iwe,
    "ncc":         loss_ncc,
    "aligned_var": loss_aligned_var,
}


def cm_refine(
    Gs,
    images_raw: torch.Tensor,
    patches_cm: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    kk: torch.Tensor,
    h: int,
    w: int,
    cm_steps: int = 3,
    lr_cm: float = 1e-3,
    cm_loss_type: str = "ncc",
) -> tuple:
    """Contrast Maximization pose refinement.

    Two-part output:
      cm_loss    — CM quality measured at the INPUT Gs (feature-based stage output).
                   Differentiable w.r.t. Gs when Gs is in the computation graph
                   (i.e. when called right after the last BA iteration without
                   detaching Gs). This provides the training signal.

      Gs_refined — Poses improved by cm_steps gradient-ascent iterations on a
                   fully DETACHED copy of Gs. Used as the returned pose for
                   downstream trajectory entries.

    Args:
        Gs:          SE3 poses (B, N, 7) — may or may not be in computation graph
        images_raw:  (B, N, C, H, W)    full-resolution pre-norm voxels
        patches_cm:  (B, M, 3, P, P)    patch coordinates + depths (detached)
        intrinsics:  (B, N, 4)          feature-map-resolution intrinsics
        ii, jj, kk:  (E,)               edge indices
        h, w:        feature-map spatial dimensions (H/4, W/4)
        cm_steps:    number of gradient-ascent steps (default 3)
        lr_cm:       gradient-ascent step size (default 1e-3)
        cm_loss_type: one of "var_iwe", "ncc", "aligned_var"

    Returns:
        (Gs_refined, cm_loss)
    """
    if cm_loss_type not in _LOSS_FNS:
        raise ValueError(f"Unknown cm_loss_type '{cm_loss_type}'. "
                         f"Choose from {list(_LOSS_FNS)}")

    loss_fn = _LOSS_FNS[cm_loss_type]

    # Downsample raw voxels to feature-map resolution once
    images_small = _downsample_images(images_raw, h, w)  # (B, N, C, h, w)

    # ---- Training signal: CM loss at current feature-based pose ----
    # Gradient flows: cm_loss → Gs (via pops.transform) → last BA → Update → model
    cm_loss = loss_fn(images_small, Gs, patches_cm, intrinsics, ii, jj, kk)

    # ---- Inference improvement: gradient ascent on a detached copy ----
    B, N = Gs.shape[0], Gs.shape[1]
    device = Gs.data.device

    Gs_refined = SE3(Gs.data.detach().clone())
    for _ in range(cm_steps):
        delta_xi = torch.zeros(B, N, 6, requires_grad=True, device=device,
                               dtype=Gs.data.dtype)
        Gs_p = SE3.exp(delta_xi) * Gs_refined
        L = loss_fn(images_small, Gs_p, patches_cm, intrinsics, ii, jj, kk)
        grad = torch.autograd.grad(L, delta_xi)[0]   # (B, N, 6)
        with torch.no_grad():
            Gs_refined = SE3(
                (SE3.exp(-lr_cm * grad.detach()) * Gs_refined).data.clone()
            )

    return Gs_refined, cm_loss
