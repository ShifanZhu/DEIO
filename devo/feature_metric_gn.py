"""Feature-Metric Gauss-Newton Solver for event-based odometry.

Replaces the CorrBlock + GRU Update + BA pipeline from DEIO with a direct
feature-metric alignment approach ("DSO with learned features"):

    For each edge (source patch k in frame i → target frame j):
        1. Project source patch into target frame using current pose + depth.
        2. Sample target feature map at the projected location (bilinear).
        3. Compute feature residual:  e_k = f_target - f_source
        4. Compute Jacobian via chain rule:
               J_k = (∂fmap/∂pixel) @ (∂pixel/∂ξ)     shape: (C, 6)
        5. Accumulate weighted normal equations:
               H += w_k * J_k^T J_k     (6×6)
               b += w_k * J_k^T e_k     (6,)
        6. Solve:  δξ = -(H + λI)^{-1} b
        7. Retract pose and depth.

Reuses from ba.py:
    CholeskySolver, safe_scatter_add_mat, safe_scatter_add_vec, pose_retr, disp_retr

Reuses from projective_ops.py:
    transform(..., jacobian=True)  →  coords, valid, (Ji, Jj, Jz)
"""

import torch
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .ba import CholeskySolver, safe_scatter_add_mat, safe_scatter_add_vec, pose_retr, disp_retr
from . import projective_ops as pops


def _fmap_spatial_gradient(fmap: torch.Tensor) -> torch.Tensor:
    """Compute spatial gradients of a feature map.

    Args:
        fmap: (B, C, H, W)

    Returns:
        grad: (B, C, 2, H, W)  — [∂fmap/∂x, ∂fmap/∂y] via central differences.
    """
    # Pad by 1 on each spatial edge (replicate) so output has same spatial size.
    fp = F.pad(fmap, (1, 1, 1, 1), mode='replicate')
    dx = (fp[:, :, 1:-1, 2:] - fp[:, :, 1:-1, :-2]) * 0.5   # ∂/∂x
    dy = (fp[:, :, 2:, 1:-1] - fp[:, :, :-2, 1:-1]) * 0.5   # ∂/∂y
    return torch.stack([dx, dy], dim=2)  # (B, C, 2, H, W)


def _sample_features(fmap: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinear sample features at sub-pixel coordinates.

    Args:
        fmap:   (B, C, H, W)
        coords: (B, E, 2)  — pixel coordinates [x, y] in feature-map space

    Returns:
        feats: (B, E, C)
    """
    B, C, H, W = fmap.shape
    E = coords.shape[1]

    # Normalise to [-1, 1] for grid_sample
    x_norm = 2.0 * coords[..., 0] / (W - 1) - 1.0  # (B, E)
    y_norm = 2.0 * coords[..., 1] / (H - 1) - 1.0  # (B, E)
    grid = torch.stack([x_norm, y_norm], dim=-1)     # (B, E, 2)
    grid = grid.unsqueeze(2)                          # (B, E, 1, 2)

    # grid_sample expects (B, C, H_out, W_out) grid shape (B, H_out, W_out, 2)
    sampled = F.grid_sample(fmap, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
    # sampled: (B, C, E, 1) → (B, E, C)
    return sampled.squeeze(-1).permute(0, 2, 1)


def _sample_gradient(grad: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinear sample feature-map gradients at sub-pixel coordinates.

    Args:
        grad:   (B, C, 2, H, W)
        coords: (B, E, 2)

    Returns:
        g: (B, E, C, 2)   — [∂f/∂x, ∂f/∂y] at each location
    """
    B, C, _, H, W = grad.shape
    E = coords.shape[1]

    x_norm = 2.0 * coords[..., 0] / (W - 1) - 1.0
    y_norm = 2.0 * coords[..., 1] / (H - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # (B, E, 1, 2)

    # Merge C and 2 into one dim for a single grid_sample call
    grad_flat = grad.view(B, C * 2, H, W)
    sampled = F.grid_sample(grad_flat, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
    # sampled: (B, C*2, E, 1) → (B, E, C, 2)
    return sampled.squeeze(-1).permute(0, 2, 1).view(B, E, C, 2)


def feature_metric_gn_step(
    fmaps:      torch.Tensor,   # (B, N, C, H, W)
    patches:    torch.Tensor,   # (B, M, 3, P, P)  — (x, y, disp) patch grids
    poses,                      # SE3 object, shape (B, N, 7)
    intrinsics: torch.Tensor,   # (B, 4)  [fx, fy, cx, cy] at feature-map resolution
    ii:         torch.Tensor,   # (E,)  source frame indices
    jj:         torch.Tensor,   # (E,)  target frame indices
    kk:         torch.Tensor,   # (E,)  patch indices (into patches dim 1)
    weights:    torch.Tensor,   # (B, E)  per-edge scalar confidence
    fmaps_grad: torch.Tensor = None,  # (B, N, C, 2, H, W) precomputed; recomputed if None
    ep:         float = 1.0,
    lm:         float = 1e-4,
    fixedp:     int = 1,
) -> tuple:
    """One Gauss-Newton iteration on the feature metric.

    Returns:
        poses:   updated SE3
        patches: updated patches (inverse-depth updated)
        resids:  (B, E, C) feature residuals (detached) for logging / loss
    """
    B = fmaps.shape[0]
    N = fmaps.shape[1]
    C = fmaps.shape[2]
    P = patches.shape[3]
    E = ii.shape[0]

    # ------------------------------------------------------------------
    # 1. Project source patches → target frame, get pixel Jacobians
    # ------------------------------------------------------------------
    # pops.transform expects intrinsics (B, N, 4); expand from (B, 4) if needed.
    N_poses = poses.shape[1]
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(1).expand(-1, N_poses, -1)

    # transform returns:
    #   coords: (B, E, P, P, 2) — projected pixel locations in target frame
    #   valid:  (B, E)          — 1.0 if centre point has positive depth
    #   Ji:     (1, E, 2, 6)   — ∂pixel/∂ξ_i
    #   Jj:     (1, E, 2, 6)   — ∂pixel/∂ξ_j
    #   Jz:     (1, E, 2, 4)   — ∂pixel/∂z (we fold into depth update)
    coords, valid, (Ji, Jj, Jz) = pops.transform(
        poses, patches, intrinsics, ii, jj, kk, jacobian=True)

    # Patch centre pixel in target frame
    pc = coords[:, :, P // 2, P // 2, :]  # (B, E, 2)

    # ------------------------------------------------------------------
    # 2. Sample source features (gmap equivalent — features at patch centres)
    # ------------------------------------------------------------------
    # Use source fmap at source patch centre (fixed reference)
    src_coords_flat = patches[:, kk, :, P // 2, P // 2]  # (B, E, 3) — (x, y, disp)
    src_xy = src_coords_flat[:, :, :2]                    # (B, E, 2)

    # Sample source features — O(N) passes, each fully vectorised over its edges.
    # Sort edges by source frame so we can torch.cat results and reorder once.
    src_sort = torch.argsort(ii)
    ii_s = ii[src_sort]
    src_xy_s = src_xy[:, src_sort]
    f_src_parts = []
    for frame_idx in range(N):
        mask = (ii_s == frame_idx)
        if not mask.any():
            continue
        f_src_parts.append(
            _sample_features(fmaps[:, frame_idx], src_xy_s[:, mask]))  # (B, E_f, C)
    f_src = torch.cat(f_src_parts, dim=1)[:, torch.argsort(src_sort)]  # (B, E, C)

    # ------------------------------------------------------------------
    # 3. Sample target features and gradient at projected location
    # ------------------------------------------------------------------
    tgt_sort = torch.argsort(jj)
    jj_s = jj[tgt_sort]
    pc_s  = pc[:, tgt_sort]
    f_tgt_parts = []
    g_tgt_parts = []
    for frame_idx in range(N):
        mask = (jj_s == frame_idx)
        if not mask.any():
            continue
        fmap_j   = fmaps[:, frame_idx]                    # (B, C, H, W)
        coords_j = pc_s[:, mask]                          # (B, E_f, 2)
        f_tgt_parts.append(_sample_features(fmap_j, coords_j))       # (B, E_f, C)
        if fmaps_grad is not None:
            grad_j = fmaps_grad[:, frame_idx]             # (B, C, 2, H, W) — precomputed
        else:
            grad_j = _fmap_spatial_gradient(fmap_j)       # fallback
        g_tgt_parts.append(_sample_gradient(grad_j, coords_j))       # (B, E_f, C, 2)
    inv_tgt = torch.argsort(tgt_sort)
    f_tgt = torch.cat(f_tgt_parts, dim=1)[:, inv_tgt]    # (B, E, C)
    g_tgt = torch.cat(g_tgt_parts, dim=1)[:, inv_tgt]    # (B, E, C, 2)

    # ------------------------------------------------------------------
    # 4. Feature residual
    # ------------------------------------------------------------------
    resids = f_tgt - f_src                               # (B, E, C)

    # ------------------------------------------------------------------
    # 5 & 6. Pixel-space collapse — standard photometric-BA trick.
    #
    # Instead of lifting to (B, E, C, 6) Jacobians and doing large matmuls,
    # aggregate over the C feature dimension first to obtain a (B, E, 2, 2)
    # pixel-space information matrix G and a (B, E, 2, 1) RHS vector g_r.
    # Then the 6×6 normal-equation blocks are cheap 2-D matmuls:
    #
    #   G    = (w·g_tgt)^T @ g_tgt          (B, E, 2, 2)
    #   g_r  = (w·g_tgt)^T @ resids         (B, E, 2, 1)
    #   Bjj  = Jj_^T @ G @ Jj_              (B, E, 6, 6)
    #   vj   = Jj_^T @ g_r                  (B, E, 6)
    #
    # FLOPs: ~40× fewer than the naive (C=128)-dimensional formulation.
    # ------------------------------------------------------------------
    Jj_ = Jj.expand(B, -1, -1, -1)          # (B, E, 2, 6)
    Ji_ = Ji.expand(B, -1, -1, -1)          # (B, E, 2, 6)
    Jz_ = Jz.expand(B, -1, -1, -1)          # (B, E, 2, 1)

    w  = (valid * weights)[..., None, None]  # (B, E, 1, 1)
    wg = (w * g_tgt).transpose(-1, -2)      # (B, E, 2, C)  — weighted feature gradient

    # Pixel-space information matrix and RHS (collapse C dimension)
    G   = wg @ g_tgt                         # (B, E, 2, 2)
    g_r = wg @ resids.unsqueeze(-1)          # (B, E, 2, 1)

    Jj_T = Jj_.transpose(-1, -2)            # (B, E, 6, 2)
    Ji_T = Ji_.transpose(-1, -2)            # (B, E, 6, 2)
    Jz_T = Jz_.transpose(-1, -2)            # (B, E, 1, 2)

    # Normal equations: (6,2)@(2,2)@(2,6) and (6,2)@(2,1)
    GJj = G @ Jj_                            # (B, E, 2, 6)
    GJi = G @ Ji_                            # (B, E, 2, 6)
    GJz = G @ Jz_                            # (B, E, 2, 1)

    Bii = Ji_T @ GJi                         # (B, E, 6, 6)
    Bij = Ji_T @ GJj                         # (B, E, 6, 6)
    Bji = Jj_T @ GJi                         # (B, E, 6, 6)
    Bjj = Jj_T @ GJj                         # (B, E, 6, 6)

    vi  = (Ji_T @ g_r).squeeze(-1)           # (B, E, 6)
    vj  = (Jj_T @ g_r).squeeze(-1)           # (B, E, 6)

    C_diag = Jz_T @ GJz                      # (B, E, 1, 1)
    Eik    = Ji_T @ GJz                      # (B, E, 6, 1)
    Ejk    = Jj_T @ GJz                      # (B, E, 6, 1)
    w_disp = Jz_T @ g_r                      # (B, E, 1, 1)

    # ------------------------------------------------------------------
    # 7. Scatter-accumulate into per-frame pose system
    # ------------------------------------------------------------------
    n_fix = fixedp
    ii_s  = ii - n_fix
    jj_s  = jj - n_fix
    n_free = (max(ii.max().item(), jj.max().item()) + 1) - n_fix

    kx, kk_inv = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx)

    # Pose Hessian (6n × 6n block-sparse)
    B_mat = (
        safe_scatter_add_mat(Bii, ii_s, ii_s, n_free, n_free).view(B, n_free, n_free, 6, 6) +
        safe_scatter_add_mat(Bij, ii_s, jj_s, n_free, n_free).view(B, n_free, n_free, 6, 6) +
        safe_scatter_add_mat(Bji, jj_s, ii_s, n_free, n_free).view(B, n_free, n_free, 6, 6) +
        safe_scatter_add_mat(Bjj, jj_s, jj_s, n_free, n_free).view(B, n_free, n_free, 6, 6)
    )

    # RHS gradient
    v_mat = (
        safe_scatter_add_vec(vi, ii_s, n_free).view(B, n_free, 1, 6, 1) +
        safe_scatter_add_vec(vj, jj_s, n_free).view(B, n_free, 1, 6, 1)
    )

    # Depth normal equations
    E_mat = (
        safe_scatter_add_mat(Eik, ii_s, kk_inv, n_free, m).view(B, n_free, m, 6, 1) +
        safe_scatter_add_mat(Ejk, jj_s, kk_inv, n_free, m).view(B, n_free, m, 6, 1)
    )
    # Keep (B, m, 1, 1) shape to match ba.py's Schur complement expectations.
    C_mat = safe_scatter_add_vec(C_diag, kk_inv, m)   # (B, m, 1, 1)
    w_mat = safe_scatter_add_vec(w_disp, kk_inv, m)   # (B, m, 1, 1)

    # ------------------------------------------------------------------
    # 8. Schur complement solve (mirrors ba.py::BA)
    # ------------------------------------------------------------------
    from .ba import block_matmul, block_solve

    Q = 1.0 / (C_mat + lm)                  # (B, m, 1, 1)

    EQ = E_mat * Q[:, None]                  # (B, n_free, m, 6, 1) * (B, 1, m, 1, 1) ✓

    if n_free > 0:
        S = B_mat - block_matmul(EQ, E_mat.permute(0, 2, 1, 4, 3))
        y = v_mat - block_matmul(EQ, w_mat.unsqueeze(dim=2))
        dX = block_solve(S, y, ep=ep, lm=lm)  # (B, n_free, 1, 6, 1)

        dZ = Q * (w_mat - block_matmul(
            E_mat.permute(0, 2, 1, 4, 3), dX).squeeze(dim=-1))
        dX = dX.view(B, -1, 6)
    else:
        dX = torch.zeros(B, 0, 6, device=fmaps.device)
        dZ = Q * w_mat

    dZ = dZ.view(B, -1, 1, 1)

    # ------------------------------------------------------------------
    # 9. Retract
    # ------------------------------------------------------------------
    # Update depths (on the unique patch set kx)
    x, y, disps = patches.unbind(dim=2)
    disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches = torch.stack([x, y, disps], dim=2)

    if n_free > 0:
        poses = pose_retr(poses, dX, n_fix + torch.arange(n_free, device=fmaps.device))

    return poses, patches, resids


def feature_metric_gn(
    fmaps:      torch.Tensor,         # (B, N, C, H, W)
    patches:    torch.Tensor,         # (B, M, 3, P, P)
    poses,                            # SE3 (B, N, 7)
    intrinsics: torch.Tensor,         # (B, 4)
    ii:         torch.Tensor,         # (E,)
    jj:         torch.Tensor,         # (E,)
    kk:         torch.Tensor,         # (E,)
    weights:    torch.Tensor,         # (B, E)
    fmaps_grad: torch.Tensor = None,  # (B, N, C, 2, H, W) optional precomputed gradient
    n_iters:    int   = 4,
    lm:         float = 1e-4,
    ep:         float = 1.0,
    fixedp:     int   = 1,
) -> tuple:
    """Run n_iters Gauss-Newton iterations on the feature metric.

    Returns:
        poses:   updated SE3
        patches: updated patches
        all_resids: list of (B, E, C) feature residuals per iteration
    """
    all_resids = []
    for _ in range(n_iters):
        poses = poses.detach()
        patches = patches.detach()
        poses, patches, resids = feature_metric_gn_step(
            fmaps, patches, poses, intrinsics, ii, jj, kk, weights,
            fmaps_grad=fmaps_grad, ep=ep, lm=lm, fixedp=fixedp)
        all_resids.append(resids)

    return poses, patches, all_resids
