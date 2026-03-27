"""FMNet: Feature-Metric Event Odometry Network.

Replaces DEIO's CorrBlock + GRU Update + BA with a direct feature-metric
Gauss-Newton solver.  Architecture:

    Event voxel → BasicEncoder4Evs (fnet) → fmap (128 ch)
    Event voxel → BasicEncoder4Evs (inet) → imap (384 ch)
    Patchifier (Scorer) selects M patches per frame
    Weight MLP (from imap) → per-patch confidence w_k
    feature_metric_gn() → pose + depth update
    Repeat coarse-to-fine for STEPS iterations

Kept from DEIO / DEVO:
    BasicEncoder4Evs, Patchifier, Scorer, SE3 retraction.

Removed:
    CorrBlock, Update (GRU), BA call.

Added:
    weight_head (MLP: 384 → 1), feature_metric_gn().
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dpvo import altcorr
from dpvo import lietorch
from dpvo.lietorch import SE3

from .extractor import BasicEncoder4Evs
from .blocks import GradientClip
from .selector import Scorer, SelectionMethod, PatchSelector
from .utils import coords_grid_with_index, set_depth, flatmeshgrid
from .feature_metric_gn import feature_metric_gn
from . import projective_ops as pops

from utils.voxel_utils import std, rescale, voxel_augment

autocast = torch.cuda.amp.autocast

DIM_INET = 384
DIM_FNET = 128
DIM      = 32


class FMNet(nn.Module):
    """Feature-Metric Odometry Network.

    Args:
        bins:      Number of event voxel bins (default: 5).
        dim_fnet:  Feature map channels (default: 128).
        dim_inet:  Context feature channels (default: 384).
        dim:       Initial encoder dimension (default: 32).
        patch_size: Patch size P (default: 3).
        norm:      Event normalisation method ('rescale', 'std2', 'none').
        randaug:   Apply random voxel augmentation during training.
    """

    def __init__(
        self,
        bins:       int   = 5,
        dim_fnet:   int   = DIM_FNET,
        dim_inet:   int   = DIM_INET,
        dim:        int   = DIM,
        patch_size: int   = 3,
        norm:       str   = 'std2',
        randaug:    bool  = False,
    ):
        super().__init__()
        self.P        = patch_size
        self.dim_fnet = dim_fnet
        self.dim_inet = dim_inet
        self.RES      = 4.0   # feature map is 1/4 of input resolution
        self.norm     = norm
        self.randaug  = randaug

        # Feature extractor: matching features (instance norm for invariance)
        self.fnet = BasicEncoder4Evs(
            bins=bins, output_dim=dim_fnet, dim=dim, norm_fn='instance')

        # Context extractor: weight features (no norm to preserve values)
        self.inet = BasicEncoder4Evs(
            bins=bins, output_dim=dim_inet, dim=dim, norm_fn='none')

        # Patch scorer for informative patch selection
        self.scorer = Scorer(bins)

        # Weight head: context features → per-patch confidence in [0, 1]
        self.weight_head = nn.Sequential(
            nn.Linear(dim_inet, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            GradientClip(),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise(self, images: torch.Tensor) -> torch.Tensor:
        if self.norm == 'none':
            return images
        elif self.norm in ('rescale', 'norm'):
            return rescale(images)
        elif self.norm in ('standard', 'std'):
            return std(images, sequence=False)
        elif self.norm in ('standard2', 'std2'):
            return std(images)
        raise NotImplementedError(f"norm '{self.norm}' not implemented")

    def _select_patches(
        self,
        fmap:              torch.Tensor,  # (B, N, C, H, W)
        images:            torch.Tensor,  # (B, N, bins, H_orig, W_orig)
        disps:             torch.Tensor,  # (B, N, H, W)  at feature-map resolution
        patches_per_image: int,
    ):
        """Select informative patches using learned scorer.

        Returns:
            patches: (B, N*patches_per_image, 3, P, P) — (x, y, disp) grids
            imap_p:  (B, N*patches_per_image, dim_inet) — context at patch centres
            ix:      (N*patches_per_image,) — frame index per patch
        """
        B, N, C, H, W = fmap.shape
        P = self.P
        with torch.amp.autocast('cuda'):
            imap = self.inet(images) / 4.0     # fp16 if AMP active
            scores = self.scorer(images)
        imap   = imap.float()                  # fp32 for weight_head / GN
        scores = torch.sigmoid(scores.float()) # fp32

        if self.training:
            x = torch.randint(0, W - 2, size=[N, 3 * patches_per_image], device=fmap.device)
            y = torch.randint(0, H - 2, size=[N, 3 * patches_per_image], device=fmap.device)
            coords_cand = torch.stack([x, y], dim=-1).float()
            sc = altcorr.patchify(scores[0, :, None], coords_cand, 0).view(N, 3 * patches_per_image)
            _, idx = torch.sort(sc, dim=1)
            x = x + 1;  y = y + 1
            x = torch.gather(x, 1, idx[:, -patches_per_image:])
            y = torch.gather(y, 1, idx[:, -patches_per_image:])
        else:
            sel_fn = PatchSelector('multi', grid=True)
            x, y = sel_fn(scores, patches_per_image)
            x = x + 1;  y = y + 1

        coords = torch.stack([x, y], dim=-1).float()  # (N, M, 2)

        # Extract context at patch centres
        imap_p = altcorr.patchify(imap[0], coords, 0).view(B, -1, self.dim_inet, 1, 1)
        imap_p = imap_p.squeeze(-1).squeeze(-1)        # (B, N*M, dim_inet)

        # Build patch coordinate grids (x, y, depth)
        grid, _ = coords_grid_with_index(disps, device=fmap.device)  # (B, N, 3, H, W)
        patches = altcorr.patchify(grid[0], coords, P // 2).view(B, -1, 3, P, P)

        # Frame index per patch
        ix = torch.arange(N, device=fmap.device).view(N, 1).repeat(1, patches_per_image).reshape(-1)

        return patches, imap_p, ix

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images:            torch.Tensor,  # (B, N, bins, H, W)
        poses,                            # SE3 (B, N, 7) — initial estimate
        disps:             torch.Tensor,  # (B, N, H/4, W/4) — inv-depth at feature res
        intrinsics:        torch.Tensor,  # (B, 4) [fx, fy, cx, cy] at feature-map res
        STEPS:             int   = 12,
        patches_per_image: int   = 80,
        structure_only:    bool  = False,
        gn_iters:          int   = 4,
        lm:                float = 1e-4,
        ep:                float = 10.0,
    ):
        """Feature-metric forward pass.

        Returns:
            traj: list of (poses_pred, poses_gt_ref, resids) tuples, one per GN round.
                  poses_pred is an SE3 object; resids is (B, E, C) feature residuals.
        """
        B, N, bins, H_orig, W_orig = images.shape

        # ------------------------------------------------------------------
        # 1. Normalise event voxels
        # ------------------------------------------------------------------
        images = self._normalise(images)
        if self.training and self.randaug and np.random.rand() < 0.33:
            rescaled = self.norm in ('rescale', 'norm')
            images = voxel_augment(images, rescaled=rescaled)

        # ------------------------------------------------------------------
        # 2. Scale intrinsics to feature-map space (already done by caller,
        #    but we keep self.RES for coord-space alignment).
        # ------------------------------------------------------------------
        intr = intrinsics / self.RES   # feature-map resolution intrinsics

        # Downsample disparity to feature resolution if not already done
        if disps.shape[-1] == W_orig:
            disps = disps[:, :, 1::4, 1::4].float()

        # ------------------------------------------------------------------
        # 3. Extract feature maps  (encoders run under AMP; GN solver stays fp32)
        # ------------------------------------------------------------------
        with torch.amp.autocast('cuda'):
            fmap_amp = self.fnet(images) / 4.0   # fp16 if AMP is active
        fmap = fmap_amp.float()                  # fp32 for GN solver
        _, _, _, h, w = fmap.shape

        # ------------------------------------------------------------------
        # 4. Select patches & predict weights
        # ------------------------------------------------------------------
        Ps_gt = poses                     # ground-truth SE3 (for supervision)
        Gs    = SE3.IdentityLike(poses)   # predicted poses (start from identity)

        if structure_only:
            Gs.data[:] = poses.data[:]

        patches, imap_p, ix = self._select_patches(fmap, images, disps, patches_per_image)
        # patches: (B, M_total, 3, P, P),  imap_p: (B, M_total, dim_inet)

        # Per-patch weights from context features
        weights_patch = self.weight_head(imap_p).squeeze(-1)  # (B, M_total)

        # Randomise initial depths (as in DEIO)
        P = self.P
        d = patches[..., 2, P // 2, P // 2]
        patches = set_depth(patches, torch.rand_like(d))

        # ------------------------------------------------------------------
        # 5. Build initial patch graph
        # Use up to 8 frames for initialisation, capped at the actual N.
        # ------------------------------------------------------------------
        n_init = min(8, N)
        kk, jj = flatmeshgrid(
            torch.where(ix < n_init)[0],
            torch.arange(0, n_init, device=fmap.device),
            indexing='ij')
        ii = ix[kk]

        # Per-edge weights
        def _edge_weights(kk):
            return weights_patch[:, kk]  # (B, E)

        # ------------------------------------------------------------------
        # 6. Iterative refinement
        # ------------------------------------------------------------------
        # Precompute pyramid levels and their spatial gradients once.
        # fmap is constant across all STEPS, so fmap_l and its gradient
        # only need to be computed once per level, not 12× per level.
        from .feature_metric_gn import _fmap_spatial_gradient as _grad
        fmap_levels = {}
        fmap_grad_levels = {}
        for level in [2, 1]:
            fl = F.avg_pool2d(
                fmap.view(B * N, self.dim_fnet, h, w),
                level, stride=level
            ).view(B, N, self.dim_fnet, h // level, w // level)
            fmap_levels[level] = fl
            # (B*N, C, 2, h', w') → (B, N, C, 2, h', w')
            with torch.no_grad():  # gradient of gradient not needed
                gl = _grad(fl.view(B * N, self.dim_fnet, h // level, w // level))
                fmap_grad_levels[level] = gl.view(B, N, self.dim_fnet, 2,
                                                   h // level, w // level)
        intr_levels = {}
        for level in [2, 1]:
            il = intr.clone()
            il[:, :2] = il[:, :2] / level
            il[:, 2:] = il[:, 2:] / level
            intr_levels[level] = il

        traj = []

        for step in range(STEPS):
            Gs     = Gs.detach()
            patches = patches.detach()

            # Incremental frame addition: once initial frames are warmed up,
            # add one new frame per step until all N frames are included.
            n = ii.max().item() + 1
            if step >= n_init and n < N:
                if not structure_only:
                    Gs.data[:, n] = Gs.data[:, n - 1]

                kk1, jj1 = flatmeshgrid(
                    torch.where(ix < n)[0],
                    torch.arange(n, n + 1, device=fmap.device))
                kk2, jj2 = flatmeshgrid(
                    torch.where(ix == n)[0],
                    torch.arange(0, n + 1, device=fmap.device))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                if np.random.rand() < 0.1:
                    keep = (ii != (n - 4)) & (jj != (n - 4))
                    ii, jj, kk = ii[keep], jj[keep], kk[keep]

                patches[:, ix == n, 2] = torch.median(
                    patches[:, (ix == n - 1) | (ix == n - 2), 2])
                n = ii.max().item() + 1

            # Per-edge weights
            w_e = _edge_weights(kk)  # (B, E)

            # ------------------------------------------------------------------
            # Feature-metric GN (coarse → fine) — reuse precomputed fmap_l
            # ------------------------------------------------------------------
            for level in [2, 1]:
                Gs, patches, resids = feature_metric_gn(
                    fmap_levels[level], patches, Gs, intr_levels[level],
                    ii, jj, kk, w_e,
                    fmaps_grad=fmap_grad_levels[level],
                    n_iters=gn_iters // 2, lm=lm, ep=ep, fixedp=1)

            traj.append((Gs[:, :n], Ps_gt[:, :n], resids[-1]))

        return traj
