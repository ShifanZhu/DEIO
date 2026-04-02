"""
PatchTracker: Standalone Event Patch Tracking Network

Reuses Patchifier, CorrBlock, and Update from eVONet (devo/enet.py) unchanged.
Replaces the BA-based iterative pose refinement with RAFT-style direct flow
accumulation: at each step, coords_est += delta (no BA, no pose estimation).

GT poses and depths are used ONLY to compute supervision targets (coords_gt)
during training — the tracker itself never maintains or estimates poses.
"""

import numpy as np
import torch
import torch.nn as nn

from dpvo.lietorch import SE3

from .enet import Update, Patchifier, CorrBlock
from .selector import SelectionMethod
from .utils import flatmeshgrid, coords_grid_with_index
from . import projective_ops as pops

from utils.voxel_utils import std, rescale, voxel_augment

autocast = lambda **kwargs: torch.amp.autocast('cuda', **kwargs)

DIM = 384


class PatchTracker(nn.Module):
    """
    Standalone patch tracker built from DEIO components.

    Architecture:
        - Patchifier (fnet/inet/scorer): feature extraction + patch selection
        - CorrBlock: multi-scale correlation (±12px search window)
        - Update (GRU): recurrent flow prediction with message passing

    Replaces eVONet's BA loop with RAFT-style accumulation:
        coords_est[t+1] = coords_est[t] + delta[t]

    GT poses are used only in training to compute the reprojection supervision
    target (coords_gt). At inference, no poses are needed or produced.

    Args:
        args: config namespace (only args.resnet is used)
        P: patch size (default 3)
        dim_inet: GRU hidden state dimension (default 384)
        dim_fnet: matching feature dimension (default 128)
        dim: first encoder channel dimension (default 32)
        patch_selector: 'scorer' | 'gradient' | 'random'
        norm: voxel normalization method ('std2' | 'std' | 'rescale' | 'none')
        randaug: random voxel augmentation during training
    """

    def __init__(self, args, P=3, dim_inet=DIM, dim_fnet=128, dim=32,
                 patch_selector=SelectionMethod.SCORER, norm="std2", randaug=False,
                 corner_guidance='none'):
        super().__init__()
        self.P = P
        self.dim_inet = dim_inet
        self.dim_fnet = dim_fnet
        self.patch_selector = patch_selector if isinstance(patch_selector, str) else patch_selector

        self.patchify = Patchifier(args, patch_size=P, dim_inet=dim_inet,
                                   dim_fnet=dim_fnet, dim=dim,
                                   patch_selector=patch_selector,
                                   corner_guidance=corner_guidance)
        self.update = Update(P, dim_inet)

        self.RES = 4.0
        self.norm = norm
        self.randaug = randaug

    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, STEPS=12, patches_per_image=80):
        """
        Forward pass: RAFT-style iterative patch tracking.

        Args:
            images:     Event voxels  (B, N_frames, bins, H, W)
            poses:      GT w2c poses  (B, N_frames) as SE3  [training only]
            disps:      GT inv-depth  (B, N_frames, H, W)   [training only]
            intrinsics: Camera K      (B, N_frames, 4)  [fx,fy,cx,cy] at full res
            STEPS:      Update iterations (default 12)
            patches_per_image: Patches selected per frame (default 80)

        Returns:
            traj: list of STEPS tuples, one per iteration:
                  (coords_est_k, coords_gt_k, valid_k, weight_k, scores_or_None)
                  - coords_est_k: (B, close_edges, 2)  predicted positions
                  - coords_gt_k:  (B, close_edges, 2)  GT reprojected positions
                  - valid_k:      (B, close_edges)      depth-validity mask (float)
                  - weight_k:     (B, close_edges, 2)   GRU confidence weights
                  - scores:       (N_patches,) scorer output, or None
        """
        b, n_total, v, h, w = images.shape

        # ── Step 1: normalize event voxels ──────────────────────────────────
        if self.norm == 'none':
            pass
        elif self.norm in ('rescale', 'norm'):
            images = rescale(images)
        elif self.norm in ('standard', 'std'):
            images = std(images, sequence=False)
        elif self.norm in ('standard2', 'std2'):
            images = std(images)
        else:
            raise NotImplementedError(f"norm '{self.norm}' not implemented")

        if self.training and self.randaug:
            if np.random.rand() < 0.33:
                rescaled = self.norm in ('rescale', 'norm')
                images = voxel_augment(images, rescaled=rescaled)

        # scale intrinsics to feature-map resolution (1/4)
        intrinsics = intrinsics / self.RES

        # downsample disps to feature-map resolution
        if disps is not None:
            disps = disps[:, :, 1::4, 1::4].float()

        # ── Step 2: extract patches and features ────────────────────────────
        result = self.patchify(images, patches_per_image=patches_per_image, disps=disps)
        if len(result) == 6:
            fmap, gmap, imap, patches, ix, scores = result
        else:
            fmap, gmap, imap, patches, ix = result
            scores = None

        # fmap: (B, N_frames, dim_fnet, H/4, W/4)
        # gmap: (B, N_patches, dim_fnet, P, P)
        # imap: (B, N_patches, dim_inet, 1, 1)
        # patches: (B, N_patches, 3, P, P)  — [x, y, depth] at feature res
        # ix: (N_patches,) frame index for each patch

        b, N, c, h_f, w_f = fmap.shape
        p = self.P

        # keep GT patches (with GT depths from disps) for reprojection supervision
        patches_gt = patches.clone()
        Ps = poses  # GT w2c SE3 poses

        corr_fn = CorrBlock(fmap, gmap)

        imap = imap.view(b, -1, self.dim_inet)  # (B, N_patches, dim_inet)

        # ── Step 3: build initial co-visibility graph (first 8 frames) ──────
        kk, jj = flatmeshgrid(
            torch.where(ix < 8)[0],
            torch.arange(0, 8, device="cuda"),
            indexing="ij"
        )
        ii = ix[kk]

        # ── Step 4: initialize ───────────────────────────────────────────────
        # Zero-flow init: assume patches don't move (will be refined by GRU)
        # coords_est is in feature-map pixel coordinates of the TARGET frame
        coords_est = patches_gt[:, kk, :2, p // 2, p // 2].clone()  # (B, edges, 2)

        net = torch.zeros(b, len(kk), self.dim_inet, device="cuda", dtype=torch.float)

        # offset grid for expanding center → PxP (for correlation lookup)
        r = p // 2
        offs = torch.arange(-r, r + 1, device="cuda", dtype=torch.float)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")   # (P, P)
        pxp_offset = torch.stack([ox, oy], dim=0)            # (2, P, P)

        # ── Step 5: iterative tracking loop ─────────────────────────────────
        traj = []

        while len(traj) < STEPS:
            coords_est = coords_est.detach()

            # incremental frame addition (matches eVONet logic, minus BA depth init)
            n = ii.max().item() + 1
            if len(traj) >= 8 and n < N:
                # edges: existing patches → new frame
                kk1, jj1 = flatmeshgrid(
                    torch.where(ix < n)[0],
                    torch.arange(n, n + 1, device="cuda"))
                # edges: new patches → all existing frames
                kk2, jj2 = flatmeshgrid(
                    torch.where(ix == n)[0],
                    torch.arange(0, n + 1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), self.dim_inet, device="cuda")
                net = torch.cat([net1, net], dim=1)

                # initialize new-edge coords_est from source patch centers
                new_c1 = patches_gt[:, kk1, :2, p // 2, p // 2]
                new_c2 = patches_gt[:, kk2, :2, p // 2, p // 2]
                coords_est = torch.cat([new_c1, new_c2, coords_est], dim=1)

                # randomly prune old edges (10 % chance) to cap graph size
                if np.random.rand() < 0.1:
                    keep = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[keep]
                    jj = jj[keep]
                    kk = kk[keep]
                    net = net[:, keep]
                    coords_est = coords_est[:, keep]

                n = ii.max().item() + 1

            # expand center → PxP grid for CorrBlock  (B, edges, 2, P, P)
            coords_pxp = coords_est[:, :, :, None, None] + pxp_offset[None, None]

            corr = corr_fn(kk, jj, coords_pxp)

            net, (delta, weight, _) = self.update(
                net, imap[:, kk], corr, None, ii, jj, kk)

            # RAFT-style: accumulate delta directly (no BA!)
            coords_est = coords_est + delta

            # select close edges (temporal neighbors ≤ 2 frames) for flow supervision
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            # compute GT reprojection for close edges
            coords_gt, valid = pops.transform(
                Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], valid=True)
            # coords_gt: (B, close_edges, P, P, 2) → take center pixel
            coords_gt_center = coords_gt[..., p // 2, p // 2, :]  # (B, close_edges, 2)

            traj.append((
                coords_est[:, k],    # (B, close_edges, 2)
                coords_gt_center,    # (B, close_edges, 2)
                valid,               # (B, close_edges) float validity mask
                weight[:, k],        # (B, close_edges, 2)
                scores,              # (N_frames, patches_per_image) or None
                kk[k],               # (close_edges,) patch indices into scores.view(-1)
            ))

        return traj
