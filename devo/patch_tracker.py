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

from dpvo import altcorr
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
        self.last_forward_state = None

    def _normalize_images(self, images):
        """Match the training/inference voxel normalization used by forward()."""
        if self.norm == 'none':
            return images
        elif self.norm in ('rescale', 'norm'):
            return rescale(images)
        elif self.norm in ('standard', 'std'):
            return std(images, sequence=False)
        elif self.norm in ('standard2', 'std2'):
            return std(images)
        raise NotImplementedError(f"norm '{self.norm}' not implemented")

    def _extract_dense_features(self, images):
        """Extract dense feature maps at 1/4 resolution for replay tracking."""
        fmap = self.patchify.fnet(images) / 4.0
        imap_dense = self.patchify.inet(images) / 4.0
        return fmap, imap_dense

    def _extract_patch_features(self, fmap, imap_dense, frame_idx, coords):
        """Extract patch matching/context features from dense maps at given coords."""
        if coords.numel() == 0:
            empty_g = fmap.new_empty((fmap.shape[0], 0, self.dim_fnet, self.P, self.P))
            empty_i = fmap.new_empty((fmap.shape[0], 0, self.dim_inet))
            return empty_g, empty_i

        coords = coords.detach().float().view(1, -1, 2)
        gmap = altcorr.patchify(fmap[0, frame_idx:frame_idx + 1], coords, self.P // 2)
        gmap = gmap.view(fmap.shape[0], -1, self.dim_fnet, self.P, self.P)

        imap = altcorr.patchify(imap_dense[0, frame_idx:frame_idx + 1], coords, 0)
        imap = imap.view(fmap.shape[0], -1, self.dim_inet)
        return gmap, imap

    def _valid_center_mask(self, coords, feat_h, feat_w):
        """Check whether patch centers are finite and keep the full PxP patch in-bounds."""
        radius = self.P // 2
        finite = torch.isfinite(coords).all(dim=-1)
        valid_x = (coords[..., 0] >= radius) & (coords[..., 0] <= (feat_w - 1 - radius))
        valid_y = (coords[..., 1] >= radius) & (coords[..., 1] <= (feat_h - 1 - radius))
        return finite & valid_x & valid_y

    @autocast(enabled=False)
    def _track_pair(self, fmap, gmap, imap, target_frame, start_coords, steps):
        """Track a set of patches from their current coords into one target frame."""
        num_patches = int(start_coords.shape[0])
        if num_patches == 0:
            empty = fmap.new_empty((fmap.shape[0], 0, 2))
            empty_conf = fmap.new_empty((fmap.shape[0], 0))
            empty_valid = torch.zeros((fmap.shape[0], 0), device=fmap.device, dtype=torch.bool)
            return {
                "coords": empty,
                "conf": empty_conf,
                "valid": empty_valid,
            }

        corr_fn = CorrBlock(fmap, gmap)
        coords_est = start_coords.detach().float()[None].clone()
        net = torch.zeros(fmap.shape[0], num_patches, self.dim_inet, device=fmap.device, dtype=torch.float32)

        kk = torch.arange(num_patches, device=fmap.device, dtype=torch.long)
        jj_corr = torch.full((num_patches,), int(target_frame), device=fmap.device, dtype=torch.long)
        zero_idx = torch.zeros(num_patches, device=fmap.device, dtype=torch.long)

        r = self.P // 2
        offs = torch.arange(-r, r + 1, device=fmap.device, dtype=torch.float32)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        pxp_offset = torch.stack([ox, oy], dim=0)

        weight_last = torch.ones(fmap.shape[0], num_patches, 2, device=fmap.device, dtype=torch.float32)

        for _ in range(int(steps)):
            coords_est = coords_est.detach()
            coords_pxp = coords_est[:, :, :, None, None] + pxp_offset[None, None]
            corr = corr_fn(kk, jj_corr, coords_pxp)
            net, (delta, weight_last, _) = self.update(
                net, imap, corr, None, zero_idx, zero_idx, kk, tracker_fast_path=True
            )
            coords_est = coords_est + delta

        feat_h, feat_w = fmap.shape[-2:]
        valid = self._valid_center_mask(coords_est, feat_h, feat_w)
        conf = (-0.5 * weight_last.mean(dim=-1)).exp()
        return {
            "coords": coords_est,
            "conf": conf,
            "valid": valid,
        }

    @staticmethod
    def _pairwise_endpoint_spread(endpoints, valid):
        """Mean pairwise endpoint distance across valid replay runs for each patch."""
        if endpoints.numel() == 0:
            return endpoints.new_empty((0,))

        runs, num_patches, _ = endpoints.shape
        spread = endpoints.new_full((num_patches,), float("inf"))
        if runs < 2:
            return endpoints.new_zeros((num_patches,))

        for patch_idx in range(num_patches):
            keep = valid[:, patch_idx]
            if int(keep.sum().item()) < 2:
                continue
            pts = endpoints[keep, patch_idx]
            dmat = torch.cdist(pts, pts)
            triu = torch.triu_indices(dmat.shape[0], dmat.shape[1], offset=1, device=dmat.device)
            spread[patch_idx] = dmat[triu[0], triu[1]].mean()
        return spread

    @autocast(enabled=False)
    def compute_replay_metrics(self, images, *, horizon=2, replay_steps=4, replay_runs=3):
        """Compute replay-based patch reliability metrics from the last forward selection."""
        state = self.last_forward_state
        if state is None:
            raise RuntimeError("compute_replay_metrics() requires a preceding forward() call")

        fmap_clean = state["fmap"]
        imap_dense_clean = state["imap_dense"]
        patches = state["patches"]
        ix = state["ix"]
        clean_images = state["images"]

        if patches is None or ix is None:
            raise RuntimeError("No selected patches available for replay metrics")

        b, n_frames = clean_images.shape[:2]
        if b != 1:
            raise RuntimeError("Replay metrics currently require batch size 1")

        num_patches = patches.shape[1]
        device = clean_images.device
        feat_h, feat_w = fmap_clean.shape[-2:]

        fb_error_px = torch.full((num_patches,), float("nan"), device=device, dtype=torch.float32)
        stability_error_px = torch.full((num_patches,), float("nan"), device=device, dtype=torch.float32)
        replay_valid = torch.zeros((num_patches,), device=device, dtype=torch.bool)
        forward_endpoint = torch.full((num_patches, 2), float("nan"), device=device, dtype=torch.float32)
        forward_conf = torch.full((num_patches,), float("nan"), device=device, dtype=torch.float32)

        centers = patches[0, :, :2, self.P // 2, self.P // 2].detach().float()
        replay_runs = max(int(replay_runs), 1)

        perturbed_states = []
        for _ in range(max(0, replay_runs - 1)):
            perturbed = clean_images.detach().clone()
            if hasattr(self.patchify, "scorer") and perturbed.shape[2] > 0:
                keep_prob = 0.85
                mask = (torch.rand_like(perturbed) < keep_prob).to(perturbed.dtype)
                perturbed = perturbed * mask / keep_prob
                shift = int(torch.randint(-1, 2, size=(1,), device=device).item())
                if shift != 0:
                    perturbed = torch.roll(perturbed, shifts=shift, dims=2)
                gain = 0.9 + 0.2 * torch.rand((1, n_frames, 1, 1, 1), device=device, dtype=perturbed.dtype)
                perturbed = perturbed * gain
            fmap_p, imap_p = self._extract_dense_features(perturbed)
            perturbed_states.append((fmap_p.detach(), imap_p.detach()))

        for src_frame in range(n_frames):
            target_frame = src_frame + int(horizon)
            patch_mask = (ix == src_frame)
            if target_frame >= n_frames or not patch_mask.any():
                continue

            patch_idx = torch.where(patch_mask)[0]
            coords_src = centers[patch_idx]
            gmap_src = state["gmap"][:, patch_idx]
            imap_src = state["imap"][:, patch_idx]

            forward = self._track_pair(
                fmap_clean, gmap_src, imap_src, target_frame, coords_src, replay_steps
            )

            forward_endpoint[patch_idx] = forward["coords"][0]
            forward_conf[patch_idx] = forward["conf"][0]
            forward_valid = forward["valid"][0]

            if forward_valid.any():
                valid_idx = torch.where(forward_valid)[0]
                coords_target = forward["coords"][0, valid_idx]
                gmap_back, imap_back = self._extract_patch_features(
                    fmap_clean, imap_dense_clean, target_frame, coords_target
                )
                backward = self._track_pair(
                    fmap_clean, gmap_back, imap_back, src_frame, coords_target, replay_steps
                )
                backward_valid = backward["valid"][0]
                good_idx = valid_idx[backward_valid]
                replay_valid[patch_idx[good_idx]] = True
                fb_error_px[patch_idx[good_idx]] = (
                    backward["coords"][0, backward_valid] - coords_src[good_idx]
                ).norm(dim=-1)

            endpoints = [forward["coords"][0]]
            endpoints_valid = [forward_valid]
            for fmap_p, imap_p in perturbed_states:
                gmap_p, imap_src_p = self._extract_patch_features(
                    fmap_p, imap_p, src_frame, coords_src
                )
                forward_p = self._track_pair(
                    fmap_p, gmap_p, imap_src_p, target_frame, coords_src, replay_steps
                )
                endpoints.append(forward_p["coords"][0])
                endpoints_valid.append(forward_p["valid"][0])

            spread = self._pairwise_endpoint_spread(
                torch.stack(endpoints, dim=0),
                torch.stack(endpoints_valid, dim=0),
            )
            stability_error_px[patch_idx] = spread

        replay_valid = replay_valid & torch.isfinite(fb_error_px) & self._valid_center_mask(
            forward_endpoint, feat_h, feat_w
        )
        return {
            "fb_error_px": fb_error_px,
            "stability_error_px": stability_error_px,
            "replay_valid": replay_valid,
            "forward_endpoint": forward_endpoint,
            "forward_conf": forward_conf,
        }

    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, STEPS=12, patches_per_image=80,
                scorer_eval_mode="multi", scorer_eval_use_grid=True):
        """
        Forward pass: RAFT-style iterative patch tracking.

        Args:
            images:     Event voxels  (B, N_frames, bins, H, W)
            poses:      GT w2c poses  (B, N_frames) as SE3  [training only]
            disps:      GT inv-depth  (B, N_frames, H, W)   [training only]
            intrinsics: Camera K      (B, N_frames, 4)  [fx,fy,cx,cy] at full res
            STEPS:      Update iterations (default 12)
            patches_per_image: Patches selected per frame (default 80)
            scorer_eval_mode: Patch selector used at eval time for scorer mode
            scorer_eval_use_grid: Whether scorer eval selection uses the grid path

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
        images = self._normalize_images(images)

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
        result = self.patchify(images,
                               patches_per_image=patches_per_image,
                               disps=disps,
                               scorer_eval_mode=scorer_eval_mode,
                               scorer_eval_use_grid=scorer_eval_use_grid)
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
        dense_cache = getattr(self.patchify, "last_dense_features", None) or {}
        self.last_forward_state = {
            "images": images.detach(),
            "fmap": dense_cache.get("fmap", fmap.detach()),
            "imap_dense": dense_cache.get("imap_dense"),
            "gmap": gmap.detach(),
            "imap": imap.detach(),
            "patches": patches.detach(),
            "ix": ix.detach(),
            "scores": scores.detach() if isinstance(scores, torch.Tensor) else None,
        }

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
