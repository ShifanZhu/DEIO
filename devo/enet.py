"""
eVONet: Event-based Visual Odometry Network

This module implements the core neural network architecture for DEIO (Deep Event Inertial Odometry).
The network performs event-based recurrent optical flow estimation using a patch-based approach.

Paper Reference: "DEIO: Deep Event Inertial Odometry" (ICCV 2025)
Architecture Components:
    1. Patchifier: Extracts and selects informative event patches
    2. CorrBlock: Computes multi-scale visual similarity between patches
    3. Update: Recurrent update operator for optical flow prediction
    4. eVONet: Main network combining all components with differentiable BA

Key Innovation: Tight integration of learned event-based optical flow with
                graph-based optimization (Section III of paper)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum
from torchvision.ops import batched_nms

# DPVO (baseline) modules for fast BA and correlation operations
from dpvo import fastba
from dpvo import altcorr
from dpvo import lietorch
from dpvo.lietorch import SE3

# Feature extractors for event voxel grids
from .extractor import BasicEncoder, BasicEncoder4Evs
from .res_net_extractor import ResNetFPN
from .blocks import GradientClip, GatedResidual, SoftAgg
from .selector import Scorer, SelectionMethod, PatchSelector

from .utils import *
from .ba import BA  # Differentiable Bundle Adjustment
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

# Event voxel processing utilities
from utils.voxel_utils import std, rescale, voxel_augment
from utils.viz_utils import visualize_voxel, visualize_N_voxels, visualize_scorer_map

# Default hidden state dimension (Paper Section III-A)
DIM = 384  # Context feature dimension for update operator

class Update(nn.Module):
    """
    Recurrent Update Operator for Event-based Optical Flow

    This module implements the edge state updater described in Paper Section III-A.
    It predicts optical flow corrections (δ_inj) and confidence weights (Σ_inj) using:
        - Correlation features from multi-scale matching
        - Context features from event patches
        - Message passing from neighboring edges in the patch graph
        - GRU-style recurrent updates for temporal consistency

    Architecture corresponds to Paper Equation (2):
        Minimize ||π[T_j^(-1)·T_i·π^(-1)(P̂_in)] - [P̂_in + δ_inj]||²_Σ_inj

    Args:
        p: Patch size (default: 3x3)
        dim: Hidden state dimension (default: 384)
    """
    def __init__(self, p, dim=DIM):
        super(Update, self).__init__()
        self.dim = dim

        # MLPs for aggregating information from neighboring edges in the patch graph
        # These implement message passing between connected patches
        self.c1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

        self.c2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

        self.norm = nn.LayerNorm(dim, eps=1e-3)

        # Soft aggregation modules for pooling features
        self.agg_kk = SoftAgg(dim)  # Aggregate over patches (same patch, different frames)
        self.agg_ij = SoftAgg(dim)  # Aggregate over frame pairs (i, j)

        # GRU-style recurrent unit for maintaining temporal consistency
        # This is the "recurrent" part of the recurrent optical flow network
        self.gru = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-3),
            GatedResidual(dim),
            nn.LayerNorm(dim, eps=1e-3),
            GatedResidual(dim),
        )

        # Process correlation features from CorrBlock
        # Input size: 2 pyramid levels * 49 (7x7 search radius) * p*p (patch size)
        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        # Predict optical flow delta: δ_inj ∈ R^2 (Paper Eq. 2)
        # This is the 2D correction to the reprojected patch center
        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            GradientClip())

        # Predict confidence weights: Σ_inj ∈ R^2 (Paper Eq. 2)
        # These weights indicate uncertainty in the flow prediction
        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """
        Recurrent update step for optical flow prediction

        Args:
            net: Hidden state from previous iteration (B, num_edges, dim)
            inp: Context features from patches (B, num_edges, dim)
            corr: Correlation features from CorrBlock (B, num_edges, 2*49*p*p)
            flow: Not used (kept for compatibility)
            ii: Source frame indices for edges (num_edges,)
            jj: Target frame indices for edges (num_edges,)
            kk: Patch indices for edges (num_edges,)

        Returns:
            net: Updated hidden state (B, num_edges, dim)
            (delta, weights, None):
                - delta: Optical flow correction δ_inj (B, num_edges, 2)
                - weights: Confidence Σ_inj (B, num_edges, 2)
        """
        # Combine hidden state, context features, and correlation features
        net = net + inp + self.corr(corr)
        net = self.norm(net)  # (B, num_edges, 384)

        # Message passing: aggregate information from neighboring edges
        # This implements graph convolution on the patch co-visibility graph
        ix, jx = fastba.neighbors(kk, jj)  # Find neighboring edges
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        # Aggregate from neighbors
        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        # Soft aggregation over patches and frame pairs
        net = net + self.agg_kk(net, kk)  # Pool features from same patch
        net = net + self.agg_ij(net, ii*12345 + jj)  # Pool features from same frame pair

        # Recurrent update (GRU-style)
        net = self.gru(net)

        # Predict confidence weights
        weights = self.w(net)

        # Return updated state and predictions
        return net, (self.d(net), weights, None)


class Patchifier(nn.Module):
    """
    Event Patch Extraction and Selection Module

    This module implements the CNN-based feature extraction and patch selection
    described in Paper Section III-A. It extracts two types of features:
        1. Matching features (fnet): For computing visual similarity (correlation)
        2. Context features (inet): For the recurrent update operator

    The module also selects informative patches using one of three strategies:
        - SCORER: Learned patch scoring network (default, Paper Section III-B)
        - GRADIENT: Event gradient-based selection
        - RANDOM: Random sampling baseline

    Args:
        args: Configuration arguments
        patch_size: Size of patches (default: 3x3)
        dim_inet: Context feature dimension (default: 384)
        dim_fnet: Matching feature dimension (default: 128)
        dim: Initial feature dimension in encoder (default: 32)
        patch_selector: Method for selecting patches
    """
    def __init__(self, args, patch_size=3, dim_inet=DIM, dim_fnet=128, dim=32, patch_selector=SelectionMethod.SCORER):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.dim_inet = dim_inet  # Dimension for context features (update operator)
        self.dim_fnet = dim_fnet  # Dimension for matching features (correlation)
        self.patch_selector = patch_selector.lower()

        # Feature extractors for event voxel grids
        # Two separate encoders extract different feature representations
        if args.resnet:
            # ResNet FPN backbone (alternative architecture)
            self.fnet = ResNetFPN(args, input_dim=5, output_dim=self.dim_fnet,
                                 norm_layer=nn.BatchNorm2d, init_weight=True)
            self.inet = ResNetFPN(args, input_dim=5, output_dim=self.dim_inet,
                                 norm_layer=nn.BatchNorm2d, init_weight=True)
        else:
            # BasicEncoder (default from DEVO baseline)
            # fnet uses instance norm for better matching invariance
            # inet uses no norm to preserve absolute feature values for context
            self.fnet = BasicEncoder4Evs(output_dim=self.dim_fnet, dim=dim, norm_fn='instance')
            self.inet = BasicEncoder4Evs(output_dim=self.dim_inet, dim=dim, norm_fn='none')

        # Scorer network for learned patch selection (Paper Section III-B)
        # Predicts importance scores for each spatial location
        if self.patch_selector == SelectionMethod.SCORER:
            self.scorer = Scorer(5)  # Input: 5-channel event voxel grid

    def __event_gradient(self, images):
        """
        Compute gradient magnitude of event voxel grids for patch selection

        Args:
            images: Event voxel grids (B, N, bins, H, W)

        Returns:
            g: Gradient magnitude map (B, N, H/4, W/4)
        """
        images = images.sum(dim=2)  # Sum across voxel bins
        # Compute spatial gradients
        dx = images[...,:-1,1:] - images[...,:-1,:-1]
        dy = images[...,1:,:-1] - images[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)  # Downsample to match feature map resolution
        return g

    def forward(self, images, patches_per_image=80, disps=None, return_color=False, scorer_eval_mode="multi", scorer_eval_use_grid=True):
        """
        Extract and select event patches from voxel grids

        This implements the patch extraction pipeline from Paper Section III-A:
            1. Extract dense features using fnet and inet
            2. Select N patches per frame (N=80 by default, paper uses 96)
            3. Extract patch-based features at selected locations

        Args:
            images: Event voxel grids (B, N_frames, voxel_bins, H, W)
            patches_per_image: Number of patches to select per frame (default: 80)
            disps: Depth maps (optional, for initialization)
            return_color: Whether to return color visualization
            scorer_eval_mode: Evaluation mode for scorer ("multi", "single", etc.)
            scorer_eval_use_grid: Whether to use grid sampling in evaluation

        Returns:
            fmap: Dense matching features (B, N_frames, dim_fnet, H/4, W/4)
            gmap: Patch matching features (B, N_patches, dim_fnet, P, P)
            imap: Patch context features (B, N_patches, dim_inet, 1, 1)
            patches: Patch coordinates and depths (B, N_patches, 3, P, P)
            ix: Frame index for each patch (N_patches,)
            scores (optional): Patch selection scores
        """
        # Extract dense feature maps at 1/4 resolution
        fmap = self.fnet(images) / 4.0  # (B, N_frames, 128, H/4, W/4)
        imap = self.inet(images) / 4.0  # (B, N_frames, 384, H/4, W/4)

        b, n, c, h, w = fmap.shape  # n = number of frames
        P = self.patch_size

        # ==== PATCH SELECTION STRATEGIES ====
        # Select informative patches using one of three methods:
        if self.patch_selector == SelectionMethod.GRADIENT:
            # GRADIENT: Bias towards regions with high event gradient (hand-crafted)
            g = self.__event_gradient(images)  # (B, N_frames, H/4-1, W/4-1)

            if self.training:
                # Training: Random sampling with 3x oversampling
                patch_selector_fn = PatchSelector("3xrandom")
            else:
                # Evaluation: Grid or multi-scale sampling
                patch_selector_fn = PatchSelector(scorer_eval_mode, grid=scorer_eval_use_grid)

            x, y = patch_selector_fn(g, patches_per_image)

            # Clamp to valid range (avoid boundary)
            x = x.clamp(min=1, max=w-2)
            y = y.clamp(min=1, max=h-2)

        elif self.patch_selector == SelectionMethod.RANDOM:
            # RANDOM: Uniform random sampling (baseline)
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")

        elif self.patch_selector == SelectionMethod.SCORER:
            # SCORER: Learned patch importance network (Paper Section III-B, L_score)
            # This is the default and best-performing method
            scores = self.scorer(images)  # (B, N_frames, H/4, W/4)
            scores = torch.sigmoid(scores)  # Normalize to [0, 1]

            if self.training:
                # Training: Sample 3x patches, then select top-scoring ones
                # This provides supervision signal for the scorer network
                x = torch.randint(0, w-2, size=[n, 3*patches_per_image], device="cuda")
                y = torch.randint(0, h-2, size=[n, 3*patches_per_image], device="cuda")

                coords = torch.stack([x, y], dim=-1).float()  # (N_frames, 3*patches_per_image, 2)
                # Extract scores at sampled locations
                scores = altcorr.patchify(scores[0,:,None], coords, 0).view(n, 3 * patches_per_image)

                # Sort by score and keep top patches_per_image
                vx, ix = torch.sort(scores, dim=1)
                x = x + 1  # Offset for boundary
                y = y + 1
                x = torch.gather(x, 1, ix[:, -patches_per_image:])  # Top scoring patches
                y = torch.gather(y, 1, ix[:, -patches_per_image:])
                scores = vx[:, -patches_per_image:].contiguous().view(n, patches_per_image)

            else:
                # Evaluation: Use patch selector with learned scores
                patch_selector_fn = PatchSelector(scorer_eval_mode, grid=scorer_eval_use_grid)
                x, y = patch_selector_fn(scores, patches_per_image)
                coords = torch.stack([x, y], dim=-1).float()
                # Extract score values at selected locations
                scores = altcorr.patchify(scores[0,:,None], coords, 0).view(n, patches_per_image)

                x += 1  # Offset for boundary
                y += 1

        else:
            print(f"{self.patch_selector} not implemented")
            raise NotImplementedError

        # ==== EXTRACT PATCH FEATURES ====
        # Extract features at selected (x, y) locations
        coords = torch.stack([x, y], dim=-1).float()  # (N_frames, patches_per_image, 2)

        # Extract context features at patch centers
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, self.dim_inet, 1, 1)
        # Shape: [B, N_frames*patches_per_image, dim_inet, 1, 1]

        # Extract matching features in PxP neighborhoods
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, self.dim_fnet, P, P)
        # Shape: [B, N_frames*patches_per_image, dim_fnet, P, P]

        # Optional: Extract color for visualization
        if return_color:
            clr = altcorr.patchify(images[0].abs().sum(dim=1,keepdim=True), 4*(coords + 0.5), 0).clamp(min=0,max=255).view(b, -1, 1)

        # ==== CREATE PATCH COORDINATES ====
        # Each patch contains (x, y, depth) for PxP grid
        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        # Create coordinate grid with depth
        grid, _ = coords_grid_with_index(disps, device=fmap.device)  # [B, N_frames, 3, H/4, W/4]
        # Extract PxP patches at selected locations
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)
        # Shape: [B, N_frames*patches_per_image, 3, P, P]
        # where patches[:, :, 0] = x-coords, patches[:, :, 1] = y-coords, patches[:, :, 2] = depths

        # Create frame index for each patch (Paper Equation 1: P_in corresponds to frame i)
        index = torch.arange(n, device="cuda").view(n, 1)  # [N_frames, 1]
        index = index.repeat(1, patches_per_image).reshape(-1)  # [N_frames * patches_per_image]
        # E.g., [0,0,...,0, 1,1,...,1, ..., 14,14,...,14] where each value repeats patches_per_image times

        # Return appropriate outputs based on mode
        if self.training:
            if self.patch_selector == SelectionMethod.SCORER:
                return fmap, gmap, imap, patches, index, scores  # Include scores for L_score loss
        else:
            if return_color:
                return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    """
    Multi-scale Correlation Volume for Patch Matching

    This class implements the correlation layer described in Paper Section III-A:
    "The correlation layer computes visual similarity between event patches"

    It builds a multi-scale pyramid and computes correlation between patches
    at different scales to handle various motion magnitudes robustly.

    The correlation volume measures how well patch features from frame i match
    with features in a local search window around the reprojected location in frame j.

    Args:
        fmap: Dense matching features (B, N_frames, dim_fnet, H/4, W/4)
        gmap: Patch matching features (B, N_patches, dim_fnet, P, P)
        radius: Search radius for correlation (default: 3, giving 7x7 window)
        dropout: Dropout probability for correlation features
        levels: Pyramid levels for multi-scale matching (default: [1, 4])
    """
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius  # Local search radius (7x7 window with radius=3)
        self.levels = levels  # Multi-scale pyramid levels

        self.gmap = gmap  # Patch features to match
        self.pyramid = pyramidify(fmap, lvls=levels)  # Multi-scale feature pyramid

    def __call__(self, ii, jj, coords):
        """
        Compute multi-scale correlation features for patch matching

        For each edge (patch i → frame j), computes correlation between:
            - Patch features from frame i (gmap[kk])
            - Features in a search window around coords in frame j (pyramid[j])

        Args:
            ii: Source frame indices (num_edges,)
            jj: Target frame indices (num_edges,)
            coords: Predicted patch locations in target frames (B, num_edges, 2, P, P)

        Returns:
            corrs: Concatenated correlation features from all pyramid levels
                   Shape: (1, num_edges, num_levels * 49 * P * P)
                   where 49 = (2*radius+1)^2 is the search window size
        """
        corrs = []
        for i in range(len(self.levels)):
            # Compute correlation at each pyramid level
            # coords are scaled by pyramid level to maintain receptive field
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i],
                                   ii, jj, self.radius, self.dropout) ]
        # Concatenate all pyramid levels
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class eVONet(nn.Module):
    """
    Event-based Visual Odometry Network (Main Architecture)

    This is the complete eVONet architecture described in Paper Section III-A and Fig. 1.
    It combines:
        1. Event patch extraction and selection (Patchifier)
        2. Multi-scale correlation computation (CorrBlock)
        3. Recurrent optical flow prediction (Update)
        4. Differentiable bundle adjustment (BA)

    The network estimates camera poses and patch depths by minimizing reprojection
    error (Paper Equation 2) through iterative refinement.

    Key Features:
        - Patch-based representation (N=80 patches per frame, paper uses 96)
        - Recurrent architecture for temporal consistency
        - Differentiable BA for pose/depth optimization
        - Multi-scale correlation for robust matching

    This is the LEARNING component of DEIO. It's later integrated with IMU
    pre-integration in dba.py (Paper Section III-D).

    Args:
        args: Configuration arguments
        P: Patch size (default: 3x3)
        use_viewer: Enable visualization (not used)
        dim_inet: Context feature dimension (default: 384)
        dim_fnet: Matching feature dimension (default: 128)
        dim: Initial encoder dimension (default: 32)
        patch_selector: Patch selection method (default: SCORER)
        norm: Event normalization method (default: "std2")
        randaug: Random augmentation during training
    """
    def __init__(self, args, P=3, use_viewer=False, dim_inet=DIM, dim_fnet=128, dim=32, patch_selector=SelectionMethod.SCORER, norm="std2", randaug=False):
        super(eVONet, self).__init__()
        self.P = P  # Patch size (3x3)
        self.dim_inet = dim_inet  # Context feature dimension (384)
        self.dim_fnet = dim_fnet  # Matching feature dimension (128)
        self.patch_selector = patch_selector

        # Patch extraction and feature encoding module
        self.patchify = Patchifier(args, patch_size=self.P, dim_inet=self.dim_inet,
                                  dim_fnet=self.dim_fnet, dim=dim,
                                  patch_selector=patch_selector)

        # Recurrent update operator for optical flow prediction
        self.update = Update(self.P, self.dim_inet)

        self.dim = dim  # Initial encoder dimension
        self.RES = 4.0  # Resolution factor (features are 1/4 of input resolution)
        self.norm = norm  # Event voxel normalization method
        self.randaug = randaug  # Random augmentation flag


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, plot_patches=False, patches_per_image=80):
        """
        Forward pass: Event-based visual odometry with differentiable BA

        This implements the complete pipeline from Paper Fig. 1 (Learning Phase):
            1. Normalize event voxels
            2. Extract patches and features (Patchifier)
            3. Build patch graph (co-visibility graph)
            4. Iteratively refine poses and depths:
                a. Project patches using current poses
                b. Compute correlation features
                c. Predict optical flow corrections (Update)
                d. Run differentiable BA to update poses/depths
            5. Return trajectory with supervision signals

        Training Supervision (Paper Section III-B):
            - L_pose: Pose error between estimated and ground truth
            - L_flow: Optical flow error
            - L_score: Patch selection score loss

        Args:
            images: Event voxel grids (B, N_frames, voxel_bins, H, W)
            poses: Ground truth poses for supervision (B, N_frames)
            disps: Depth maps (B, N_frames, H, W)
            intrinsics: Camera intrinsics (B, N_frames, 4) [fx, fy, cx, cy]
            M: Not used (kept for compatibility)
            STEPS: Number of update iterations (default: 12)
            P: Not used
            structure_only: If True, fix poses and only optimize depths
            plot_patches: If True, save patch visualization data
            patches_per_image: Number of patches per frame (default: 80)

        Returns:
            traj: List of tuples containing supervision signals for each iteration:
                  (valid, coords, coords_gt, poses_pred, poses_gt, kl, scores, ...)
                  Used to compute L_pose, L_flow, L_score
        """

        # ==== INPUT SHAPES ====
        # images: (B, N_frames, voxel_bins, H, W)
        # poses: (B, N_frames) - SE3 poses for supervision
        # disps: (B, N_frames, H, W) - depth maps

        b, n, v, h, w = images.shape

        # ==== STEP 1: NORMALIZE EVENT VOXEL GRIDS ====
        # Critical preprocessing step for stable training
        if self.norm == 'none':
            pass
        elif self.norm == 'rescale' or self.norm == 'norm':
            # Normalize (rescaling) neg events into [-1,0) and pos events into (0,1] sequence-wise (by default)
            images = rescale(images)
        elif self.norm == 'standard' or self.norm == 'std':
            # Data standardization of events (voxel-wise)
            images = std(images, sequence=False)
        elif self.norm == 'standard2' or self.norm == 'std2':
            # Data standardization of events (sequence-wise by default)
            images = std(images)
        else:
            print(f"{self.norm} not implemented")
            raise NotImplementedError

        if self.training and self.randaug:
            if np.random.rand() < 0.33:
                if self.norm == 'rescale' or self.norm == 'norm':
                    images = voxel_augment(images, rescaled=True)
                elif 'std' in self.norm:
                    images = voxel_augment(images, rescaled=False)
                else:
                    print(f"{self.norm} not implemented")
                    raise NotImplementedError

        if plot_patches:
            plot_data = []

        intrinsics = intrinsics / self.RES
        if disps is not None:
            disps = disps[:, :, 1::4, 1::4].float()
        
        # ==== STEP 2: EXTRACT EVENT PATCHES AND FEATURES ====
        # This implements the patch extraction from Paper Equation (1)
        if self.patch_selector == SelectionMethod.SCORER:
            fmap, gmap, imap, patches, ix, scores = self.patchify(images, patches_per_image=patches_per_image, disps=disps)
        else:
            fmap, gmap, imap, patches, ix = self.patchify(images, patches_per_image=patches_per_image, disps=disps)

        # Output shapes:
        # fmap: (B, N_frames, 128, H/4, W/4) - Dense matching features
        # gmap: (B, N_patches, 128, 3, 3) - Patch matching features
        # imap: (B, N_patches, 384, 1, 1) - Patch context features
        # patches: (B, N_patches, 3, 3, 3) - Patch coordinates [x, y, depth]
        # ix: (N_patches,) - Frame index for each patch
        # Example: 15 frames * 80 patches/frame = 1200 total patches

        # Build correlation function for multi-scale matching
        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        # Save ground truth for supervision
        patches_gt = patches.clone()  # Ground truth patch coordinates
        Ps = poses  # Ground truth poses

        # ==== STEP 3: INITIALIZE DEPTH ====
        # Initialize patch depths randomly (will be optimized)
        d = patches[..., 2, p//2, p//2]  # Extract center depth
        patches = set_depth(patches, torch.rand_like(d))  # Random initialization

        # ==== STEP 4: BUILD INITIAL PATCH GRAPH ====
        # Use first 8 frames for initialization (Paper mentions 8-frame initialization)
        # Create edges in the patch co-visibility graph
        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"), indexing="ij")
        # kk: Patch indices (N_edges,) - which patches
        # jj: Target frame indices (N_edges,) - project patches to which frames
        # Example: 640 patches from first 8 frames × 8 target frames = 5120 edges
        ii = ix[kk]  # Source frame indices for each edge

        # Reshape context features for indexing
        imap = imap.view(b, -1, self.dim_inet)  # (B, N_patches, 384)

        # Initialize hidden state for recurrent update
        net = torch.zeros(b, len(kk), self.dim_inet, device="cuda", dtype=torch.float)

        # Initialize predicted poses to identity
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            # Fix poses to ground truth, only optimize structure (depths)
            Gs.data[:] = poses.data[:]

        # ==== STEP 5: ITERATIVE REFINEMENT ====
        # Run STEPS iterations of the update operator + BA (default: 12 iterations)
        traj = []  # Store results for each iteration
        bounds = [-64, -64, w + 64, h + 64]  # Image bounds for clipping

        while len(traj) < STEPS:
            # Detach gradients for each iteration (fresh optimization)
            Gs = Gs.detach()
            patches = patches.detach()

            # ==== INCREMENTAL FRAME ADDITION ====
            # After 8 iterations, start adding new frames incrementally
            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                # Initialize new frame pose from previous frame
                if not structure_only:
                    Gs.data[:,n] = Gs.data[:,n-1]

                # Add edges for new frame to patch graph
                # kk1, jj1: Edges from existing patches to new frame
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                # kk2, jj2: Edges from new patches to all existing frames
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                # Update graph indices
                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                # Initialize hidden states for new edges
                net1 = torch.zeros(b, len(kk1) + len(kk2), self.dim_inet, device="cuda")
                net = torch.cat([net1, net], dim=1)

                # Randomly remove old edges (10% probability) to maintain graph size
                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                # Initialize new patch depths from previous frames
                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            # ==== PROJECTION AND CORRELATION ====
            # Project patches from frame i to frame j using current pose estimates
            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()  # (B,edges,P,P,2) -> (B,edges,2,P,P)

            # Compute multi-scale correlation features
            corr = corr_fn(kk, jj, coords1)
            # corr: (B, num_edges, 2*49*P*P) - Correlation from 2 pyramid levels × 7x7 window

            # ==== RECURRENT UPDATE: PREDICT OPTICAL FLOW ====
            # This implements the Update operator from Paper Section III-A
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)
            # delta: (B, num_edges, 2) - Predicted flow correction δ_inj
            # weight: (B, num_edges, 2) - Confidence weights Σ_inj
            # net: (B, num_edges, 384) - Updated hidden state

            # ==== DIFFERENTIABLE BUNDLE ADJUSTMENT ====
            # This implements Paper Equation (2): Minimize reprojection error
            lmbda = 1e-4  # Damping factor for Gauss-Newton
            target = coords[...,p//2,p//2,:] + delta  # Add flow correction to projection

            # Run 2 iterations of Gauss-Newton optimization
            # Paper Section III-B: "perform two Gauss-Newton iterations"
            ep = 10  # Edge weight damping
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk,
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            # ==== COLLECT SUPERVISION SIGNALS ====
            # Prepare data for computing training losses (Paper Section III-B)
            kl = torch.as_tensor(0)  # KL divergence (not used)
            dij = (ii - jj).abs()  # Frame distance for each edge

            # Select close edges for flow supervision (temporal neighbors within 2 frames)
            k = (dij > 0) & (dij <= 2)

            if self.patch_selector == SelectionMethod.SCORER:
                # Full supervision for scorer training
                coords_full = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
                coords_gt_full, valid_full = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk, valid=True)

                # Close edge supervision for flow loss
                coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
                coords_gt, valid = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], valid=True)

                # Extended edges for score loss (within 16 frames)
                k = (dij > 0) & (dij <= 16)

                # Append supervision tuple:
                # - valid, coords, coords_gt: For L_flow (optical flow loss)
                # - Gs[:,:n], Ps[:,:n]: For L_pose (pose loss)
                # - scores: For L_score (patch selection loss)
                # - Additional data for extended supervision
                traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl, scores,
                           valid_full[0,k], coords_full.detach()[0,k], coords_gt_full.detach()[0,k],
                           weight.detach()[0,k], kk[k], dij[k]))
            else:
                # Standard supervision (no scorer)
                coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
                coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], valid=True)

                traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))
            
            # Optional: Save visualization data
            if plot_patches:
                coords_gt = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk)
                coords1_gt = coords_gt.permute(0, 1, 4, 2, 3).contiguous()
                coordsAll = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
                coordsAll = coordsAll.permute(0, 1, 4, 2, 3).contiguous()
                plot_data.append((ii, jj, patches, coordsAll, coords1_gt))

        # Return trajectory with all iterations
        # Each element in traj contains supervision signals for one iteration
        if plot_patches:
            traj.append(plot_data)
        return traj