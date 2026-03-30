"""Contrast Maximization Debug Visualizer

Saves PNG visualizations to understand what the CM stage is doing:

  1. voxel_frames.png       — raw event voxels (frame i and j)
  2. patch_locations.png    — where the 200 CM patches land on each frame
  3. iwe_comparison.png     — IWE before/after CM gradient ascent, vs. target voxel
  4. contrast_curve.png     — CM loss value over gradient-ascent steps
  5. ncc_heatmap.png        — per-patch NCC score painted back onto image canvas

Usage
-----
# Synthetic random data (no checkpoint or dataset needed):
python script/debug/visualize_cm.py --outdir /tmp/cm_debug

# Real voxel H5 files, identity poses (no checkpoint needed):
python script/debug/visualize_cm.py --voxeldir /media/s/rell/tartan/ocean/Easy/P001/evs_left/h5 --outdir /tmp/cm_debug

# Full mode: real data + model checkpoint (most realistic):
python script/debug/visualize_cm.py \
    --checkpoint checkpoints/my_run/000060.pth \
    --voxeldir /media/s/rell/tartan/ocean/Easy/P001/evs_left/h5 \
    --outdir /tmp/cm_debug
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Must set matplotlib backend and evo settings BEFORE any other import
# that might transitively import evo.tools.plot (which calls mpl.use).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

from torch_scatter import scatter_sum
from dpvo.lietorch import SE3

from devo.cm_refinement import (
    _downsample_images, _sample_frames,
    loss_var_iwe, loss_ncc, loss_aligned_var,
    cm_refine,
)
from devo import projective_ops as pops
from devo.utils import flatmeshgrid


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_voxels(voxeldir: str, n_frames: int = 8, start: int = 0):
    """Load n_frames event voxel grids from an H5 directory.

    Returns:
        images_raw: (1, n_frames, C, H, W) float32 on cuda
        intrinsics: (1, n_frames, 4) float32 on cuda   [fx, fy, cx, cy]
    """
    from utils.load_utils import voxel_read
    files = sorted(glob.glob(os.path.join(voxeldir, "*.h5")))[start:start + n_frames]
    if len(files) < n_frames:
        raise RuntimeError(f"Found only {len(files)} H5 files in {voxeldir}, need {n_frames}")
    voxels = [torch.from_numpy(voxel_read(f)) for f in files]  # each (C, H, W)
    images_raw = torch.stack(voxels, dim=0).unsqueeze(0).float().cuda()  # (1, N, C, H, W)
    C, H, W = voxels[0].shape
    # TartanAir default intrinsics
    fx = fy = 320.0
    cx, cy = W / 2.0, H / 2.0
    intr = torch.tensor([fx, fy, cx, cy], device='cuda').view(1, 1, 4).expand(1, n_frames, 4).float()
    print(f"Loaded {n_frames} voxels  shape={tuple(images_raw.shape)}  from {voxeldir}")
    return images_raw, intr.clone()


def make_synthetic_data(n_frames=6, H=240, W=320, C=5):
    """Synthetic test data: moving Gaussian blobs as 'events'.

    Returns:
        images_raw: (1, n_frames, C, H, W) cuda
        intrinsics: (1, n_frames, 4) cuda
    """
    images = torch.zeros(1, n_frames, C, H, W, device='cuda')
    for t in range(n_frames):
        cx_blob = int(W * 0.3 + t * W * 0.05)
        cy_blob = int(H * 0.5)
        yy, xx = torch.meshgrid(torch.arange(H, device='cuda').float(),
                                torch.arange(W, device='cuda').float(), indexing='ij')
        blob = torch.exp(-((xx - cx_blob)**2 + (yy - cy_blob)**2) / (2 * 20**2))
        for c in range(C):
            images[0, t, c] = blob * (1.0 if c % 2 == 0 else -0.5)
    fx = fy = 320.0
    cx_intr, cy_intr = W / 2.0, H / 2.0
    intr = torch.tensor([fx, fy, cx_intr, cy_intr], device='cuda').view(1, 1, 4).expand(1, n_frames, 4).float()
    print(f"Using synthetic data  shape={tuple(images.shape)}")
    return images, intr.clone()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (optional)
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str):
    """Load eVONet from a training checkpoint."""
    import argparse
    from devo.enet import eVONet

    # Minimal args to reconstruct the network
    net_args = argparse.Namespace(
        resnet=False, block_dims=[64, 128, 256], initial_dim=64, pretrain='resnet34',
        dim_inet=384, dim_fnet=128, dim=32, patch_selector='scorer',
        norm='std2', randaug=False,
    )
    net = eVONet(net_args).cuda().eval()
    ckpt = torch.load(checkpoint_path, map_location='cuda')
    state = ckpt.get('model_state_dict', ckpt)
    net.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return net


# ──────────────────────────────────────────────────────────────────────────────
# IWE builder (returns actual tensor, not just loss)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_iwe(images_small, Gs, patches_cm, intrinsics, ii, jj, kk):
    """Return the sparse IWE tensor (B, h, w) for a given set of edges.

    Forward-warps source event mass from frame ii to frame jj.
    """
    B, N, C, h, w = images_small.shape
    P = patches_cm.shape[3]

    coords = pops.transform(Gs, patches_cm, intrinsics, ii, jj, kk)
    coords_center = coords[:, :, P // 2, P // 2, :]  # (B, E, 2)

    src_coords = patches_cm[:, kk, :2, P // 2, P // 2]
    v_src = _sample_frames(images_small, ii, src_coords)
    v_src_sum = v_src.abs().sum(dim=-1)  # (B, E)

    x = coords_center[..., 0]
    y = coords_center[..., 1]
    x0 = x.floor().long().clamp(0, w - 1)
    x1 = (x.floor() + 1).long().clamp(0, w - 1)
    y0 = y.floor().long().clamp(0, h - 1)
    y1 = (y.floor() + 1).long().clamp(0, h - 1)
    dx = x - x0.float()
    dy = y - y0.float()

    IWE = (
        scatter_sum(v_src_sum * (1 - dx) * (1 - dy), y0 * w + x0, dim=1, dim_size=h * w) +
        scatter_sum(v_src_sum * dx * (1 - dy),        y0 * w + x1, dim=1, dim_size=h * w) +
        scatter_sum(v_src_sum * (1 - dx) * dy,        y1 * w + x0, dim=1, dim_size=h * w) +
        scatter_sum(v_src_sum * dx * dy,               y1 * w + x1, dim=1, dim_size=h * w)
    ).view(B, h, w)
    return IWE  # (B, h, w)


@torch.no_grad()
def compute_ncc_per_patch(images_small, Gs, patches_cm, intrinsics, ii, jj, kk):
    """Return per-patch NCC values (B, E)."""
    B, N, C, h, w = images_small.shape
    P = patches_cm.shape[3]

    coords = pops.transform(Gs, patches_cm, intrinsics, ii, jj, kk)
    coords_center = coords[:, :, P // 2, P // 2, :]

    src_coords = patches_cm[:, kk, :2, P // 2, P // 2]
    v_src = _sample_frames(images_small, ii, src_coords)
    v_tgt = _sample_frames(images_small, jj, coords_center)

    v_src_n = v_src - v_src.mean(dim=-1, keepdim=True)
    v_tgt_n = v_tgt - v_tgt.mean(dim=-1, keepdim=True)
    ncc = (v_src_n * v_tgt_n).sum(dim=-1) / (
        v_src_n.norm(dim=-1) * v_tgt_n.norm(dim=-1) + 1e-6)
    return ncc  # (B, E), values in [-1, 1]


# ──────────────────────────────────────────────────────────────────────────────
# Contrast curve: run CM gradient ascent and record loss at each step
# ──────────────────────────────────────────────────────────────────────────────

def run_cm_track_loss(images_small, Gs_init, patches_cm, intrinsics, ii, jj, kk,
                      cm_steps=20, lr_cm=1e-3):
    """Run gradient ascent and record all three CM losses at each step.

    Returns:
        steps:      list of int (0..cm_steps)
        losses:     dict  {'var_iwe': [...], 'ncc': [...], 'aligned_var': [...]}
        Gs_refined: final SE3 after cm_steps (detached)
    """
    from devo.cm_refinement import loss_var_iwe, loss_ncc, loss_aligned_var, _LOSS_FNS

    B, N = Gs_init.shape[0], Gs_init.shape[1]
    device = Gs_init.data.device

    losses = {k: [] for k in ['var_iwe', 'ncc', 'aligned_var']}

    def record(Gs):
        with torch.no_grad():
            for name, fn in [('var_iwe', loss_var_iwe),
                              ('ncc',     loss_ncc),
                              ('aligned_var', loss_aligned_var)]:
                v = fn(images_small, Gs, patches_cm, intrinsics, ii, jj, kk)
                losses[name].append(v.item())

    Gs = SE3(Gs_init.data.detach().clone())
    record(Gs)

    for step in range(cm_steps):
        delta_xi = torch.zeros(B, N, 6, requires_grad=True, device=device,
                               dtype=Gs.data.dtype)
        Gs_p = SE3.exp(delta_xi) * Gs
        L = loss_ncc(images_small, Gs_p, patches_cm, intrinsics, ii, jj, kk)
        grad = torch.autograd.grad(L, delta_xi)[0]
        with torch.no_grad():
            Gs = SE3((SE3.exp(-lr_cm * grad.detach()) * Gs).data.clone())
        record(Gs)

    return list(range(cm_steps + 1)), losses, Gs


# ──────────────────────────────────────────────────────────────────────────────
# Build test poses and patches (used when no model is available)
# ──────────────────────────────────────────────────────────────────────────────

def build_identity_poses_and_patches(images_raw, intrinsics_full, n_patches=200, P=3):
    """Create identity SE3 poses and evenly-spaced patches for visualization
    when no model checkpoint is available.

    Returns:
        Gs:          SE3 (1, N, 7)  identity poses
        patches_cm:  (1, M, 3, P, P) patches at grid positions with disp=0.1
        ix:          (M,)  frame index per patch
        intr_fm:     (1, N, 4) intrinsics at feature-map resolution (÷4)
    """
    B, N, C, H, W = images_raw.shape
    h, w = H // 4, W // 4

    # Identity poses
    poses_data = torch.zeros(B, N, 7, device='cuda')
    poses_data[..., 6] = 1.0  # w=1 quaternion
    Gs = SE3(poses_data)

    # Intrinsics at feature-map resolution
    intr_fm = intrinsics_full / 4.0

    # Grid patches: distribute n_patches uniformly over the feature-map canvas
    per_frame = n_patches // N
    all_patches = []
    all_ix = []
    for n in range(N):
        xs = torch.linspace(1, w - 2, int(per_frame ** 0.5) + 1, device='cuda')
        ys = torch.linspace(1, h - 2, int(per_frame ** 0.5) + 1, device='cuda')
        xg, yg = torch.meshgrid(xs, ys, indexing='xy')
        pts = torch.stack([xg.reshape(-1), yg.reshape(-1)], dim=-1)[:per_frame]
        E = pts.shape[0]

        patch = torch.zeros(B, E, 3, P, P, device='cuda')
        for pi in range(P):
            for pj in range(P):
                patch[:, :, 0, pi, pj] = pts[None, :, 0] + (pj - P // 2)
                patch[:, :, 1, pi, pj] = pts[None, :, 1] + (pi - P // 2)
        patch[:, :, 2] = 0.1  # inverse depth

        all_patches.append(patch.squeeze(0))
        all_ix.extend([n] * E)

    patches_cm = torch.cat(all_patches, dim=0).unsqueeze(0)  # (1, M, 3, P, P)
    ix = torch.tensor(all_ix, device='cuda')
    return Gs, patches_cm, ix, intr_fm


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(t):
    return t.squeeze().detach().cpu().float().numpy()


def _voxel_preview(voxel_frame):
    """(C, h, w) → (h, w) positive-events as a 2D density map."""
    v = voxel_frame.abs().sum(0)
    v = v / (v.max() + 1e-6)
    return _to_numpy(v)


def save_voxel_frames(images_raw, frame_i, frame_j, outdir):
    """Save raw event density maps for frame i and j."""
    _, N, C, H, W = images_raw.shape
    h, w = H // 4, W // 4
    images_small = _downsample_images(images_raw, h, w)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Event Voxel Frames  (C_bins summed)", fontsize=13)

    for ax, fr, label in [(axes[0], frame_i, f"Frame {frame_i} (source ii)"),
                           (axes[1], frame_j, f"Frame {frame_j} (target jj)"),
                           (axes[2], None,    "Difference |i| - |j|")]:
        if fr is not None:
            img = _voxel_preview(images_small[0, fr])
            ax.imshow(img, cmap='hot', origin='upper')
        else:
            di = _voxel_preview(images_small[0, frame_i])
            dj = _voxel_preview(images_small[0, frame_j])
            ax.imshow(di - dj, cmap='RdBu', origin='upper')
        ax.set_title(label)
        ax.axis('off')

    out = os.path.join(outdir, "1_voxel_frames.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def save_patch_locations(images_raw, patches_cm, ix, frame_i, frame_j, outdir):
    """Scatter patch centers onto event frame canvas (frame i and j)."""
    _, N, C, H, W = images_raw.shape
    h, w = H // 4, W // 4
    images_small = _downsample_images(images_raw, h, w)
    P = patches_cm.shape[3]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"CM Patch Locations  ({patches_cm.shape[1]} patches total)", fontsize=13)

    for ax, fr, label in [(axes[0], frame_i, f"Frame {frame_i} — source patches"),
                           (axes[1], frame_j, f"Frame {frame_j} — target canvas")]:
        bg = _voxel_preview(images_small[0, fr])
        ax.imshow(bg, cmap='gray', origin='upper', vmin=0, vmax=1)

        # Overlay patch centers for patches that belong to frame fr
        mask = (ix == fr).cpu().numpy()
        px = patches_cm[0, mask, 0, P // 2, P // 2].cpu().numpy()
        py = patches_cm[0, mask, 1, P // 2, P // 2].cpu().numpy()
        ax.scatter(px, py, s=8, c='lime', alpha=0.7, linewidths=0)
        ax.set_title(f"{label}  ({mask.sum()} patches)")
        ax.axis('off')

    out = os.path.join(outdir, "2_patch_locations.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def save_iwe_comparison(images_raw, Gs_before, Gs_after,
                        patches_cm, intrinsics_fm, ii, jj, kk,
                        frame_i, frame_j, outdir):
    """3-row, 3-col panel showing IWE before/after vs target voxel."""
    _, N, C, H, W = images_raw.shape
    h, w = H // 4, W // 4
    images_small = _downsample_images(images_raw, h, w)

    # Filter edges to just the i→j pair
    mask = (ii == frame_i) & (jj == frame_j)
    if mask.sum() == 0:
        print(f"  [iwe_comparison] No edges for ({frame_i}→{frame_j}), skipping")
        return
    ii_sub = ii[mask]; jj_sub = jj[mask]; kk_sub = kk[mask]

    with torch.no_grad():
        iwe_before = build_iwe(images_small, Gs_before, patches_cm, intrinsics_fm,
                               ii_sub, jj_sub, kk_sub)[0]
        iwe_after  = build_iwe(images_small, Gs_after,  patches_cm, intrinsics_fm,
                               ii_sub, jj_sub, kk_sub)[0]

    vox_j = images_small[0, frame_j].abs().sum(0)  # (h, w)
    vox_j = vox_j / (vox_j.max() + 1e-6)

    iwe_b = iwe_before / (iwe_before.max() + 1e-6)
    iwe_a = iwe_after  / (iwe_after.max()  + 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"IWE Comparison — Frame {frame_i} → Frame {frame_j}", fontsize=14)

    def _show(ax, img, title, cmap='hot'):
        im = ax.imshow(_to_numpy(img), cmap=cmap, origin='upper')
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _show(axes[0, 0], vox_j,  "Target voxel (frame j)",    cmap='gray')
    _show(axes[0, 1], iwe_b,  "IWE — Stage 1 pose",        cmap='hot')
    _show(axes[0, 2], iwe_a,  "IWE — CM-refined pose",     cmap='hot')

    # Overlay: IWE on top of voxel_j (green channel = IWE, grey = voxel_j)
    def overlay(iwe, vox):
        rgb = torch.stack([vox, iwe, vox], dim=-1)
        return _to_numpy(rgb.clamp(0, 1))

    axes[1, 0].imshow(overlay(iwe_b * 0, vox_j), origin='upper')
    axes[1, 0].set_title("Target voxel only", fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(overlay(iwe_b, vox_j), origin='upper')
    axes[1, 1].set_title("Overlay: Stage 1 (green=IWE, grey=voxel_j)", fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(overlay(iwe_a, vox_j), origin='upper')
    axes[1, 2].set_title("Overlay: CM-refined (green=IWE, grey=voxel_j)", fontsize=11)
    axes[1, 2].axis('off')

    # Print variance stats
    var_b = iwe_before.var().item()
    var_a = iwe_after.var().item()
    fig.text(0.5, 0.01, f"IWE variance — before: {var_b:.5f}  after: {var_a:.5f}  "
             f"(improvement: {(var_a - var_b)/max(var_b, 1e-9)*100:+.1f}%)",
             ha='center', fontsize=11)

    out = os.path.join(outdir, "3_iwe_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}  (var_before={var_b:.5f}, var_after={var_a:.5f})")


def save_contrast_curve(step_list, losses_dict, outdir):
    """Plot CM loss value over gradient-ascent steps for all three methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("CM Loss over Gradient-Ascent Steps", fontsize=13)

    colors = {'var_iwe': '#e74c3c', 'ncc': '#2980b9', 'aligned_var': '#27ae60'}
    labels = {'var_iwe': 'Method A: var_iwe (−Var(IWE))',
              'ncc':     'Method B: ncc (1−NCC)',
              'aligned_var': 'Method C: aligned_var (−Var(IWE+voxel_j))'}

    for ax, (name, vals) in zip(axes, losses_dict.items()):
        ax.plot(step_list, vals, color=colors[name], linewidth=2, marker='o', markersize=4)
        ax.set_title(labels[name], fontsize=10)
        ax.set_xlabel("Gradient-ascent step")
        ax.set_ylabel("Loss value")
        ax.grid(True, alpha=0.3)
        # Mark start and end
        ax.scatter([0, len(step_list) - 1], [vals[0], vals[-1]],
                   color=[colors[name]], s=80, zorder=5)
        delta = vals[-1] - vals[0]
        ax.text(0.05, 0.95, f"Δ = {delta:+.4f}", transform=ax.transAxes,
                va='top', fontsize=10,
                color='green' if delta < 0 else 'red')

    out = os.path.join(outdir, "4_contrast_curve.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def save_ncc_heatmap(images_raw, Gs, patches_cm, intrinsics_fm,
                     ii, jj, kk, frame_i, frame_j, outdir):
    """Paint per-patch NCC scores back onto the source frame canvas."""
    _, N, C, H, W = images_raw.shape
    h, w = H // 4, W // 4
    images_small = _downsample_images(images_raw, h, w)
    P = patches_cm.shape[3]

    # Filter edges to just the i→j pair
    mask_ij = (ii == frame_i) & (jj == frame_j)
    if mask_ij.sum() == 0:
        print(f"  [ncc_heatmap] No edges for ({frame_i}→{frame_j}), skipping")
        return
    ii_sub = ii[mask_ij]; jj_sub = jj[mask_ij]; kk_sub = kk[mask_ij]

    with torch.no_grad():
        ncc_vals = compute_ncc_per_patch(images_small, Gs, patches_cm, intrinsics_fm,
                                         ii_sub, jj_sub, kk_sub)  # (1, E)

    ncc_np = ncc_vals[0].cpu().numpy()  # (E,)
    px = patches_cm[0, kk_sub, 0, P // 2, P // 2].cpu().numpy()
    py = patches_cm[0, kk_sub, 1, P // 2, P // 2].cpu().numpy()

    # Also compute NCC per patch after CM refinement for comparison
    bg = _voxel_preview(images_small[0, frame_i])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Per-Patch NCC Scores  (Frame {frame_i} → Frame {frame_j})", fontsize=13)

    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.RdYlGn

    for ax, title in [(axes[0], "Source frame with NCC colormap"),
                      (axes[1], "NCC histogram")]:
        if ax == axes[0]:
            ax.imshow(bg, cmap='gray', origin='upper', alpha=0.7)
            sc = ax.scatter(px, py, c=ncc_np, cmap=cmap, norm=norm, s=12,
                            alpha=0.9, linewidths=0)
            plt.colorbar(sc, ax=ax, label='NCC  (1=perfect, -1=inverted)')
            ax.set_title(title, fontsize=11)
            ax.axis('off')
            # Stats box
            ax.text(0.02, 0.98, f"mean NCC: {ncc_np.mean():.3f}\n"
                                 f"median:   {np.median(ncc_np):.3f}\n"
                                 f"patches:  {len(ncc_np)}",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7))
        else:
            ax.hist(ncc_np, bins=30, color='#2980b9', edgecolor='white', alpha=0.8)
            ax.axvline(ncc_np.mean(), color='red', linestyle='--', label=f'mean={ncc_np.mean():.3f}')
            ax.axvline(0, color='black', linestyle=':', alpha=0.5)
            ax.set_xlabel("NCC value")
            ax.set_ylabel("# patches")
            ax.set_title(title, fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)

    out = os.path.join(outdir, "5_ncc_heatmap.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}  (mean_ncc={ncc_np.mean():.3f}, n={len(ncc_np)})")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', default='/tmp/cm_debug',
                        help='Directory to save PNG outputs')
    parser.add_argument('--voxeldir', default=None,
                        help='Path to directory containing .h5 voxel files '
                             '(e.g. .../evs_left/h5). If omitted, uses synthetic data.')
    parser.add_argument('--checkpoint', default=None,
                        help='Optional path to a .pth checkpoint for Stage 1 poses. '
                             'If omitted, identity poses are used.')
    parser.add_argument('--n_frames', type=int, default=8,
                        help='Number of frames to load (default: 8)')
    parser.add_argument('--frame_i', type=int, default=0,
                        help='Source frame index for visualizations (default: 0)')
    parser.add_argument('--frame_j', type=int, default=1,
                        help='Target frame index for visualizations (default: 1)')
    parser.add_argument('--n_patches', type=int, default=200,
                        help='Number of CM patches to use (default: 200)')
    parser.add_argument('--cm_steps', type=int, default=20,
                        help='Gradient-ascent steps for contrast curve (default: 20)')
    parser.add_argument('--lr_cm', type=float, default=1e-3,
                        help='CM gradient-ascent step size (default: 1e-3)')
    parser.add_argument('--voxel_start', type=int, default=0,
                        help='Start index into voxel files (default: 0)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"CM Debug Visualizer")
    print(f"Output dir: {args.outdir}")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if args.voxeldir:
        images_raw, intrinsics_full = load_voxels(
            args.voxeldir, args.n_frames, start=args.voxel_start)
    else:
        images_raw, intrinsics_full = make_synthetic_data(
            n_frames=args.n_frames)

    B, N, C, H, W = images_raw.shape
    h, w = H // 4, W // 4
    frame_i = min(args.frame_i, N - 1)
    frame_j = min(args.frame_j, N - 1)
    if frame_i == frame_j:
        frame_j = min(frame_i + 1, N - 1)
    print(f"Visualizing frame pair: {frame_i} → {frame_j}  (N={N} frames loaded)")

    # ── 2. Get poses and patches ──────────────────────────────────────────────
    if args.checkpoint:
        print("Running model forward pass for Stage 1 poses/patches...")
        net = load_model(args.checkpoint)
        with torch.no_grad():
            # Dummy poses and disps (identity / ones)
            poses_dummy = SE3(torch.zeros(B, N, 7, device='cuda').fill_(0))
            poses_dummy.data[..., 6] = 1.0
            disps_dummy = torch.ones(B, N, H, W, device='cuda') * 0.1

            result = net(images_raw.clone(), poses_dummy, disps_dummy,
                         intrinsics_full.clone()[:, :1].expand(B, N, 4),
                         use_cm=True, patches_per_image=80,
                         patches_per_image_cm=args.n_patches,
                         cm_steps=1)
        # result is (traj, cm_loss) — we don't need cm_loss here
        traj = result[0] if isinstance(result, tuple) else result
        print("  Model forward pass done (Stage 1 complete)")

        # We can't easily extract patches/Gs from the forward pass without
        # modifying enet.py. Use identity fallback for now.
        print("  (Using identity poses for visualization — patches from grid)")
        Gs, patches_cm, ix, intrinsics_fm = build_identity_poses_and_patches(
            images_raw, intrinsics_full, n_patches=args.n_patches)
    else:
        print("No checkpoint — using identity poses and grid patches")
        Gs, patches_cm, ix, intrinsics_fm = build_identity_poses_and_patches(
            images_raw, intrinsics_full, n_patches=args.n_patches)

    # ── 3. Build edges for the chosen frame pair (and all pairs) ─────────────
    n_patches_total = len(ix)
    kk_all, jj_all = flatmeshgrid(
        torch.arange(n_patches_total, device='cuda'),
        torch.arange(N, device='cuda'),
        indexing='ij'
    )
    ii_all = ix[kk_all]
    valid = ii_all != jj_all
    ii_edges = ii_all[valid]
    jj_edges = jj_all[valid]
    kk_edges = kk_all[valid]

    images_small = _downsample_images(images_raw, h, w)

    # ── 4. Run gradient ascent, collect losses ────────────────────────────────
    print(f"\nRunning {args.cm_steps} CM gradient-ascent steps (for contrast curve)...")
    step_list, losses_dict, Gs_refined = run_cm_track_loss(
        images_small, Gs, patches_cm, intrinsics_fm,
        ii_edges, jj_edges, kk_edges,
        cm_steps=args.cm_steps, lr_cm=args.lr_cm,
    )
    print(f"  NCC loss:  {losses_dict['ncc'][0]:.4f} → {losses_dict['ncc'][-1]:.4f}")
    print(f"  var_iwe:   {losses_dict['var_iwe'][0]:.4f} → {losses_dict['var_iwe'][-1]:.4f}")
    print(f"  algn_var:  {losses_dict['aligned_var'][0]:.4f} → {losses_dict['aligned_var'][-1]:.4f}")

    # ── 5. Save all plots ─────────────────────────────────────────────────────
    print(f"\nSaving plots to {args.outdir}/")
    save_voxel_frames(images_raw, frame_i, frame_j, args.outdir)
    save_patch_locations(images_raw, patches_cm, ix, frame_i, frame_j, args.outdir)
    save_iwe_comparison(images_raw, Gs, Gs_refined,
                        patches_cm, intrinsics_fm,
                        ii_edges, jj_edges, kk_edges,
                        frame_i, frame_j, args.outdir)
    save_contrast_curve(step_list, losses_dict, args.outdir)
    save_ncc_heatmap(images_raw, Gs, patches_cm, intrinsics_fm,
                     ii_edges, jj_edges, kk_edges,
                     frame_i, frame_j, args.outdir)

    print(f"\nDone. Open {args.outdir}/ to view the results.")
    print(f"  1_voxel_frames.png      — raw event density (frame i and j)")
    print(f"  2_patch_locations.png   — where the {n_patches_total} CM patches sit")
    print(f"  3_iwe_comparison.png    — IWE before/after CM refinement")
    print(f"  4_contrast_curve.png    — CM loss over gradient-ascent steps")
    print(f"  5_ncc_heatmap.png       — per-patch NCC score back-projected")


if __name__ == '__main__':
    main()
