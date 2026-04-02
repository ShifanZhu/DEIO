"""
Inspect TartanAir EVS (event + depth) dataset.

Directory layout:
  <root>/<scene>/<Easy|Hard>/<P00X>/
      evs_left/h5/<frame>.h5     — voxel grids  (5, H, W) float32
      depth_left/<frame>.npy     — depth maps    (H, W) float32
      pose_left.txt              — poses         (N, 7) [tx,ty,tz,qx,qy,qz,qw]

Usage:
    conda run -n DEIO python script/inspect_tartan.py \
        --datadir /media/s/rell/tartan/abandonedfactory/Easy/P000 \
        --outdir /tmp/tartan_inspect \
        --stride 50
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── helpers ────────────────────────────────────────────────────────────────────

def load_voxel(path):
    """Load voxel grid from .h5 file → (5, H, W) float32 ndarray."""
    with h5py.File(path, "r") as f:
        return f["voxel"][:].astype(np.float32)   # stored as float16, cast for viz

def load_depth(path):
    """Load depth map from .npy file → (H, W) float32 ndarray."""
    d = np.load(path).astype(np.float32)
    d[~np.isfinite(d)] = 1.0
    return d

def load_poses(path):
    """Load poses → (N, 7) [tx, ty, tz, qx, qy, qz, qw]."""
    poses = np.loadtxt(path)
    # TartanAir is in NED order (z,x,y); remap to (x,y,z)
    poses_xyz = poses[:, [1, 2, 0, 4, 5, 3, 6]]
    return poses_xyz


# ── per-sample inspection ──────────────────────────────────────────────────────

def print_sample_info(voxel, depth, pose, idx):
    print(f"\n  Frame {idx:>4d}")
    print(f"    voxel  shape : {voxel.shape}   (bins, H, W)")
    print(f"    voxel  dtype : {voxel.dtype}")
    print(f"    voxel  range : [{voxel.min():.4f}, {voxel.max():.4f}]   "
          f"mean={voxel.mean():.6f}  std={voxel.std():.4f}")
    nz = (np.abs(voxel) > 1e-4).mean()
    print(f"    voxel  nonzero: {nz:.4f}  ({nz*100:.2f}%)")
    print(f"    depth  shape : {depth.shape}   range=[{depth.min():.2f}, {depth.max():.2f}] m")
    print(f"    pose         : tx={pose[0]:.3f}  ty={pose[1]:.3f}  tz={pose[2]:.3f}  "
          f"qx={pose[3]:.3f}  qy={pose[4]:.3f}  qz={pose[5]:.3f}  qw={pose[6]:.3f}")


# ── figures ────────────────────────────────────────────────────────────────────

def save_single_frame(voxel, depth, pose, idx, outpath):
    """
    One figure per frame showing:
      Row 1: 5 temporal voxel bins (positive=ON, negative=OFF)
      Row 2: collapsed voxel (sum bins) | depth map | disparity (1/depth)
    """
    nbins = voxel.shape[0]
    fig = plt.figure(figsize=(4 * nbins, 8))
    gs = gridspec.GridSpec(2, nbins, figure=fig, hspace=0.35, wspace=0.05)

    # Row 1 – individual bins
    for b in range(nbins):
        ax = fig.add_subplot(gs[0, b])
        bin_img = voxel[b]
        vmax = max(np.abs(bin_img).max(), 1e-3)
        ax.imshow(bin_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"voxel bin {b}", fontsize=9)
        ax.axis("off")

    # Row 2, col 0 – collapsed voxel (sum over bins)
    ax = fig.add_subplot(gs[1, 0])
    collapsed = voxel.sum(axis=0)
    vmax = max(np.abs(collapsed).max(), 1e-3)
    ax.imshow(collapsed, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("voxel (sum all bins)", fontsize=9)
    ax.axis("off")

    # Row 2, col 1 – depth map
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(depth, cmap="magma_r", vmin=0, vmax=np.percentile(depth, 95))
    ax.set_title(f"depth  [{depth.min():.1f}, {depth.max():.1f}] m", fontsize=9)
    ax.axis("off")

    # Row 2, col 2 – disparity
    ax = fig.add_subplot(gs[1, 2])
    disp = 1.0 / np.clip(depth, 0.1, None)
    ax.imshow(disp, cmap="plasma")
    ax.set_title("disparity (1/depth)", fontsize=9)
    ax.axis("off")

    # Row 2, col 3 – polarity mask (ON / OFF counts per-pixel across bins)
    if nbins >= 4:
        ax = fig.add_subplot(gs[1, 3])
        on_map  = np.clip( voxel, 0, None).sum(axis=0)
        off_map = np.clip(-voxel, 0, None).sum(axis=0)
        rgb = np.stack([
            off_map / (off_map.max() + 1e-6),
            np.zeros_like(on_map),
            on_map  / (on_map.max()  + 1e-6),
        ], axis=-1)
        ax.imshow(rgb)
        ax.set_title("ON (blue) vs OFF (red)", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"TartanAir EVS  frame={idx:04d}\n"
                 f"pose: t=[{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]  "
                 f"q=[{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}, {pose[6]:.3f}]",
                 fontsize=10)
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved frame viz       → {outpath}")


def save_summary_grid(samples, outpath):
    """Grid of collapsed voxels + depth maps for all frames."""
    n = len(samples)
    ncols = min(n, 6)
    nrows = ((n + ncols - 1) // ncols) * 2  # two rows per strip: voxel + depth

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for col, (idx, voxel, depth) in enumerate(samples):
        strip_row = (col // ncols) * 2
        strip_col = col % ncols

        # voxel row
        ax = axes[strip_row, strip_col]
        collapsed = voxel.sum(axis=0)
        vmax = max(np.abs(collapsed).max(), 1e-3)
        ax.imshow(collapsed, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"f={idx:04d}\nvoxel", fontsize=7)
        ax.axis("off")

        # depth row
        ax = axes[strip_row + 1, strip_col]
        ax.imshow(depth, cmap="magma_r", vmin=0, vmax=np.percentile(depth, 95))
        ax.set_title("depth", fontsize=7)
        ax.axis("off")

    # hide unused cells
    for r in range(nrows):
        for c in range(ncols):
            col_global = (r // 2) * ncols + c
            if col_global >= n:
                axes[r, c].axis("off")

    fig.suptitle("TartanAir EVS – voxel (∑bins) + depth per frame", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary grid    → {outpath}")


def save_trajectory(poses, outpath):
    """Top-down and side-view of the ground truth trajectory."""
    t = poses[:, :3]   # (N, 3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(t[:, 0], t[:, 1])
    axes[0].scatter(t[0, 0], t[0, 1], color="green", zorder=5, label="start")
    axes[0].scatter(t[-1, 0], t[-1, 1], color="red", zorder=5, label="end")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    axes[0].set_title("Top-down view (x-y)"); axes[0].legend(); axes[0].set_aspect("equal")

    axes[1].plot(t[:, 0], t[:, 2])
    axes[1].scatter(t[0, 0], t[0, 2], color="green", zorder=5)
    axes[1].scatter(t[-1, 0], t[-1, 2], color="red", zorder=5)
    axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("z (m)")
    axes[1].set_title("Side view (x-z)"); axes[1].set_aspect("equal")

    axes[2].plot(t[:, 1], t[:, 2])
    axes[2].scatter(t[0, 1], t[0, 2], color="green", zorder=5)
    axes[2].scatter(t[-1, 1], t[-1, 2], color="red", zorder=5)
    axes[2].set_xlabel("y (m)"); axes[2].set_ylabel("z (m)")
    axes[2].set_title("Side view (y-z)"); axes[2].set_aspect("equal")

    fig.suptitle(f"Ground truth trajectory  ({len(poses)} frames)", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved trajectory      → {outpath}")


def save_dataset_stats(voxels_stack, depths_stack, outpath):
    """Histograms and stats across all inspected frames."""
    # voxels_stack: (N, 5, H, W)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Voxel value distribution
    flat = voxels_stack.ravel()
    nonzero = flat[np.abs(flat) > 1e-4]
    axes[0].hist(nonzero, bins=80, color="steelblue", edgecolor="none")
    axes[0].set_title(f"Voxel value histogram (nonzero)\n"
                      f"mean={nonzero.mean():.4f}  std={nonzero.std():.4f}")
    axes[0].set_xlabel("voxel value")

    # Per-bin mean activation across frames
    mean_per_bin = np.abs(voxels_stack).mean(axis=(0, 2, 3))  # (5,)
    axes[1].bar(range(len(mean_per_bin)), mean_per_bin, color="steelblue")
    axes[1].set_title("Mean |voxel| per temporal bin\n(averaged over frames & pixels)")
    axes[1].set_xlabel("bin index"); axes[1].set_ylabel("mean |activation|")
    axes[1].set_xticks(range(len(mean_per_bin)))

    # Depth distribution
    flat_d = depths_stack.ravel()
    flat_d = flat_d[np.isfinite(flat_d) & (flat_d > 0) & (flat_d < 1000)]
    axes[2].hist(flat_d, bins=80, color="darkorange", edgecolor="none")
    axes[2].set_title(f"Depth histogram\nmean={flat_d.mean():.2f} m  "
                      f"median={np.median(flat_d):.2f} m")
    axes[2].set_xlabel("depth (m)")

    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved dataset stats   → {outpath}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect TartanAir EVS sequence")
    parser.add_argument("--datadir", default="/media/s/rell/tartan/abandonedfactory/Easy/P000",
                        help="Path to a single sequence (contains evs_left/, depth_left/, pose_left.txt)")
    parser.add_argument("--outdir", default="/tmp/tartan_inspect",
                        help="Directory to save output images")
    parser.add_argument("--stride", type=int, default=1,
                        help="Step between frames (1=every frame, 50=every 50th frame)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── discover files ────────────────────────────────────────────────────────
    import glob as _glob
    voxel_files = sorted(_glob.glob(os.path.join(args.datadir, "evs_left", "h5", "*.h5")))
    depth_files = sorted(_glob.glob(os.path.join(args.datadir, "depth_left", "*.npy")))
    pose_file   = os.path.join(args.datadir, "pose_left.txt")

    if not voxel_files:
        raise FileNotFoundError(f"No .h5 voxel files in {args.datadir}/evs_left/h5/")
    if not depth_files:
        raise FileNotFoundError(f"No .npy depth files in {args.datadir}/depth_left/")

    # TartanAirEVS skips first depth (offset by 1)
    depth_files = depth_files[1:]
    assert len(voxel_files) == len(depth_files), \
        f"Mismatch: {len(voxel_files)} voxels vs {len(depth_files)} depths"

    poses_raw = np.loadtxt(pose_file)           # (N, 7), NED convention
    poses = poses_raw[1:, [1, 2, 0, 4, 5, 3, 6]]  # → (x, y, z, qx, qy, qz, qw)
    assert len(poses) == len(voxel_files), \
        f"Mismatch: {len(poses)} poses vs {len(voxel_files)} voxels"

    print(f"\n{'='*60}")
    print(f"Sequence : {args.datadir}")
    print(f"Frames   : {len(voxel_files)} total")
    print(f"{'='*60}")

    # ── print one voxel h5 structure ─────────────────────────────────────────
    print(f"\nHDF5 voxel file structure ({os.path.basename(voxel_files[0])}):")
    with h5py.File(voxel_files[0], "r") as f:
        def _show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  [{name}]  shape={obj.shape}  dtype={obj.dtype}")
        f.visititems(_show)

    # ── sample frames ─────────────────────────────────────────────────────────
    total = len(voxel_files)
    indices = list(range(0, total, args.stride))
    print(f"\nInspecting {len(indices)} frames (stride={args.stride})")

    voxels_all, depths_all, samples_for_grid = [], [], []

    for idx in indices:
        voxel = load_voxel(voxel_files[idx])   # (5, H, W)
        depth = load_depth(depth_files[idx])   # (H, W)
        pose  = poses[idx]                     # (7,)

        print_sample_info(voxel, depth, pose, idx)

        # Per-frame figure
        save_single_frame(voxel, depth, pose, idx,
                          os.path.join(args.outdir, f"frame_{idx:04d}.png"))

        # Also save raw numpy
        np.save(os.path.join(args.outdir, f"voxel_{idx:04d}.npy"), voxel)
        np.save(os.path.join(args.outdir, f"depth_{idx:04d}.npy"), depth)

        voxels_all.append(voxel)
        depths_all.append(depth)
        samples_for_grid.append((idx, voxel, depth))

    # ── summary figures ───────────────────────────────────────────────────────
    save_summary_grid(samples_for_grid,
                      os.path.join(args.outdir, "summary_grid.png"))

    save_trajectory(poses,
                    os.path.join(args.outdir, "trajectory.png"))

    save_dataset_stats(np.stack(voxels_all),   # (N, 5, H, W)
                       np.stack(depths_all),   # (N, H, W)
                       os.path.join(args.outdir, "dataset_stats.png"))

    # ── print final summary ───────────────────────────────────────────────────
    voxel0 = voxels_all[0]
    depth0 = depths_all[0]
    H, W = depth0.shape
    print(f"\n{'='*60}")
    print("Data type summary:")
    print(f"  voxel  : shape=(5, {H}, {W})  dtype=float32  "
          f"range=[{np.stack(voxels_all).min():.3f}, {np.stack(voxels_all).max():.3f}]")
    print(f"  depth  : shape=({H}, {W})    dtype=float32  unit=meters")
    print(f"  pose   : shape=(7,)          [tx, ty, tz, qx, qy, qz, qw]")
    print(f"  intrinsics (fixed): fx=320  fy=320  cx=320  cy=240")
    print(f"\n  All outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
