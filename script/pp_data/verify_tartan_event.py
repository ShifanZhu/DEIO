"""
Verify processed TartanAir-Event sequences.

Checks:
  1. File counts: voxels == depths - 1 == poses - 1
  2. Voxel shapes and dtype
  3. Non-empty voxels (event activity statistics)
  4. Visual alignment: depth edges overlaid with event activity

For alignment: voxel i contains events between frame i-1 and frame i,
so depth[i] and voxel[i] represent the same scene moment.

Usage:
  python script/verify_tartan_event.py \
      --seq /tmp/tartan_test/ocean_Hard_P002 \
      [--every 50] \
      [--outdir /tmp/verify_viz]
"""

import os
import sys
import argparse
import glob
import numpy as np
import h5py
import hdf5plugin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────

def load_voxel(h5_path):
    with h5py.File(h5_path, "r") as f:
        return f["voxel"][:]  # (5, H, W) float16


def depth_to_rgb(depth, percentile=99):
    """Colorize depth map (jet, clipped at percentile)."""
    vmax = np.percentile(depth[depth > 0], percentile) if (depth > 0).any() else 1.0
    norm = np.clip(depth / vmax, 0, 1)
    return plt.cm.turbo(norm)[..., :3]  # (H, W, 3)


def depth_edges(depth, sigma=1.0):
    """Sobel edge magnitude on log-depth."""
    from scipy.ndimage import sobel, gaussian_filter
    log_d = np.log1p(np.clip(depth, 0, None))
    smooth = gaussian_filter(log_d, sigma=sigma)
    ex = sobel(smooth, axis=1)
    ey = sobel(smooth, axis=0)
    mag = np.sqrt(ex**2 + ey**2)
    mag /= (mag.max() + 1e-8)
    return mag.astype(np.float32)


def event_activity(voxel):
    """Sum absolute values across time bins → (H, W) activity map."""
    act = np.abs(voxel.astype(np.float32)).sum(axis=0)
    act /= (act.max() + 1e-8)
    return act


def overlay(depth_rgb, act, alpha=0.6):
    """Overlay event activity (hot colormap) on depth image."""
    hot = plt.cm.hot(act)[..., :3]
    mask = act[..., None]  # use activity as alpha
    blended = depth_rgb * (1 - alpha * mask) + hot * (alpha * mask)
    return np.clip(blended, 0, 1)


def make_figure(frame_idx, depth_path, voxel_path):
    depth = np.load(depth_path)
    voxel = load_voxel(voxel_path)

    act = event_activity(voxel)
    edges = depth_edges(depth)
    depth_rgb = depth_to_rgb(depth)
    blended = overlay(depth_rgb, act)

    # Pearson correlation of activity vs edges
    flat_act = act.flatten()
    flat_edg = edges.flatten()
    corr = np.corrcoef(flat_act, flat_edg)[0, 1]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Frame {frame_idx}  |  edge-event corr={corr:.3f}", fontsize=13)

    axes[0].imshow(depth_rgb)
    axes[0].set_title("Depth (colorized)")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Depth edges")

    axes[2].imshow(act, cmap="hot")
    axes[2].set_title("Event activity |voxel|.sum(bins)")

    axes[3].imshow(blended)
    axes[3].set_title("Overlay (events on depth)")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return fig, corr, voxel, depth


# ──────────────────────────────────────────────────────────────────────────────

def run_checks(seq_dir: Path, frame_indices, outdir: Path):
    h5_dir    = seq_dir / "evs_left" / "h5"
    depth_dir = seq_dir / "depth_left"
    pose_path = seq_dir / "pose_left.txt"

    # ── 1. Existence checks ────────────────────────────────────────────────
    print("=" * 60)
    print(f"Sequence: {seq_dir}")
    ok = True
    for p, name in [(h5_dir, "evs_left/h5"), (depth_dir, "depth_left"), (pose_path, "pose_left.txt")]:
        exists = p.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name}")
        if not exists:
            ok = False
    if not ok:
        print("  Aborting — required paths missing.")
        return

    # ── 2. Count checks ───────────────────────────────────────────────────
    voxels = sorted(h5_dir.glob("*.h5"))
    depths = sorted(depth_dir.glob("*.npy"))
    poses  = np.loadtxt(pose_path)

    n_vox   = len(voxels)
    n_dep   = len(depths)
    n_poses = poses.shape[0]

    print(f"\n  Counts:")
    print(f"    voxels : {n_vox}")
    print(f"    depths : {n_dep}")
    print(f"    poses  : {n_poses}")

    expected_vox = n_dep - 1
    if n_vox == expected_vox:
        print(f"  [OK] voxels == depths - 1  ({n_vox} == {n_dep} - 1)")
    else:
        print(f"  [WARN] expected {expected_vox} voxels, got {n_vox}")

    if n_poses == n_dep:
        print(f"  [OK] poses == depths  ({n_poses})")
    else:
        print(f"  [WARN] poses {n_poses} != depths {n_dep}")

    # ── 3. Shape / dtype spot-check ────────────────────────────────────────
    print(f"\n  Voxel format (checking first, middle, last):")
    check_indices = [0, n_vox // 2, n_vox - 1]
    for ci in check_indices:
        v = load_voxel(voxels[ci])
        print(f"    [{ci:>6}] shape={v.shape}  dtype={v.dtype}"
              f"  min={v.min():.3f}  max={v.max():.3f}"
              f"  zeros={100*(v==0).mean():.1f}%")

    # ── 4. Empty voxels ────────────────────────────────────────────────────
    print(f"\n  Scanning for empty voxels (all zeros)...")
    empty_count = 0
    sample_size = min(50, n_vox)
    step = max(1, n_vox // sample_size)
    for vp in voxels[::step]:
        v = load_voxel(vp)
        if (v == 0).all():
            empty_count += 1
    print(f"    {empty_count}/{len(voxels[::step])} sampled voxels are all-zero")

    # ── 5. Index alignment: voxel[i] ↔ depth[i+1] ─────────────────────────
    # depth filename: 000001_left_depth.npy -> index 1 -> voxel 0000000001.h5
    print(f"\n  Index alignment check:")
    for ci in check_indices:
        vp = voxels[ci]
        voxel_idx = int(vp.stem)        # e.g. 1 for 0000000001.h5
        depth_idx = voxel_idx           # depth at same frame number
        depth_fname = f"{depth_idx:06d}_left_depth.npy"
        dp = depth_dir / depth_fname
        status = "OK" if dp.exists() else "MISSING"
        print(f"    voxel {vp.name} <-> {depth_fname} [{status}]")

    # ── 6. Visual alignment figures ───────────────────────────────────────
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Generating alignment figures in {outdir}...")

        corrs = []
        for fi in frame_indices:
            voxel_path = h5_dir / f"{fi:010d}.h5"
            depth_path = depth_dir / f"{fi:06d}_left_depth.npy"

            if not voxel_path.exists():
                print(f"    [SKIP] voxel {voxel_path.name} not found")
                continue
            if not depth_path.exists():
                print(f"    [SKIP] depth {depth_path.name} not found")
                continue

            fig, corr, voxel, depth = make_figure(fi, depth_path, voxel_path)
            save_path = outdir / f"frame_{fi:06d}.png"
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            corrs.append(corr)
            print(f"    frame {fi:>4d}: edge-event corr={corr:.3f}  -> {save_path.name}")

        if corrs:
            print(f"\n  Mean edge-event correlation: {np.mean(corrs):.3f}")
            print("  (>0.1 is a good sign of spatial alignment)")

    print("\n" + "=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", required=True,
                        help="Path to processed sequence dir, e.g. /tmp/tartan_test/ocean_Hard_P002")
    parser.add_argument("--every", type=int, default=50,
                        help="Save a figure every N frames (default: 50).")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory to save alignment figures. "
                             "If omitted, figures are not saved.")
    args = parser.parse_args()

    seq_dir = Path(args.seq)

    voxels = sorted((seq_dir / "evs_left" / "h5").glob("*.h5"))
    if not voxels:
        print("No voxel files found — run prepare_tartan_event.py first.")
        sys.exit(1)

    # every N frames, 1-indexed
    all_indices = [int(v.stem) for v in voxels]
    frame_indices = all_indices[::args.every]

    outdir = Path(args.outdir) if args.outdir else None

    run_checks(seq_dir, frame_indices, outdir)


if __name__ == "__main__":
    main()
